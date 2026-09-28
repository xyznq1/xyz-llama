#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// The scalar-gate token loop preloads token t+1 while processing token t.
template <int S_v, bool KDA, bool keep_rs_t, bool pf = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const float * prefix,
                                     const int32_t * prefix_n,
                                     int           n_prev,
                                     int64_t       pack_row,
                                     float *       pack_out,
                                     int           flags,
                                     int64_t       H_k,
                                     int64_t       prefix_sstride,
                                     const int32_t * state_idx,
                                     const int32_t * prefix_idx) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // The initial state is row state_idx[sequence] of the recurrent cache.
    // GET_ROWS gather in front of this op was skipped (ggml_cuda_try_fuse). Rows are D = H * S_v * S_v floats.
    const int64_t state_in_offset      = (state_idx ? (int64_t) state_idx[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

// Prefix replay re-runs the recurrence over n_prev earlier
    // tokens, packed per (seq, token), with no attention output. Same arithmetic, same order as the
    // main loop below, so the state after the prefix is bit-identical to what a full pass would
    // have held after those tokens. Then, if asked, publish it as the committed state (slot 0).
    const int64_t pk_off_q = 0;
    const int64_t pk_off_k = H_k * S_v;
    const int64_t pk_off_v = 2 * H_k * S_v;
    const int64_t pk_off_g = 2 * H_k * S_v + H * S_v;
    const int64_t pk_off_b = pk_off_g + (KDA ? H * S_v : H);
    const int n_prev_seq = prefix_n ? prefix_n[sequence] : n_prev; // per-sequence runtime count (fixed-shape prefix)
    // prefix_idx selects a pack row directly when its gather was skipped.
    const int64_t prefix_row = prefix_idx ? (int64_t) prefix_idx[sequence] : (int64_t) sequence;
    if constexpr (pf && !KDA) {
        // the prefix loop below, pipelined: token t+1's k, v[col], g, beta are in flight while token t runs
        // (own names: EDG lost the later function-scope k_nx while this block's k_nx existed -- "k_nx is undefined")
        float kp_nx[rows_per_lane];
        float vp_nx = 0.0f, gp_nx = 0.0f, bp_nx = 0.0f;
        if (n_prev_seq > 0) {
            const float * base = prefix + prefix_row * prefix_sstride;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kp_nx[r] = base[pk_off_k + iq1 * S_v + r * warp_size + lane];
            }
            vp_nx = base[pk_off_v + h_idx * S_v + col];
            gp_nx = base[pk_off_g + h_idx];
            bp_nx = base[pk_off_b + h_idx];
        }
        for (int t = 0; t < n_prev_seq; t++) {
            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = kp_nx[r];
            }
            const float v_val    = vp_nx;
            const float g_raw    = gp_nx;
            const float beta_val = bp_nx;
            if (t + 1 < n_prev_seq) {
                const float * base = prefix + prefix_row * prefix_sstride + (int64_t) (t + 1) * pack_row;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kp_nx[r] = base[pk_off_k + iq1 * S_v + r * warp_size + lane];
                }
                vp_nx = base[pk_off_v + h_idx * S_v + col];
                gp_nx = base[pk_off_g + h_idx];
                bp_nx = base[pk_off_b + h_idx];
            }
            const float g_val = expf(g_raw);
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard = __fmaf_rn(s_shard[r], k_reg[r], kv_shard); // pinned: identical rounding in prefix and main loops
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col, v_val), beta_val); // pinned
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = __fmaf_rn(g_val, s_shard[r], __fmul_rn(k_reg[r], delta_col)); // pinned
            }
        }
    } else {
    for (int t = 0; t < n_prev_seq; t++) {
        const float * base   = prefix + prefix_row * prefix_sstride + (int64_t) t * pack_row;
        const float * k_t    = base + pk_off_k + iq1 * S_v;
        const float * v_t    = base + pk_off_v + h_idx * S_v;
        const float * g_t    = base + pk_off_g + (KDA ? h_idx * S_v : h_idx);
        const float beta_val = base[pk_off_b + h_idx];
        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_t[r * warp_size + lane];
        }
        if constexpr (!KDA) {
            const float g_val = expf(*g_t);
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard = __fmaf_rn(s_shard[r], k_reg[r], kv_shard); // pinned: identical rounding in prefix and main loops
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col, v_t[col]), beta_val); // pinned
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = __fmaf_rn(g_val, s_shard[r], __fmul_rn(k_reg[r], delta_col)); // pinned
            }
        } else {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard = __fmaf_rn(__fmul_rn(expf(g_t[i]), s_shard[r]), k_reg[r], kv_shard); // pinned
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = __fmul_rn(__fsub_rn(v_t[col], kv_col), beta_val); // pinned
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = __fmaf_rn(expf(g_t[i]), s_shard[r], __fmul_rn(k_reg[r], delta_col)); // pinned
            }
        }
    }
    }   // else (plain prefix loop)
    if (flags & GGML_GDN_COMMIT_AFTER_PREFIX) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }

    // pf: token t+1's inputs, loaded while token t runs (the plain path loads each token's at the top of its iteration)
    float k_nx[rows_per_lane];
    float q_nx[rows_per_lane];
    float v_nx = 0.0f, g_nx = 0.0f, b_nx = 0.0f;
    GGML_UNUSED(v_nx); GGML_UNUSED(g_nx); GGML_UNUSED(b_nx);
    if constexpr (pf && !KDA) {
        if (n_tokens > 0) {
            const float * q_0 = q + iq3 * sq3 + iq1 * sq1;
            const float * k_0 = k + iq3 * sq3 + iq1 * sq1;
            const int64_t gb0 = sequence * sb3 + h_idx * sb1;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_nx[r] = k_0[r * warp_size + lane];
                q_nx[r] = q_0[r * warp_size + lane];
            }
            v_nx = v[sequence * sv3 + h_idx * sv1 + col];
            g_nx = g[gb0];
            b_nx = beta[gb0];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float v_col_pf = 0.0f, g_raw_pf = 0.0f;
        GGML_UNUSED(v_col_pf); GGML_UNUSED(g_raw_pf);
        if constexpr (pf && !KDA) {
            beta_val = b_nx;
            v_col_pf = v_nx;
            g_raw_pf = g_nx;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_nx[r];
                q_reg[r] = q_nx[r];
            }
            if (t + 1 < n_tokens) {
                const int64_t gb1 = sequence * sb3 + (t + 1) * sb2 + h_idx * sb1;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_nx[r] = k_t[sq2 + r * warp_size + lane];
                    q_nx[r] = q_t[sq2 + r * warp_size + lane];
                }
                v_nx = v_t[sv2 + col];
                g_nx = g[gb1];
                b_nx = beta[gb1];
            }
        } else {
            beta_val = *beta_t;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(pf ? g_raw_pf : *g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard = __fmaf_rn(s_shard[r], k_reg[r], kv_shard); // pinned: identical rounding in prefix and main loops
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col, pf ? v_col_pf : v_t[col]), beta_val); // pinned

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = __fmaf_rn(g_val, s_shard[r], __fmul_rn(k_reg[r], delta_col)); // pinned
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard = __fmaf_rn(__fmul_rn(expf(g_t[i]), s_shard[r]), k_reg[r], kv_shard); // pinned
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = __fmul_rn(__fsub_rn(v_t[col], kv_col), beta_val); // pinned

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = __fmaf_rn(expf(g_t[i]), s_shard[r], __fmul_rn(k_reg[r], delta_col)); // pinned
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if ((flags & GGML_GDN_WRITE_PACK) && col == 0) {
            // one warp per (seq, head) publishes this token's inputs for next round's prefix;
            // q/k belong to the shared k-head iq1 and are written once, by its first v-head
            float * base = pack_out + ((int64_t) sequence * n_tokens + t) * pack_row;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                base[pk_off_v + h_idx * S_v + i] = v_t[i];
                if (h_idx < H_k) {
                    base[pk_off_q + h_idx * S_v + i] = q_reg[r];
                    base[pk_off_k + h_idx * S_v + i] = k_reg[r];
                }
            }
            if constexpr (KDA) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    base[pk_off_g + h_idx * S_v + i] = g_t[i];
                }
            } else if (lane == 0) {
                base[pk_off_g + h_idx] = *g_t;
            }
            if (lane == 0) {
                base[pk_off_b + h_idx] = beta_val;
            }
        }
        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (!(flags & GGML_GDN_COMMIT_AFTER_PREFIX)) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * warp_size + lane;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }
}

// Process multiple columns per warp while preserving each recurrence chain's operation order.
template <int S_v, int NC>
__device__ __forceinline__ void gated_delta_net_cuda_2col_task(const float * q,
                          const float * k,
                          const float * v,
                          const float * g,
                          const float * beta,
                          const float * curr_state,
                          float *       dst,
                          float *       state,
                          int64_t       H,
                          int64_t       n_tokens,
                          int64_t       n_seqs,
                          int64_t       sq1,
                          int64_t       sq2,
                          int64_t       sq3,
                          int64_t       sv1,
                          int64_t       sv2,
                          int64_t       sv3,
                          int64_t       sb1,
                          int64_t       sb2,
                          int64_t       sb3,
                          const uint3   neqk1_magic,
                          const uint3   rq3_magic,
                          float         scale,
                          const float * prefix,
                          const int32_t * prefix_n,
                          int           n_prev,
                          int64_t       pack_row,
                          float *       pack_out,
                          int           flags,
                          int64_t       H_k,
                          int64_t       prefix_sstride,
                          const int32_t * state_idx,
                          const int32_t * prefix_idx,
                          const uint32_t h_idx,
                          const uint32_t sequence,
                          const int      lane,
                          const int      col0) {

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = (state_idx ? (int64_t) state_idx[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float s_shard[NC][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[(col0 + c) * S_v + r * warp_size + lane];
        }
    }

    // the prefix replay, pipelined (the pf kernel's loop, two columns)
    const int64_t pk_off_q = 0;
    const int64_t pk_off_k = H_k * S_v;
    const int64_t pk_off_v = 2 * H_k * S_v;
    const int64_t pk_off_g = 2 * H_k * S_v + H * S_v;
    const int64_t pk_off_b = pk_off_g + H;
    const int n_prev_seq = prefix_n ? prefix_n[sequence] : n_prev;
    const int64_t prefix_row = prefix_idx ? (int64_t) prefix_idx[sequence] : (int64_t) sequence;
    {
        float kp_nx[rows_per_lane];
        float vp_nx[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            vp_nx[c] = 0.0f;
        }
        float gp_nx = 0.0f, bp_nx = 0.0f;
        if (n_prev_seq > 0) {
            const float * base = prefix + prefix_row * prefix_sstride;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kp_nx[r] = base[pk_off_k + iq1 * S_v + r * warp_size + lane];
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                vp_nx[c] = base[pk_off_v + h_idx * S_v + col0 + c];
            }
            gp_nx = base[pk_off_g + h_idx];
            bp_nx = base[pk_off_b + h_idx];
        }
        for (int t = 0; t < n_prev_seq; t++) {
            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = kp_nx[r];
            }
            float v_val[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                v_val[c] = vp_nx[c];
            }
            const float g_raw    = gp_nx;
            const float beta_val = bp_nx;
            if (t + 1 < n_prev_seq) {
                const float * base = prefix + prefix_row * prefix_sstride + (int64_t) (t + 1) * pack_row;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kp_nx[r] = base[pk_off_k + iq1 * S_v + r * warp_size + lane];
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    vp_nx[c] = base[pk_off_v + h_idx * S_v + col0 + c];
                }
                gp_nx = base[pk_off_g + h_idx];
                bp_nx = base[pk_off_b + h_idx];
            }
            const float g_val = expf(g_raw);
            float kv_shard[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                kv_shard[c] = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard[c] = __fmaf_rn(s_shard[c][r], k_reg[r], kv_shard[c]); // pinned, as the pf kernel
                }
            }
            float kv_col[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col[c], v_val[c]), beta_val); // pinned
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = __fmaf_rn(g_val, s_shard[c][r], __fmul_rn(k_reg[r], delta_col)); // pinned
                }
            }
        }
    }
    if (flags & GGML_GDN_COMMIT_AFTER_PREFIX) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }

    // the main loop, pipelined: token t+1's k, q, v, g, beta in flight while token t runs
    float k_nx[rows_per_lane];
    float q_nx[rows_per_lane];
    float v_nx[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        v_nx[c] = 0.0f;
    }
    float g_nx = 0.0f, b_nx = 0.0f;
    if (n_tokens > 0) {
        const float * q_0 = q + iq3 * sq3 + iq1 * sq1;
        const float * k_0 = k + iq3 * sq3 + iq1 * sq1;
        const int64_t gb0 = sequence * sb3 + h_idx * sb1;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_nx[r] = k_0[r * warp_size + lane];
            q_nx[r] = q_0[r * warp_size + lane];
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            v_nx[c] = v[sequence * sv3 + h_idx * sv1 + col0 + c];
        }
        g_nx = g[gb0];
        b_nx = beta[gb0];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * g_t = g + gb_offset;

        const float beta_val = b_nx;
        const float g_raw    = g_nx;
        float v_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            v_col[c] = v_nx[c];
        }
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_nx[r];
            q_reg[r] = q_nx[r];
        }
        if (t + 1 < n_tokens) {
            const int64_t gb1 = sequence * sb3 + (t + 1) * sb2 + h_idx * sb1;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_nx[r] = k_t[sq2 + r * warp_size + lane];
                q_nx[r] = q_t[sq2 + r * warp_size + lane];
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                v_nx[c] = v_t[sv2 + col0 + c];
            }
            g_nx = g[gb1];
            b_nx = beta[gb1];
        }

        const float g_val = expf(g_raw);

        float kv_shard[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard[c] = __fmaf_rn(s_shard[c][r], k_reg[r], kv_shard[c]); // pinned, as the pf kernel
            }
        }
        float kv_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
        }
        float attn_partial[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col[c], v_col[c]), beta_val); // pinned
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]    = __fmaf_rn(g_val, s_shard[c][r], __fmul_rn(k_reg[r], delta_col)); // pinned
                attn_partial[c] += s_shard[c][r] * q_reg[r];   // the pf kernel's exact form (contracted the same way)
            }
        }
        float attn_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            attn_col[c] = warp_reduce_sum<warp_size>(attn_partial[c]);
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }

        attn_data += S_v * H;

        if ((flags & GGML_GDN_WRITE_PACK) && col0 == 0) {
            // the warp holding column 0 publishes this token's inputs for next round's prefix (the pf kernel's col == 0)
            float * base = pack_out + ((int64_t) sequence * n_tokens + t) * pack_row;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                base[pk_off_v + h_idx * S_v + i] = v_t[i];
                if (h_idx < H_k) {
                    base[pk_off_q + h_idx * S_v + i] = q_reg[r];
                    base[pk_off_k + h_idx * S_v + i] = k_reg[r];
                }
            }
            if (lane == 0) {
                base[pk_off_g + h_idx] = *g_t;
                base[pk_off_b + h_idx] = beta_val;
            }
        }
    }

    if (!(flags & GGML_GDN_COMMIT_AFTER_PREFIX)) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

template <int S_v, int NC>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda_2col(const float * q,
                          const float * k,
                          const float * v,
                          const float * g,
                          const float * beta,
                          const float * curr_state,
                          float *       dst,
                          float *       state,
                          int64_t       H,
                          int64_t       n_tokens,
                          int64_t       n_seqs,
                          int64_t       sq1,
                          int64_t       sq2,
                          int64_t       sq3,
                          int64_t       sv1,
                          int64_t       sv2,
                          int64_t       sv3,
                          int64_t       sb1,
                          int64_t       sb2,
                          int64_t       sb3,
                          const uint3   neqk1_magic,
                          const uint3   rq3_magic,
                          float         scale,
                          const float * prefix,
                          const int32_t * prefix_n,
                          int           n_prev,
                          int64_t       pack_row,
                          float *       pack_out,
                          int           flags,
                          int64_t       H_k,
                          int64_t       prefix_sstride,
                          const int32_t * state_idx,
                          const int32_t * prefix_idx) {
    gated_delta_net_cuda_2col_task<S_v, NC>(
        q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
        prefix, prefix_n, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx,
        blockIdx.x, blockIdx.y, threadIdx.x, (blockIdx.z*blockDim.y + threadIdx.y)*NC);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const float * prefix_d, const int32_t * prefix_n_d, int n_prev, int64_t pack_row, float * pack_out, int flags, int64_t H_k,
        int64_t prefix_sstride, const int32_t * state_idx, const int32_t * prefix_idx, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
            break;
        }
        case 128: {
            if constexpr (!KDA && !keep_rs_t) {
                const dim3 grid_2c(H, n_seqs, (S_v + 2*num_warps - 1) / (2*num_warps));
                const ggml_cuda_kernel_launch_params lp_2c = ggml_cuda_kernel_launch_params(grid_2c, block_dims, 0, stream);
                ggml_cuda_kernel_launch(gated_delta_net_cuda_2col<128, 2>, lp_2c,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
                    prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
                break;
            }
            if constexpr (!KDA) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, true>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                    prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
            } else {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                    prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, H_k, prefix_sstride, state_idx, prefix_idx);
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the state gather in front of this op was skipped: read the initial state from the cache rows directly
    const int32_t * state_idx = nullptr;
    {
        const auto it = ctx.gdn_state_src.find(dst);
        if (it != ctx.gdn_state_src.end()) {
            s_d       = it->second.first;
            state_idx = it->second.second;
            ctx.gdn_state_src.erase(it);
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // prefix replay (ggml_gated_delta_net_ext): src[6] packed prefix, op_params[1] n_prev,
    // op_params[2] flags; the pack region sits after the K state slots in dst
    const ggml_tensor * src_prefix = dst->src[6];
    const int     n_prev   = ggml_get_op_params_i32(dst, 1);
    const int     flags    = ggml_get_op_params_i32(dst, 2);
    const int64_t pack_row = ggml_gated_delta_net_pack_row(neq0, neqk1, S_v, H, kda);
    const float * prefix_d = src_prefix ? (const float *) src_prefix->data : nullptr;
    const ggml_tensor * src_prefix_n = dst->src[7]; // I32 [n_seqs] per-sequence replay counts, or null
    const int32_t * prefix_n_d = src_prefix_n ? (const int32_t *) src_prefix_n->data : nullptr;
    int64_t prefix_sstride = src_prefix ? (int64_t) (src_prefix->nb[2] / sizeof(float)) : 0;   // seq stride: the pack row may be padded
    // Read prefix_idx[seq] directly when the pack-row gather was skipped.
    const int32_t * prefix_idx = nullptr;
    {
        const auto it = ctx.gdn_prefix_src.find(dst);
        if (it != ctx.gdn_prefix_src.end()) {
            prefix_d       = it->second.table;
            prefix_idx     = it->second.idx;
            prefix_sstride = it->second.row_floats;
            ctx.gdn_prefix_src.erase(it);
        }
    }
    float *       pack_out = (flags & GGML_GDN_WRITE_PACK)
        ? dst_d + S_v * H * n_tokens * n_seqs + (int64_t) K * S_v * S_v * H * n_seqs
        : nullptr;
    GGML_ASSERT(n_prev == 0 || (src_prefix != nullptr && !keep_rs));

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, neqk1, prefix_sstride, state_idx, prefix_idx, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, neqk1, prefix_sstride, state_idx, prefix_idx, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, neqk1, prefix_sstride, state_idx, prefix_idx, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                prefix_d, prefix_n_d, n_prev, pack_row, pack_out, flags, neqk1, prefix_sstride, state_idx, prefix_idx, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
