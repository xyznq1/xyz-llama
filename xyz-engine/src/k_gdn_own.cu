// The engine's own gated delta-net recurrence for qwen35: S_v 128, 48 value heads over 16 key heads (key head h % 16),
// one sequence, the optional prefix replay (packed tokens). Armed (the verify): the state committed after the replay, this
// batch's inputs packed for the next round. Prompt (a prefill chunk: GGML_GDN flags 0, the fused cache): the state
// committed after the whole chunk, in place. Two state columns per warp, four warps per CTA, grid (48, 1, 16).
// Bit-identical to the fork's gated_delta_net_cuda_2col<128, 2> (gated_delta_net.cu, the served instance): the same
// loads, the same pinned __fmaf_rn / __fmul_rn forms, the same warp reductions, the same contracted output dot.
// Also the prompt's conv section (k_gdn_conv_prompt).
#include "had.cuh"   // act_silu

#include "kernels.h"

namespace eng {

namespace {

constexpr int GS_V  = 128;               // state width (value and key head dim)
constexpr int GH    = 48;                // value heads
constexpr int GH_K  = 16;                // key heads
constexpr int GNC   = 2;                 // state columns per warp
constexpr int GRPL  = GS_V / WARP_SIZE;  // state rows per lane
constexpr int GNW   = 4;                 // warps per CTA

// the packed row's fields (ggml_gated_delta_net_pack_row(128, 16, 128, 48, false)): q[16][128], k[16][128], v[48][128],
// g[48], beta[48]
constexpr int64_t PK_Q = 0;
constexpr int64_t PK_K = GH_K*GS_V;
constexpr int64_t PK_V = 2*GH_K*GS_V;
constexpr int64_t PK_G = 2*GH_K*GS_V + GH*GS_V;
constexpr int64_t PK_B = PK_G + GH;

template <bool armed>
__launch_bounds__(WARP_SIZE*GNW, 2)
static __global__ void k_gdn_own(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
                                 const float * __restrict__ g, const float * __restrict__ beta,
                                 const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
                                 const int n_tokens, const int64_t sq1, const int64_t sq2, const int64_t sv1, const int64_t sv2,
                                 const float scale, const float * __restrict__ prefix, const int32_t * __restrict__ prefix_n,
                                 const int64_t pack_row, float * __restrict__ pack_out, const int write_pack) {
    const int h    = blockIdx.x;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z*blockDim.y + threadIdx.y)*GNC;
    const int iq1  = h % GH_K;

    state      += (int64_t) h*GS_V*GS_V;
    curr_state += (int64_t) h*GS_V*GS_V;
    float * attn_data = dst + (int64_t) h*GS_V;

    float s_shard[GNC][GRPL];
#pragma unroll
    for (int c = 0; c < GNC; c++) {
#pragma unroll
        for (int r = 0; r < GRPL; r++) {
            s_shard[c][r] = curr_state[(col0 + c)*GS_V + r*WARP_SIZE + lane];
        }
    }

    // the prefix replay, pipelined: token t+1's packed inputs in flight while token t updates the state
    int n_prev;
    if constexpr (armed) {
        n_prev = prefix_n[0];
    } else {
        n_prev = prefix_n ? prefix_n[0] : 0;
    }
    {
        float kp_nx[GRPL];
        float vp_nx[GNC] = { 0.0f, 0.0f };
        float gp_nx = 0.0f, bp_nx = 0.0f;
        if (n_prev > 0) {
            const float * base = prefix;
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                kp_nx[r] = base[PK_K + iq1*GS_V + r*WARP_SIZE + lane];
            }
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                vp_nx[c] = base[PK_V + h*GS_V + col0 + c];
            }
            gp_nx = base[PK_G + h];
            bp_nx = base[PK_B + h];
        }
        for (int t = 0; t < n_prev; t++) {
            float k_reg[GRPL];
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                k_reg[r] = kp_nx[r];
            }
            float v_val[GNC];
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                v_val[c] = vp_nx[c];
            }
            const float g_raw    = gp_nx;
            const float beta_val = bp_nx;
            if (t + 1 < n_prev) {
                const float * base = prefix + (int64_t) (t + 1)*pack_row;
#pragma unroll
                for (int r = 0; r < GRPL; r++) {
                    kp_nx[r] = base[PK_K + iq1*GS_V + r*WARP_SIZE + lane];
                }
#pragma unroll
                for (int c = 0; c < GNC; c++) {
                    vp_nx[c] = base[PK_V + h*GS_V + col0 + c];
                }
                gp_nx = base[PK_G + h];
                bp_nx = base[PK_B + h];
            }
            const float g_val = expf(g_raw);
            float kv_shard[GNC];
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                kv_shard[c] = 0.0f;
#pragma unroll
                for (int r = 0; r < GRPL; r++) {
                    kv_shard[c] = __fmaf_rn(s_shard[c][r], k_reg[r], kv_shard[c]);
                }
            }
            float kv_col[GNC];
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                kv_col[c] = warp_reduce_sum<WARP_SIZE>(kv_shard[c]);
            }
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col[c], v_val[c]), beta_val);
#pragma unroll
                for (int r = 0; r < GRPL; r++) {
                    s_shard[c][r] = __fmaf_rn(g_val, s_shard[c][r], __fmul_rn(k_reg[r], delta_col));
                }
            }
        }
    }
    // armed: commit the state after the replay
    if constexpr (armed) {
#pragma unroll
        for (int c = 0; c < GNC; c++) {
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                state[(col0 + c)*GS_V + r*WARP_SIZE + lane] = s_shard[c][r];
            }
        }
    }

    // this batch, pipelined: token t+1's k, q, v, g, beta in flight while token t runs
    float k_nx[GRPL];
    float q_nx[GRPL];
    float v_nx[GNC] = { 0.0f, 0.0f };
    float g_nx = 0.0f, b_nx = 0.0f;
    if (n_tokens > 0) {
        const float * q_0 = q + iq1*sq1;
        const float * k_0 = k + iq1*sq1;
#pragma unroll
        for (int r = 0; r < GRPL; r++) {
            k_nx[r] = k_0[r*WARP_SIZE + lane];
            q_nx[r] = q_0[r*WARP_SIZE + lane];
        }
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            v_nx[c] = v[h*sv1 + col0 + c];
        }
        g_nx = g[h];
        b_nx = beta[h];
    }
    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + t*sq2 + iq1*sq1;
        const float * k_t = k + t*sq2 + iq1*sq1;
        const float * v_t = v + t*sv2 + h*sv1;
        const float * g_t = g + t*GH + h;

        const float beta_val = b_nx;
        const float g_raw    = g_nx;
        float v_col[GNC];
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            v_col[c] = v_nx[c];
        }
        float k_reg[GRPL];
        float q_reg[GRPL];
#pragma unroll
        for (int r = 0; r < GRPL; r++) {
            k_reg[r] = k_nx[r];
            q_reg[r] = q_nx[r];
        }
        if (t + 1 < n_tokens) {
            const int64_t gb1 = (int64_t) (t + 1)*GH + h;
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                k_nx[r] = k_t[sq2 + r*WARP_SIZE + lane];
                q_nx[r] = q_t[sq2 + r*WARP_SIZE + lane];
            }
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                v_nx[c] = v_t[sv2 + col0 + c];
            }
            g_nx = g[gb1];
            b_nx = beta[gb1];
        }

        const float g_val = expf(g_raw);

        float kv_shard[GNC];
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                kv_shard[c] = __fmaf_rn(s_shard[c][r], k_reg[r], kv_shard[c]);
            }
        }
        float kv_col[GNC];
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            kv_col[c] = warp_reduce_sum<WARP_SIZE>(kv_shard[c]);
        }
        float attn_partial[GNC];
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            const float delta_col = __fmul_rn(__fmaf_rn(-g_val, kv_col[c], v_col[c]), beta_val);
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                s_shard[c][r]    = __fmaf_rn(g_val, s_shard[c][r], __fmul_rn(k_reg[r], delta_col));
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }
        float attn_col[GNC];
#pragma unroll
        for (int c = 0; c < GNC; c++) {
            attn_col[c] = warp_reduce_sum<WARP_SIZE>(attn_partial[c]);
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < GNC; c++) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }
        attn_data += GS_V*GH;

        if (write_pack && col0 == 0) {
            float * base = pack_out + (int64_t) t*pack_row;
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                const int i = r*WARP_SIZE + lane;
                base[PK_V + h*GS_V + i] = v_t[i];
                if (h < GH_K) {
                    base[PK_Q + h*GS_V + i] = q_reg[r];
                    base[PK_K + h*GS_V + i] = k_reg[r];
                }
            }
            if (lane == 0) {
                base[PK_G + h] = *g_t;
                base[PK_B + h] = beta_val;
            }
        }
    }

    // prompt: commit the state after the chunk (in place: each thread rewrites exactly the shard it read)
    if constexpr (!armed) {
#pragma unroll
        for (int c = 0; c < GNC; c++) {
#pragma unroll
            for (int r = 0; r < GRPL; r++) {
                state[(col0 + c)*GS_V + r*WARP_SIZE + lane] = s_shard[c][r];
            }
        }
    }
}

// The prompt's conv section as the server's prompt graph runs it -- build_conv_state's window (concat), the last 3 steps
// back into the state (cpy), ssm_conv + silu, the q/k heads' L2 norms (l2_norm) -- in one launch, each value by the fork
// kernels' statements (ssm-conv.cu ssm_conv_f32 / ssm_conv_long_token_f32: sum = 0, += x*w per tap, += the absent bias,
// silu; norm.cu l2_norm_f32<32>: four squares per lane in order, the warp sum, rsqrtf(fmaxf(sum, eps*eps)) * x). The
// window of channel c: 3 old steps -- the state's rows, or with conv_idx rows conv_idx[0..2] of [state(3) ++ pack(5)] --
// then this chunk's n inputs. Grid (80 channel blocks of 128, 32-token chunks); the first chunk's threads read the old
// steps and then commit steps n..n+2 of the window for their own channel (no other thread touches that channel's state).
constexpr int CC   = 10240;   // conv channels (q 16 heads, k 16 heads, v 48 heads of 128)
constexpr int CCPB = 128;     // channels per block = one head
constexpr int CTCH = 32;      // tokens per chunk
constexpr int CQKB = 32;      // q/k heads (blocks 0..31)

__launch_bounds__(CCPB, 1)
static __global__ void k_gdn_conv_prompt(const float * state, const float * __restrict__ pack,
                                         const int32_t * __restrict__ conv_idx, const float * __restrict__ qkv, const int n,
                                         const float * __restrict__ conv_w, const float bias0, float * state_out,
                                         float * __restrict__ conv_silu, float * __restrict__ qk_norm, const float eps) {
    __shared__ float sh[CTCH][CCPB];
    const int tid = threadIdx.x;
    const int c   = blockIdx.x*CCPB + tid;
    const int t0  = blockIdx.y*CTCH;
    const int nt  = min(CTCH, n - t0);

    // step j of the window [old(3) ++ qkv(n)]
    const auto step = [&](const int j) -> float {
        if (j >= 3) {
            return qkv[(int64_t) (j - 3)*CC + c];
        }
        if (conv_idx == nullptr) {
            return state[j*CC + c];
        }
        const int r = conv_idx[j];
        return r < 3 ? state[r*CC + c] : pack[(r - 3)*CC + c];
    };

    ggml_cuda_pdl_sync();
    float w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = conv_w[c*4 + j];
    }
    float x[4];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        x[j] = step(t0 + j);
    }
    if (blockIdx.y == 0) {   // commit: the window's last 3 steps (read before written)
        float nw[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            nw[s] = step(n + s);
        }
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            state_out[s*CC + c] = nw[s];
        }
    }
    for (int i = 0; i < nt; ++i) {
        x[3] = step(t0 + i + 3);
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += x[j] * w[j];
        }
        sumf += bias0;
        const float value = act_silu(sumf);
        conv_silu[(int64_t) (t0 + i)*CC + c] = value;
        sh[i][tid] = value;
        x[0] = x[1]; x[1] = x[2]; x[2] = x[3];
    }

    if (blockIdx.x < CQKB) {   // a q/k head: its L2 norm per token (warp w: tokens w, w + 4, ...)
        __syncthreads();
        const int lane = tid % WARP_SIZE;
        for (int i = tid / WARP_SIZE; i < nt; i += CCPB / WARP_SIZE) {
            float tmp = 0.0f;
#pragma unroll
            for (int col = lane; col < CCPB; col += WARP_SIZE) {
                const float xi = sh[i][col];
                tmp += xi * xi;
            }
            tmp = warp_reduce_sum<WARP_SIZE>(tmp);
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
            for (int col = lane; col < CCPB; col += WARP_SIZE) {
                qk_norm[((int64_t) (t0 + i)*CQKB + blockIdx.x)*CCPB + col] = scale * sh[i][col];
            }
        }
    }
}

} // namespace

void gdn_own(cudaStream_t st, const GdnArgs & a) {
    const int64_t pack_row = 2*GH_K*GS_V + GH*GS_V + GH + GH;   // ggml_gated_delta_net_pack_row(128, 16, 128, 48, false)
    const float   scale    = 1.0f / sqrtf((float) GS_V);
    const dim3 grid(GH, 1, GS_V / (GNC*GNW)), block(WARP_SIZE, GNW, 1);
    k_gdn_own<true><<<grid, block, 0, st>>>(a.q, a.k, a.v, a.g, a.beta, a.s, a.dst, a.state_out, a.n_tokens, a.sq1, a.sq2,
                                            a.sv1, a.sv2, scale, a.prefix, a.prefix_n, pack_row, a.pack_out, 1);
}

void gdn_prompt(cudaStream_t st, const GdnArgs & a) {
    const int64_t pack_row = 2*GH_K*GS_V + GH*GS_V + GH + GH;
    const float   scale    = 1.0f / sqrtf((float) GS_V);
    const dim3 grid(GH, 1, GS_V / (GNC*GNW)), block(WARP_SIZE, GNW, 1);
    k_gdn_own<false><<<grid, block, 0, st>>>(a.q, a.k, a.v, a.g, a.beta, a.s, a.dst, a.state_out, a.n_tokens, a.sq1, a.sq2,
                                             a.sv1, a.sv2, scale, a.prefix, a.prefix_n, pack_row, nullptr, 0);
}

void gdn_conv_prompt(cudaStream_t st, float * state, const float * pack, const int32_t * conv_idx, const float * qkv, int n,
                     const float * conv_w, float * conv_silu, float * qk_norm, float eps) {
    const dim3 grid(CC / CCPB, (unsigned) ((n + CTCH - 1) / CTCH), 1);
    // bias0: ssm_conv_f32's absent bias, the runtime zero it adds
    k_gdn_conv_prompt<<<grid, CCPB, 0, st>>>(state, pack, conv_idx, qkv, n, conv_w, 0.0f, state, conv_silu, qk_norm, eps);
}

} // namespace eng
