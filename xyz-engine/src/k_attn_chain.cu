// The engine's attention-layer chains around the cache: the Q chain (RMS norm * q_norm, M-RoPE, the KV rotation's
// 256-point Hadamard, xyzkv's forward 128-point WHT) and the K / V cache writes (the same prologue for K, the V
// rotation's 64-point Hadamard for V, then the xyzkv2 quantizer of each 128-group into the cache row the token's index
// names). One CTA of 256 threads per (head, token) row; the rotations run in registers (lane stages by shuffle, the wider
// ones through shared memory) in the fork's stage order, so every value is the fork's bits (tools/attn_test.cu).
// InnerQ is disabled in the served cache-write path.
#include "had.cuh"
#include "xyzkv2.cuh"
#include "yarn.cuh"
#include "ggml.h"

#include "kernels.h"

namespace eng {

namespace {

// One (head, token) row of 256 at x: RMS norm * w (one element per thread, a 256-thread block reduction), M-RoPE (thread
// p < 128 rotates pair p in s), then the 256-point Hadamard (x * scale, stages h = 1..128). Returns element tid; s is free.
__device__ __forceinline__ float head_prologue(float * s, float * s_sum, const float * x, const float * w, const float eps,
                                               const int32_t * pos, const int tok, const int ne02, const MRope & rp,
                                               const float fwht_scale) {
    constexpr int N = 256;
    const int tid = threadIdx.x;

    const float xi = x[tid];
    float tmp = 0.0f;
    tmp += xi * xi;
    tmp = block_reduce<block_reduce_method::SUM, N>(tmp, s_sum);
    const float mean  = tmp / N;
    const float scale = rsqrtf(mean + eps);
    s[tid] = scale * xi * w[tid];
    __syncthreads();

    if (tid < N/2) {
        const int i0 = 2*tid;
        if (!(i0 < rp.n_offs || i0 >= rp.n_offs + rp.n_dims)) {
            const int iw = i0 - rp.n_offs;
            const int sect_dims = rp.sections[0] + rp.sections[1] + rp.sections[2] + rp.sections[3];
            const int sec_w = rp.sections[1] + rp.sections[0];
            const int sector = (iw / 2) % sect_dims;

            float theta_base = 0.0;
            if (rp.is_imrope) {
                if (sector % 3 == 1 && sector < 3 * rp.sections[1]) {
                    theta_base = pos[tok + ne02 * 1] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector % 3 == 2 && sector < 3 * rp.sections[2]) {
                    theta_base = pos[tok + ne02 * 2] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector % 3 == 0 && sector < 3 * rp.sections[0]) {
                    theta_base = pos[tok] * powf(rp.theta_scale, iw / 2.0f);
                } else {
                    theta_base = pos[tok + ne02 * 3] * powf(rp.theta_scale, iw / 2.0f);
                }
            } else {
                if (sector < rp.sections[0]) {
                    theta_base = pos[tok] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= rp.sections[0] && sector < sec_w) {
                    theta_base = pos[tok + ne02 * 1] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= sec_w && sector < sec_w + rp.sections[2]) {
                    theta_base = pos[tok + ne02 * 2] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= sec_w + rp.sections[2]) {
                    theta_base = pos[tok + ne02 * 3] * powf(rp.theta_scale, iw / 2.0f);
                }
            }

            const float freq_factor = 1.0f;
            float cos_theta;
            float sin_theta;
            yarn(theta_base/freq_factor, rp.freq_scale, rp.corr_dims[0], rp.corr_dims[1], iw, rp.ext_factor,
                 rp.attn_factor, cos_theta, sin_theta);

            const int a = i0/2 + rp.n_offs/2;
            const int b = a + rp.n_dims/2;
            const float x0 = s[a];
            const float x1 = s[b];
            s[a] = x0*cos_theta - x1*sin_theta;
            s[b] = x0*sin_theta + x1*cos_theta;
        }
    }
    __syncthreads();

    float v[1] = { s[tid] * fwht_scale };
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(v, h, tid % WARP_SIZE);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        had_stage_smem<1, N>(v, s, h, tid);
    }
    return v[0];
}

// One 128-group into its xyzkv2 block: thread j of the group holds element j in v; both groups of the CTA call it
// together (CTA-wide syncs); s: the group's 128 floats of shared memory. The L2 norm (warp butterflies, then the 4 warp
// sums in order), normalize, xyzkv's forward WHT (* S1, stages h = 1..64, * 1/sqrt(128) * S2), the nearest
// centroid, 4 indices per byte, the reconstruction's norm the same way, and the corrected norm grp / recon.
__device__ __forceinline__ void xyzkv2_store128(float v, const int j, block_xyzkv2_0 * blk, float * s, float * warp_accum,
                                                float * s_norm_sq, float * s_recon_sq) {
    constexpr int n_warps = 128 / WARP_SIZE;

    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    }
    if (j % WARP_SIZE == 0) {
        warp_accum[j / WARP_SIZE] = v2;
    }
    __syncthreads();
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        *s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(*s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    float r[1] = { v };
    r[0] *= inv_norm;
    r[0] *= xyzkv_s1[j];
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(r, h, j % WARP_SIZE);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < 128; h *= 2) {
        had_stage_smem<1, 128>(r, s, h, j);
    }
    constexpr float inv_sqrt_group = 0.08838834764831845f;
    const float rv = r[0] * inv_sqrt_group * xyzkv_s2[j];

    const uint8_t idx = xyzkv2_nearest(rv);
    const int lane = j % WARP_SIZE;
    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) {
        blk->qs[j / 4] = qs_byte;
    }

    const float c = xyzkv2_centroid[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    }
    if (j % WARP_SIZE == 0) {
        warp_accum[j / WARP_SIZE] = rc;
    }
    __syncthreads();
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        *s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(*s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

// the Q chain: dst [256, n_heads, tokens], row = head + token*n_heads
__launch_bounds__(256, 1)
__global__ void k_attn_q(const float * __restrict__ x, const int64_t sx1, const int64_t sx2, const float * __restrict__ w,
                         const float eps, const int32_t * __restrict__ pos, const int ne02, const MRope rp,
                         const float fwht_scale, const float * __restrict__ scale_inv, float * __restrict__ dst,
                         const int n_heads) {
    constexpr int N = 256;
    __shared__ float s[N];
    __shared__ float s_sum[32];

    const int row  = blockIdx.x;
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int tid  = threadIdx.x;

    ggml_cuda_pdl_sync();
    float v[1] = { head_prologue(s, s_sum, x + tok*sx2 + head*sx1, w, eps, pos, tok, ne02, rp, fwht_scale) };

    // xyzkv's forward WHT on each 128-half: * scale_inv, * S1, stages h = 1..64, * 1/sqrt(128) * S2
    const int t = tid % 128;
    if (scale_inv != nullptr) {
        v[0] *= scale_inv[t];
    }
    v[0] *= xyzkv_s1[t];
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(v, h, tid % WARP_SIZE);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < 128; h *= 2) {
        had_stage_smem<1, N>(v, s, h, tid);
    }
    constexpr float inv_sqrt = 0.08838834764831845f;
    dst[(int64_t) row*N + tid] = v[0] * inv_sqrt * xyzkv_s2[t];
}

// K: the prologue, then groups head*2 + {0, 1} of the token's K row into cache row idx[tok]
__launch_bounds__(256, 1)
__global__ void k_attn_k_write(const float * __restrict__ x, const int64_t sx1, const int64_t sx2, const float * __restrict__ w,
                               const float eps, const int32_t * __restrict__ pos, const int ne02, const MRope rp,
                               const float fwht_scale, const int64_t * __restrict__ idx, char * __restrict__ dst,
                               const int64_t nb1, const int n_heads) {
    static_assert(QK_XYZKV2 == 128, "one xyzkv2 block per 128-group");
    __shared__ float s[256];
    __shared__ float s_sum[32];
    __shared__ float warp_accum[2][4];
    __shared__ float s_norm_sq[2];
    __shared__ float s_recon_sq[2];

    const int row  = blockIdx.x;
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int half = threadIdx.x / 128;

    ggml_cuda_pdl_sync();
    const float v = head_prologue(s, s_sum, x + tok*sx2 + head*sx1, w, eps, pos, tok, ne02, rp, fwht_scale);

    const int64_t dst_row = idx[tok];
    block_xyzkv2_0 * blk = (block_xyzkv2_0 *) (dst + dst_row*nb1) + head*2 + half;
    xyzkv2_store128(v, threadIdx.x % 128, blk, s + half*128, warp_accum[half], &s_norm_sq[half], &s_recon_sq[half]);
}

// V: the V rotation's 64-point Hadamard on the head's four 64-chunks (x * scale, stages h = 1..32), then the two groups
__launch_bounds__(256, 1)
__global__ void k_attn_v_write(const float * __restrict__ v, const int64_t sv2, const float fwht_scale,
                               const int64_t * __restrict__ idx, char * __restrict__ dst, const int64_t nb1,
                               const int n_heads) {
    static_assert(QK_XYZKV2 == 128, "one xyzkv2 block per 128-group");
    __shared__ float s[256];
    __shared__ float warp_accum[2][4];
    __shared__ float s_norm_sq[2];
    __shared__ float s_recon_sq[2];

    const int row  = blockIdx.x;
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int tid  = threadIdx.x;
    const int half = tid / 128;

    ggml_cuda_pdl_sync();
    float r[1] = { v[tok*sv2 + head*256 + tid] * fwht_scale };
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(r, h, tid % WARP_SIZE);
    }
    had_stage_smem<1, 256>(r, s, WARP_SIZE, tid);

    const int64_t dst_row = idx[tok];
    block_xyzkv2_0 * blk = (block_xyzkv2_0 *) (dst + dst_row*nb1) + head*2 + half;
    xyzkv2_store128(r[0], tid % 128, blk, s + half*128, warp_accum[half], &s_norm_sq[half], &s_recon_sq[half]);
}

} // namespace

MRope mrope_params(int n_dims, const int sections[4], int mode, int n_ctx_orig, float freq_base, float freq_scale,
                   float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const bool is_mrope  = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    if (!is_mrope || is_vision || n_dims % 2 != 0 || n_dims > 256) {
        fprintf(stderr, "mrope_params: mode %d n_dims %d is not an M-RoPE the 256-wide head prologue handles\n", mode, n_dims);
        abort();
    }
    MRope p = {};
    p.n_dims      = n_dims;
    p.n_offs      = 0;   // ggml_rope_multi's (ggml_rope_set_offset is never applied)
    p.freq_scale  = freq_scale;
    p.ext_factor  = ext_factor;
    p.attn_factor = attn_factor;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, p.corr_dims);
    p.theta_scale = powf(freq_base, -2.0f / n_dims);   // rope_multi_cuda's, on the host
    for (int i = 0; i < 4; ++i) {
        p.sections[i] = sections[i];
    }
    p.is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    return p;
}

void attn_q_chain(cudaStream_t st, const float * x, int64_t sx1, int64_t sx2, const float * w, float eps, const int32_t * pos,
                  int n_tokens, const MRope & rp, const float * scale_inv, float * dst, int n_heads) {
    const float fwht_scale = 1 / sqrtf(256);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_heads * n_tokens), 1, 1), dim3(256, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_attn_q, lp, x, sx1, sx2, w, eps, pos, n_tokens, rp, fwht_scale, scale_inv, dst, n_heads);
}

void attn_k_write(cudaStream_t st, const float * x, int64_t sx1, int64_t sx2, const float * w, float eps, const int32_t * pos,
                  int n_tokens, const MRope & rp, const int64_t * idx, char * cache, int64_t row_bytes, int n_heads) {
    const float fwht_scale = 1 / sqrtf(256);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_heads * n_tokens), 1, 1), dim3(256, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_attn_k_write, lp, x, sx1, sx2, w, eps, pos, n_tokens, rp, fwht_scale, idx, cache, row_bytes,
        n_heads);
}

void attn_v_write(cudaStream_t st, const float * v, int64_t sv2, const int64_t * idx, char * cache, int64_t row_bytes,
                  int n_heads, int n_tokens) {
    const float fwht_scale = 1 / sqrtf(64);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_heads * n_tokens), 1, 1), dim3(256, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_attn_v_write, lp, v, sv2, fwht_scale, idx, cache, row_bytes, n_heads);
}

} // namespace eng
