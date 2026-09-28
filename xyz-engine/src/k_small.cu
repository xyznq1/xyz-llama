// The engine's small kernels: the token embedding gather out of the ILV16 PTQ1_0 table, the 5120-wide RMS norm * w, the
// drafter's fused qk RMS norm * w + NEOX rope over 256-wide heads, its q4_0 cache write, the sigmoid and the two gated
// activations. Each is the fork kernel's arithmetic statement for statement (getrows.cu k_get_rows_ptq1_ilv, norm.cu
// rms_norm_f32<1024, true>, rope.cu rms_norm_mul_rope_f32<256, false>, set-rows.cu k_set_rows_quant + cpy-utils.cuh
// quantize_f32_q4_0_block, unary.cu unary_op_kernel<op_sigmoid> / unary_gated_op_kernel): bit-identical,
// tools/small_test.cu and tools/pchain_test.cu.
#include "had.cuh"
#include "ptq1.cuh"
#include "yarn.cuh"

#include "kernels.h"

namespace eng {

namespace {

// row ids[i] of an [n_embd, n_rows] PTQ1_0 table into dst row i; thread pairs as the fork's grid
__global__ void k_embed_ptq1(const block_ptq1_0 * __restrict__ table, const int32_t * __restrict__ ids,
                             float * __restrict__ dst, const int64_t n_embd, const int64_t n_rows) {
    ggml_cuda_pdl_sync();
    const int64_t stride_row = n_embd / QK_PTQ1_0;
    const int     i10 = blockIdx.x;
    const int64_t row = ids[i10];
    float * d = dst + (int64_t) i10*n_embd;
    for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < n_embd; i00 += gridDim.y*blockDim.x) {
        const block_ptq1_0 * x = table + ptq1_block(row, i00/QK_PTQ1_0, stride_row, n_rows);
        const int iqs = i00 % QK_PTQ1_0;
        const float s = x->d;
        d[i00 + 0] = ptq1_elem(x, iqs)     * s;
        d[i00 + 1] = ptq1_elem(x, iqs + 1) * s;
    }
}

// dst row = rms_norm(x row) * w, one CTA of 1024 per row
__launch_bounds__(1024, 1)
__global__ void k_rms_norm_mul(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ dst,
                               const int ncols, const float eps) {
    constexpr int BS = 1024;
    __shared__ float s_sum[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x   += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;

    float tmp = 0.0f;
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += BS) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += BS) {
        dst[col] = scale * x[col] * w[col];
    }
}

// head `row` of token `channel`: rms_norm * w over the 256-wide head, then NEOX rope on the first n_dims (pair i0/2 with
// i0/2 + n_dims/2), the rest passed through; one CTA of 256 per (head, token)
__launch_bounds__(256, 1)
__global__ void k_qk_norm_rope(const float * __restrict__ x, float * __restrict__ dst, const int ncols, const int64_t s01,
                               const int64_t s02, const int64_t s1, const int64_t s2, const float eps,
                               const float * __restrict__ w, const int n_dims, const int32_t * __restrict__ pos,
                               const float freq_scale, const float ext_factor, const float attn_factor, const float corr0,
                               const float corr1, const float theta_scale) {
    constexpr int BS = 256;
    __shared__ float s_sum[32];
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int tid     = threadIdx.x;
    x += channel*s02 + row*s01;

    float tmp = 0.0f;
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += BS) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float scale = rsqrtf(tmp/ncols + eps);

    dst += channel*s2 + row*s1;
    for (int i0 = 2*tid; i0 < ncols; i0 += 2*BS) {
        int ix0;
        int ix1;
        if (i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }
        const float x0 = scale * x[ix0] * w[ix0];
        const float x1 = scale * x[ix1] * w[ix1];
        if (i0 >= n_dims) {
            dst[ix0] = x0;
            dst[ix1] = x1;
            continue;
        }
        const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
        const float freq_factor = 1.0f;
        float cos_theta;
        float sin_theta;
        yarn(theta_base/freq_factor, freq_scale, corr0, corr1, i0, ext_factor, attn_factor, cos_theta, sin_theta);
        dst[ix0] = x0*cos_theta - x1*sin_theta;
        dst[ix1] = x0*sin_theta + x1*cos_theta;
    }
}

// one thread per 32-value block: row t of src (ncols floats) into cache row idx[t] as q4_0 (d = the signed max / -8,
// nibbles min(15, (int8) (x/d + 8.5)))
__global__ void k_set_rows_q4_0(const float * __restrict__ src, const int64_t * __restrict__ idx, char * __restrict__ cache,
                                const int64_t ncols, const int64_t n_blocks, const int64_t row_bytes) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const int64_t i_base = i * QK4_0;
    const int64_t t   = i_base / ncols;
    const int64_t i00 = i_base % ncols;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = idx[t];
    const float * x = src + t*ncols + i00;
    block_q4_0 * y = (block_q4_0 *) (cache + dst_row*row_bytes) + i00 / QK4_0;

    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }
    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;
    y->d = d;
    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;
        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));
        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

// dst = sigmoid(x), elementwise
__global__ void k_sigmoid(const float * __restrict__ x, float * __restrict__ dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = act_sigmoid(x[i]);
}

// dst[c*stride_col_dst + row] = W[row] . x[c]: a bf16 weight (rows stride_row apart) against f32 columns (stride_col_y2
// float2 apart) -- mmvf.cu mul_mat_vec_f<nv_bfloat16, float, ncols_dst, block_size> (no fusion, one channel and sample):
// per thread an FMA chain over its bf16 pairs, the warp sum, the per-warp sums in order through shared memory
template <int ncols_dst, int block_size>
__global__ void k_mmvf_bf16(const nv_bfloat16 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
                            const int ncols2, const int stride_row, const int stride_col_y2, const int stride_col_dst) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    x += (int64_t) row*stride_row;
    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[WARP_SIZE];
    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
        if (block_size > WARP_SIZE) {
            buf_iw[tid/WARP_SIZE] = sumf[j];
            __syncthreads();
            if (tid < WARP_SIZE) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
            }
            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }
    dst[tid*stride_col_dst + row] = sumf[tid];
}

// dst = op(gate) * x, elementwise
template <bool silu>
__global__ void k_gate(const float * __restrict__ gate, const float * __restrict__ x, float * __restrict__ dst,
                       const int64_t k) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = (silu ? act_silu(gate[i]) : act_sigmoid(gate[i])) * x[i];
}

} // namespace

void get_rows_ptq1(cudaStream_t st, const void * table, int64_t n_embd, int64_t n_rows, const int32_t * ids, int n, float * dst) {
    constexpr int BS = 256;
    const int ny = (int) ((n_embd + 2*BS - 1) / (2*BS));
    k_embed_ptq1<<<dim3((unsigned) n, (unsigned) MIN(ny, UINT16_MAX), 1), dim3(BS, 1, 1), 0, st>>>(
        (const block_ptq1_0 *) table, ids, dst, n_embd, n_rows);
}

void rms_norm_mul(cudaStream_t st, const float * x, const float * w, float * dst, int ncols, int nrows, float eps) {
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) nrows, 1, 1), dim3(1024, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_rms_norm_mul, lp, x, w, dst, ncols, eps);
}

void rms_norm_mul_rope(cudaStream_t st, const float * x, int64_t sx1, int nrows, const float * w, float eps, float * dst,
                       const int32_t * pos, int n_dims, float freq_base, int n_ctx_orig, int n_tok, int64_t sx2) {
    if (sx2 == 0) sx2 = sx1*nrows;   // one token: the view's own channel stride
    float corr[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, 32.0f, 1.0f, corr);
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) nrows, (unsigned) n_tok, 1), dim3(256, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_qk_norm_rope, lp, x, dst, 256, sx1, sx2, (int64_t) 256, (int64_t) 256*nrows, eps, w, n_dims,
        pos, 1.0f, 0.0f, 1.0f, corr[0], corr[1], theta_scale);
}

void set_rows_q4_0(cudaStream_t st, const float * src, int64_t ncols, int n_tok, const int64_t * idx, void * cache,
                   int64_t row_bytes) {
    const int64_t n_blocks = ncols*n_tok/QK4_0;
    constexpr int BS = 256;
    k_set_rows_q4_0<<<(unsigned) ((n_blocks + BS - 1)/BS), BS, 0, st>>>(src, idx, (char *) cache, ncols, n_blocks, row_bytes);
}

void mmvf_bf16(cudaStream_t st, const void * w, const float * x, int K, int nrows, int ncols, float * dst) {
    // launch_mul_mat_vec_f_cuda's block size: the fewest iterations over K/2 pairs, 64..256 threads (warp size first)
    int block_size_best = WARP_SIZE;
    int niter_best      = (K + 2*WARP_SIZE - 1) / (2*WARP_SIZE);
    for (int block_size = 2*WARP_SIZE; block_size <= 256; block_size += WARP_SIZE) {
        const int niter = (K + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    if (block_size_best != 256 || K % 2 != 0 || ncols < 1 || ncols > 8) {
        fprintf(stderr, "mmvf_bf16: K %d, %d columns: not an instantiated case (block 256, 1..8 columns)\n", K, ncols);
        abort();
    }
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) nrows, 1, 1), dim3(256, 1, 1), 0, st);
    const nv_bfloat16 * wb = (const nv_bfloat16 *) w;
#define ENG_MMVF(NC) ggml_cuda_kernel_launch(k_mmvf_bf16<NC, 256>, lp, wb, x, dst, K/2, K, K/2, nrows)
    switch (ncols) {
        case 1: ENG_MMVF(1); break;
        case 2: ENG_MMVF(2); break;
        case 3: ENG_MMVF(3); break;
        case 4: ENG_MMVF(4); break;
        case 5: ENG_MMVF(5); break;
        case 6: ENG_MMVF(6); break;
        case 7: ENG_MMVF(7); break;
        default: ENG_MMVF(8); break;
    }
#undef ENG_MMVF
}

void sigmoid(cudaStream_t st, const float * x, float * dst, int64_t k) {
    constexpr int BS = 256;
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) ((k + BS - 1)/BS), 1, 1), dim3(BS, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_sigmoid, lp, x, dst, (int) k);
}

void sigmoid_gate(cudaStream_t st, const float * gate, const float * x, float * dst, int64_t n, int64_t rows) {
    const int64_t k = n*rows;
    constexpr int BS = 256;
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) ((k + BS - 1)/BS), 1, 1), dim3(BS, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_gate<false>, lp, gate, x, dst, k);
}

void silu_gate(cudaStream_t st, const float * gate, const float * up, float * dst, int64_t n, int64_t rows) {
    const int64_t k = n*rows;
    constexpr int BS = 256;
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) ((k + BS - 1)/BS), 1, 1), dim3(BS, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_gate<true>, lp, gate, up, dst, k);
}

} // namespace eng
