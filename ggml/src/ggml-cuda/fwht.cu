#include "common.cuh"
#include "fwht.cuh"
#include "unary.cuh"
#include "xyzkv-quant.cuh"

// Fast Walsh-Hadamard transform, normalized Sylvester-Walsh ordering (butterfly x+y / x-y, scaled by
// 1/sqrt(N)). Two shapes:
//
//   fwht_cuda        one row per WARP, whole row in registers. Fine while N <= 256 (N/32 registers).
//   fwht_cuda_block  one row per block of NT threads for rows that do not fit in warp registers.
//
// `has_signs` folds a +/-1 vector into the input load. That is what makes a rotated-basis model
// (PrismML Ternary Bonsai 2, prism.hadamard.*) work: the weights are stored rotated, so the
// activation must be rotated the same way before the matmul. Doing the sign in the same load as the
// scale avoids a second full pass over the activation.
#define FWHT_BLOCK_THREADS 256

// Shared FWHT block and q8 twin helpers.
#include "fwht-block.cuh"
#include "gdn-out-chain.cuh"

template <int N, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    // block b of this row uses signs[b*N ...]; rows are laid out block-major, so r % n_blk is b
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide rows use shuffles within warps, shared memory across warps, and registers above the block width.
// The GLU form computes silu(first half) times the second half while loading.
template <int N, int NT, bool has_signs, bool glu>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const float * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk, char * __restrict__ q8_out, const int64_t glu_nc) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    ggml_cuda_pdl_sync();
    fwht_block_row<N, NT, has_signs, glu>(src, dst, r, scale, signs, n_blk, q8_out, glu_nc, (int) threadIdx.x, s);
}

template <bool permuted>
__launch_bounds__(FWHT_BLOCK_THREADS, 1)
static __global__ void gdn_out_chain_f32(
        const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ gate,
        const float * __restrict__ signs, float * __restrict__ dst, const float eps, const float fwht_scale,
        char * __restrict__ q8_out) {
    __shared__ float s_values[1024];
    __shared__ float s_sum[32];
    ggml_cuda_pdl_sync();
    ggml_cuda_gdn_out_chain_task<permuted>(
        x, w, gate, signs, dst, eps, fwht_scale, q8_out, blockIdx.x, threadIdx.x, s_values, s_sum, 0.0f, nullptr);
}

static bool fwht_launch(ggml_backend_cuda_context & ctx, const float * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk, char * q8_out = nullptr) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t stream = ctx.stream();
    dim3         grid_dims(num_blocks, 1, 1);
    dim3         block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define FWHT_WARP_CASE(NN)                                                                                   \
    case NN:                                                                                                 \
        if (signs) {                                                                                         \
            ggml_cuda_kernel_launch(fwht_cuda<NN, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
        } else {                                                                                             \
            ggml_cuda_kernel_launch(fwht_cuda<NN, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1);   \
        }                                                                                                    \
        return true;

#define FWHT_BLOCK_CASE(NN)                                                                                  \
    case NN: {                                                                                               \
        const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);                                     \
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream);            \
        if (signs) {                                                                                         \
            ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, true, false>,  lp, src_d, dst_d, rows, scale, signs, n_blk, q8_out, (int64_t) 0); \
        } else {                                                                                             \
            ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, false, false>, lp, src_d, dst_d, rows, scale, nullptr, 1, q8_out, (int64_t) 0); \
        }                                                                                                    \
        return true;                                                                                         \
    }

    switch (n) {
        FWHT_WARP_CASE(64)
        FWHT_WARP_CASE(128)
        FWHT_WARP_CASE(256)
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        default:
            return false;
    }
#undef FWHT_WARP_CASE
#undef FWHT_BLOCK_CASE
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t, void * q8_out = nullptr) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // The transform width is dst->ne[0]: a 5120-wide activation reshaped to [1024, 5*T] is five
    // independent 1024-blocks, which is exactly how the rotated weights were produced.
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int           n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    return fwht_launch(ctx, (const float *) src->data, (float *) dst->data, n, rows, 1 / sqrtf(n), signs, n_blk,
                       n >= 512 ? (char *) q8_out : nullptr);   // the twin comes from the block kernel only
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst, void * q8_out) {
    return fwht_dispatch(ctx, src, dst, signs, q8_out);
}

bool ggml_cuda_op_fwht_signed_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu_src,
                                  const ggml_tensor * signs, ggml_tensor * dst, void * q8_out) {
    constexpr int N  = 1024;
    const int64_t nc = glu_src->ne[0] / 2;
    if (glu_src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(glu_src) || !ggml_is_contiguous(dst) || !ggml_is_contiguous(signs) ||
            glu_src->ne[0] != 2*nc || nc % N != 0 || dst->ne[0] != N || signs->ne[0] != nc || ggml_nrows(signs) != 1 ||
            ggml_nelements(dst) != nc*ggml_nrows(glu_src)) {
        return false;
    }
    const int64_t rows = ggml_nelements(dst) / N;
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) rows, 1, 1), dim3(FWHT_BLOCK_THREADS, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(fwht_cuda_block<N, FWHT_BLOCK_THREADS, true, true>, lp, (const float *) glu_src->data,
        (float *) dst->data, rows, 1 / sqrtf(N), (const float *) signs->data, (int) (nc / N), (char *) q8_out, nc);
    return true;
}

bool ggml_cuda_op_gdn_out_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                               const ggml_tensor * mul_tensor, const ggml_tensor * silu_tensor,
                               const ggml_tensor * signs, ggml_tensor * dst, void * q8_out, bool permuted) {
    const ggml_tensor * x    = norm->src[0];
    const ggml_tensor * w    = mul_tensor->src[0] == norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * gate = silu_tensor->src[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
            signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(gate) ||
            !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst) ||
            x->ne[0] != 128 || x->ne[1] != 48 || x->ne[2] < 1 || x->ne[2] > 16 || x->ne[3] != 1 ||
            !ggml_are_same_shape(norm, x) || !ggml_are_same_shape(gate, x) ||
            w->ne[0] != 128 || ggml_nrows(w) != 1 || signs->ne[0] != 6144 || ggml_nrows(signs) != 1 ||
            dst->ne[0] != 1024 || ggml_nelements(dst) != ggml_nelements(x)) {
        return false;
    }

    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const float fwht_scale = 1/sqrtf(1024.0f);
    // one CTA per (token, 1024-block): 6 blocks of 1024 per token (48 heads x 128)
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (x->ne[2] * 6), 1, 1), dim3(FWHT_BLOCK_THREADS, 1, 1), 0,
                                            ctx.stream());
    if (permuted) {
        ggml_cuda_kernel_launch(gdn_out_chain_f32<true>, lp, (const float *) x->data, (const float *) w->data,
            (const float *) gate->data, (const float *) signs->data, (float *) dst->data, eps, fwht_scale,
            (char *) q8_out);
    } else {
        ggml_cuda_kernel_launch(gdn_out_chain_f32<false>, lp, (const float *) x->data, (const float *) w->data,
            (const float *) gate->data, (const float *) signs->data, (float *) dst->data, eps, fwht_scale,
            (char *) q8_out);
    }
    return true;
}

// Fuse optional add, RMS norm, weight multiply, signs, and a 1024-point Hadamard.
template <bool do_add, bool write_norm>
__launch_bounds__(1024, 1)
static __global__ void rms_norm_mul_fwht_f32_old(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x_out,
        const float * __restrict__ w, const float * __restrict__ signs, float * __restrict__ norm_out,
        float * __restrict__ dst, const int ncols, const float eps, const float fwht_scale, char * __restrict__ q8_out) {
    constexpr int BS = 1024;       // rms_norm_f32's block size for ncols >= 1024
    constexpr int N  = 1024;       // Hadamard block width
    constexpr int NT = FWHT_BLOCK_THREADS;
    constexpr int NE = N / NT;
    constexpr int NG = BS / NT;    // Hadamard blocks transformed at once
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(NE == 4 && NG == 4, "stage code below assumes 4 registers per thread");

    extern __shared__ float rnf_smem[];
    float * s_sum = rnf_smem;         // block_reduce scratch (32)
    float * s_row = rnf_smem + 32;    // the row: x, then the normalised row
    float * s_fw  = s_row + ncols;    // NG x N stage buffers

    const int     row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t off = (int64_t) row*ncols;

    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += BS) {
        float xi;
        if constexpr (do_add) {
            xi = a[off + col] + b[off + col];
            x_out[off + col] = xi;
        } else {
            xi = a[off + col];
        }
        s_row[col] = xi;
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += BS) {
        const float y = scale * s_row[col] * w[col];
        if constexpr (write_norm) {
            norm_out[off + col] = y;
        }
        s_row[col] = y;
    }
    __syncthreads();

    const int g    = tid / NT;
    const int t    = tid % NT;
    const int lane = tid % warp_size;
    float *   s    = s_fw + g*N;
    const int nblk = ncols / N;
    for (int b0 = 0; b0 < nblk; b0 += NG) {
        const int  blk    = b0 + g;
        const bool active = blk < nblk;
        float reg[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = active ? s_row[blk*N + i*NT + t] * fwht_scale : 0.0f;
            if (active) {
                reg[i] *= signs[blk*N + i*NT + t];
            }
        }
        // the two exchange stages stay rolled: unrolled, this kernel needs 64 registers on sm_89 and ptxas cannot
        // allocate it for sm_120a/sm_121a (C7600). Same operations in the same order
#pragma unroll 1
        for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                const float val  = reg[j];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
#pragma unroll 1
        for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                s[j*NT + t] = reg[j];
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < NE; j++) {
                const float val  = reg[j];
                const float val2 = s[j*NT + (t ^ h)];
                reg[j] = (t & h) == 0 ? val + val2 : val2 - val;
            }
            __syncthreads();
        }
#pragma unroll
        for (int h = NT; h < N; h *= 2) {
            const int step = h / NT;
#pragma unroll
            for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
                for (int k = 0; k < step; k++) {
                    const float x = reg[j + k];
                    const float y = reg[j + k + step];
                    reg[j + k]        = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }
        if (active) {   // whole warps: a group of NT = 256 threads is 8 warps
            const int64_t base = ((int64_t) row*nblk + blk)*N;
            float * d = dst + base;
#pragma unroll
            for (int i = 0; i < NE; ++i) {
                d[i*NT + t] = reg[i];
                if (q8_out != nullptr) {
                    ptq1_q8_perm_store(q8_out, base + i*NT + t, reg[i]);
                }
            }
        }
    }
}

// The 5120-wide form processes five 1024-point blocks in one pass while preserving operation order.
template <bool do_add, bool write_norm>
__launch_bounds__(1024, 1)
static __global__ void rms_norm_mul_fwht_f32_5k(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x_out,
        const float * __restrict__ w, const float * __restrict__ signs, float * __restrict__ norm_out,
        float * __restrict__ dst, const float eps, const float fwht_scale, char * __restrict__ q8_out) {
    constexpr int BS    = 1024;
    constexpr int NCOLS = 5120;
    constexpr int NPT   = NCOLS / BS;   // 5 elements of the row per thread (norm phase)
    constexpr int N     = 1024;         // Hadamard block
    constexpr int NBLK  = NCOLS / N;    // 5
    constexpr int NT    = 128;          // threads per Hadamard block
    constexpr int NE    = N / NT;       // 8 registers per thread
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(NBLK * NT <= BS && (NBLK * NT) % warp_size == 0, "whole warps per Hadamard group");

    extern __shared__ float rnf_smem[];
    float * s_sum = rnf_smem;             // block_reduce scratch (32)
    float * s_row = rnf_smem + 32;        // the normalised row
    float * s_fw  = s_row + NCOLS;        // NBLK x N stage buffers

    const int     row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t off = (int64_t) row*NCOLS;
    const int     g   = tid / NT;         // Hadamard group = block index
    const int     t   = tid % NT;
    const bool    fw  = g < NBLK;         // whole warps (640 threads = 20 warps)

    // weights first: not written by the previous kernel, so they may be read before the PDL wait
    float wv[NPT];
    float sg[NE];
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        wv[j] = w[tid + j*BS];
    }
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        sg[i] = fw ? signs[g*N + i*NT + t] : 0.0f;
    }

    ggml_cuda_pdl_sync();
    float xv[NPT];
    if constexpr (do_add) {
        float av[NPT], bv[NPT];
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            av[j] = a[off + tid + j*BS];
            bv[j] = b[off + tid + j*BS];
        }
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            xv[j] = av[j] + bv[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            xv[j] = a[off + tid + j*BS];
        }
    }
    float tmp = 0.0f;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        if constexpr (do_add) {
            x_out[off + tid + j*BS] = xv[j];
        }
        tmp += xv[j] * xv[j];
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float mean  = tmp / NCOLS;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const float y = scale * xv[j] * wv[j];
        if constexpr (write_norm) {
            norm_out[off + tid + j*BS] = y;
        }
        s_row[tid + j*BS] = y;
    }
    __syncthreads();

    const int lane = tid % warp_size;
    float *   s    = s_fw + g*N;
    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fw ? s_row[g*N + i*NT + t] * fwht_scale : 0.0f;
        reg[i] *= sg[i];
    }
    if (fw) {
#pragma unroll
        for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                const float val  = reg[j];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
        if (fw) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                s[j*NT + t] = reg[j];
            }
        }
        __syncthreads();
        if (fw) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                const float val  = reg[j];
                const float val2 = s[j*NT + (t ^ h)];
                reg[j] = (t & h) == 0 ? val + val2 : val2 - val;
            }
        }
        __syncthreads();
    }
    if (!fw) {
        return;
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    const int64_t base = ((int64_t) row*NBLK + g)*N;
    float * d = dst + base;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        d[i*NT + t] = reg[i];
        if (q8_out != nullptr) {
            ptq1_q8_perm_store(q8_out, base + i*NT + t, reg[i]);
        }
    }
}

// One CTA handles each (row, 1024-element block). Every CTA reduces the full row in the same order,
// then normalizes and transforms only its block with the established butterfly stage order.
template <bool do_add, bool write_norm>
__launch_bounds__(1024, 1)
static __global__ void rms_norm_mul_fwht_f32_5k_split(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x_out,
        const float * __restrict__ w, const float * __restrict__ signs, float * __restrict__ norm_out,
        float * __restrict__ dst, const float eps, const float fwht_scale, char * __restrict__ q8_out) {
    constexpr int BS    = 1024;
    constexpr int NCOLS = 5120;
    constexpr int NPT   = NCOLS / BS;   // 5 elements of the row per thread (norm phase)
    constexpr int N     = 1024;         // Hadamard block = this CTA's slice
    constexpr int NBLK  = NCOLS / N;    // 5
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    __shared__ float s_sum[32];
    __shared__ float s[N];

    const int     row = blockIdx.x / NBLK;
    const int     blk = blockIdx.x % NBLK;
    const int     tid = threadIdx.x;
    const int     lane = tid % warp_size;
    const int64_t off = (int64_t) row*NCOLS;
    const int     col = blk*N + tid;    // this thread's element of the CTA's block (block j == blk is element tid + blk*BS)

    // weights first: not written by the previous kernel, so they may be read before the PDL wait
    const float wv = w[col];
    const float sg = signs[col];

    ggml_cuda_pdl_sync();
    float xv[NPT];
    if constexpr (do_add) {
        float av[NPT], bv[NPT];
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            av[j] = a[off + tid + j*BS];
            bv[j] = b[off + tid + j*BS];
        }
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            xv[j] = av[j] + bv[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            xv[j] = a[off + tid + j*BS];
        }
    }
    float tmp = 0.0f;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        tmp += xv[j] * xv[j];
    }
    if constexpr (do_add) {
        x_out[off + col] = xv[blk];     // this CTA's slice of the residual sum
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float mean  = tmp / NCOLS;
    const float scale = rsqrtf(mean + eps);

    const float y = scale * xv[blk] * wv;
    if constexpr (write_norm) {
        norm_out[off + col] = y;
    }

    float v = y * fwht_scale;
    v *= sg;
    // stages within a warp: the partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, h, warp_size);
        v = (lane & h) == 0 ? v + v2 : v2 - v;
    }
    // stages across warps: the partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        s[tid] = v;
        __syncthreads();
        const float v2 = s[tid ^ h];
        v = (tid & h) == 0 ? v + v2 : v2 - v;
        __syncthreads();
    }

    const int64_t base = ((int64_t) row*NBLK + blk)*N;
    dst[base + tid] = v;
    if (q8_out != nullptr) {   // uniform per launch: whole warps call it
        ptq1_q8_perm_store(q8_out, base + tid, v);
    }
}

template <bool write_norm>
__launch_bounds__(1024, 1)
static __global__ void rms_norm_mul_fwht_f32(
        const float * __restrict__ a, const float * __restrict__ w, const float * __restrict__ signs,
        float * __restrict__ norm_out, float * __restrict__ dst, const int ncols, const float eps,
        const float fwht_scale, char * __restrict__ q8_out) {
    constexpr int BS = 1024;
    constexpr int N  = 1024;
    constexpr int NT = FWHT_BLOCK_THREADS;
    constexpr int NE = N / NT;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    extern __shared__ float rnf_smem[];
    float * s_sum = rnf_smem;
    float * s_fw  = rnf_smem + 32;

    const int nblk = ncols / N;
    const int row  = blockIdx.x / nblk;
    const int blk  = blockIdx.x % nblk;
    const int tid  = threadIdx.x;
    const int64_t off = (int64_t) row*ncols;

    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += BS) {
        const float xi = a[off + col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (write_norm) {
        if (blk == 0) {
            for (int col = tid; col < ncols; col += BS) {
                norm_out[off + col] = scale * a[off + col] * w[col];
            }
        }
    }

    const int t    = tid % NT;
    const int lane = tid % warp_size;
    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        if (tid < NT) {
            const int col = blk*N + i*NT + t;
            reg[i] = scale * a[off + col] * w[col];
            reg[i] *= fwht_scale;
            reg[i] *= signs[col];
        }
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (tid < NT) {
                const float val  = reg[j];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (tid < NT) {
                s_fw[j*NT + t] = reg[j];
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (tid < NT) {
                const float val  = reg[j];
                const float val2 = s_fw[j*NT + (t ^ h)];
                reg[j] = (t & h) == 0 ? val + val2 : val2 - val;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                if (tid < NT) {
                    const float x = reg[j + k];
                    const float y = reg[j + k + step];
                    reg[j + k]        = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }
    }

    if (tid < NT) {
        const int64_t base = ((int64_t) row*nblk + blk)*N;
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            dst[base + i*NT + t] = reg[i];
            if (q8_out != nullptr) {
                ptq1_q8_perm_store(q8_out, base + i*NT + t, reg[i]);
            }
        }
    }
}

bool ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * norm,
                                const ggml_tensor * w, const ggml_tensor * signs, ggml_tensor * norm_out, ggml_tensor * dst,
                                void * q8_out) {
    const ggml_tensor * x = add ? add : norm->src[0];
    const int     ncols = (int) x->ne[0];
    const int64_t nrows = ggml_nrows(x);
    if (ncols < 1024 || ncols % 1024 != 0 || ncols > 8192 || nrows > INT32_MAX) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst) ||
        w->ne[0] != ncols || ggml_nrows(w) != 1 || signs->ne[0] != ncols || ggml_nrows(signs) != 1 ||
        dst->ne[0] != 1024 || ggml_nelements(dst) != ggml_nelements(x)) {
        return false;
    }
    if (add && (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(add->src[0], add) || !ggml_are_same_shape(add->src[1], add) ||
                !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]))) {
        return false;
    }
    if (norm_out && (norm_out->type != GGML_TYPE_F32 || !ggml_is_contiguous(norm_out) || !ggml_are_same_shape(norm_out, x))) {
        return false;
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const float  fwht_scale = 1 / sqrtf(1024);   // fwht_dispatch's scale for n = 1024
    const size_t old_smem   = (32 + (size_t) ncols + 4*1024)*sizeof(float);

    const float * a  = add ? (const float *) add->src[0]->data : (const float *) x->data;
    const float * b  = add ? (const float *) add->src[1]->data : nullptr;
    float *       xo = add ? (float *) add->data : nullptr;
    float *       no = norm_out ? (float *) norm_out->data : nullptr;

    const size_t row_bytes = (size_t) nrows*ncols*sizeof(float);
    const auto overlaps = [row_bytes](const float * p, const float * q) {
        return q != nullptr && (const char *) p < (const char *) q + row_bytes && (const char *) q < (const char *) p + row_bytes;
    };
    if (ncols == 5120 && !(add && (overlaps(xo, a) || overlaps(xo, b)))) {
        const ggml_cuda_kernel_launch_params lp_sp(dim3((unsigned) (nrows*5), 1, 1), dim3(1024, 1, 1), 0, ctx.stream());
#define RNF_SPLIT_LAUNCH(ADD, NORM) ggml_cuda_kernel_launch(rms_norm_mul_fwht_f32_5k_split<ADD, NORM>, lp_sp, a, b, xo, \
        (const float *) w->data, (const float *) signs->data, no, (float *) dst->data, eps, fwht_scale, (char *) q8_out)
        if (add) {
            if (no) { RNF_SPLIT_LAUNCH(true, true); } else { RNF_SPLIT_LAUNCH(true, false); }
        } else {
            if (no) { RNF_SPLIT_LAUNCH(false, true); } else { RNF_SPLIT_LAUNCH(false, false); }
        }
#undef RNF_SPLIT_LAUNCH
        return true;
    }
    if (ncols == 5120) {
        const size_t smem_5k = (32 + 5120 + 5*1024)*sizeof(float);
        const ggml_cuda_kernel_launch_params lp_5k(dim3((unsigned) nrows, 1, 1), dim3(1024, 1, 1), smem_5k, ctx.stream());
#define RNF_5K_LAUNCH(ADD, NORM) ggml_cuda_kernel_launch(rms_norm_mul_fwht_f32_5k<ADD, NORM>, lp_5k, a, b, xo, \
        (const float *) w->data, (const float *) signs->data, no, (float *) dst->data, eps, fwht_scale, (char *) q8_out)
        if (add) {
            if (no) { RNF_5K_LAUNCH(true, true); } else { RNF_5K_LAUNCH(true, false); }
        } else {
            if (no) { RNF_5K_LAUNCH(false, true); } else { RNF_5K_LAUNCH(false, false); }
        }
#undef RNF_5K_LAUNCH
        return true;
    }

    const ggml_cuda_kernel_launch_params old_lp = ggml_cuda_kernel_launch_params(dim3((unsigned) nrows, 1, 1), dim3(1024, 1, 1), old_smem, ctx.stream());
#define RNF_OLD_LAUNCH(ADD, NORM) ggml_cuda_kernel_launch(rms_norm_mul_fwht_f32_old<ADD, NORM>, old_lp, a, b, xo, (const float *) w->data, \
        (const float *) signs->data, no, (float *) dst->data, ncols, eps, fwht_scale, (char *) q8_out)
    if (add) {
        if (no) { RNF_OLD_LAUNCH(true, true); } else { RNF_OLD_LAUNCH(true, false); }
    } else {
        if (no) { RNF_OLD_LAUNCH(false, true); } else { RNF_OLD_LAUNCH(false, false); }
    }
#undef RNF_OLD_LAUNCH
    return true;
}

// Fuse the attention output chain in one launch: inverse 128-point WHT with
// the InnerQ scale (k_xyzkv_wht_f32<1, 128>), the V rotation's 64-point Hadamard (a MUL_MAT with the Hadamard hint =
// fwht_cuda<64, false>), the output gate (CONT of the Qcur_full gate view + unary_gated_op_kernel<op_sigmoid>), and the
// rotated attn_output input's signed 1024-block Hadamard with its q8_1 twin (fwht_cuda_block<1024, 256, true>): five
// launches per attention layer, 16 layers. One CTA of 256 threads per (token, 1024-block); register i of thread tid is
// element i*256 + tid of the block = element tid of head blk*4 + i. Bit-identical by construction, stage by stage:
//   xyzkv  x*SIGNS2[t] (t = tid % 128), stages h = 1..64, then (x*inv_sqrt)*SIGNS1[t], then *scale_inv[t]
//   had64  x*scale64 (fwht_dispatch's 1/sqrtf(64)), stages h = 1..32
//   gate   op_sigmoid(g)*x, unary_gated_op_kernel's product with unary.cu's 1/(1 + expf(-g))
//   had1k  (x*scale1k)*signs, stages h = 1..512
// Each butterfly stage preserves the listed (a + b, a - b) operation order.
static __device__ __forceinline__ float attn_out_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));   // unary.cu op_sigmoid
}

template <int NE>
static __device__ __forceinline__ void attn_out_stage_warp(float (&reg)[NE], const int h, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const float val  = reg[j];
        const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
        reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
    }
}

template <int NE, int NT>
static __device__ __forceinline__ void attn_out_stage_smem(float (&reg)[NE], float * s, const int h, const int tid) {
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        s[j*NT + tid] = reg[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const float val  = reg[j];
        const float val2 = s[j*NT + (tid ^ h)];
        reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
    }
    __syncthreads();
}

__launch_bounds__(FWHT_BLOCK_THREADS, 1)
static __global__ void attn_out_chain_f32(
        const float * __restrict__ x, const float * __restrict__ gate, const int64_t gate_s1, const int64_t gate_s2,
        const float * __restrict__ scale_inv, const float * __restrict__ signs, float * __restrict__ dst,
        const float scale64, const float scale1k, char * __restrict__ q8_out) {
    constexpr int HD   = 256;                 // head dim
    constexpr int NC   = 6144;                // 24 heads per token
    constexpr int N    = 1024;                // Hadamard block
    constexpr int NT   = FWHT_BLOCK_THREADS;
    constexpr int NE   = N / NT;              // registers per thread = heads per block
    constexpr int NBLK = NC / N;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(NT == HD && NE == 4, "register i of thread tid is element tid of head blk*NE + i");

    __shared__ float s[N];

    const int r     = blockIdx.x;
    const int token = r / NBLK;
    const int blk   = r % NBLK;
    const int tid   = threadIdx.x;
    const int lane  = tid % warp_size;
    const int t     = tid % 128;              // index inside the xyzkv group

    ggml_cuda_pdl_sync();
    // every global load first
    float reg[NE], g[NE], sg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int head = blk*NE + i;
        reg[i] = x[(int64_t) token*NC + head*HD + tid];
        g[i]   = gate[(int64_t) token*gate_s2 + (int64_t) head*gate_s1 + tid];
        sg[i]  = signs[blk*N + i*NT + tid];
    }
    const float si = scale_inv != nullptr ? scale_inv[t] : 1.0f;
    const float s1 = XYZKV_WHT_SIGNS1[t];
    const float s2 = XYZKV_WHT_SIGNS2[t];

    // 1. xyzkv inverse WHT on each 128-group (tid bits 0..6)
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] *= s2;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
        attn_out_stage_warp(reg, h, lane);
    }
#pragma unroll
    for (int h = warp_size; h < 128; h *= 2) {
        attn_out_stage_smem<NE, NT>(reg, s, h, tid);
    }
    constexpr float inv_sqrt = 0.08838834764831845f;   // k_xyzkv_wht_f32's 1/sqrt(128)
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * inv_sqrt * s1;
        if (scale_inv != nullptr) {
            reg[i] *= si;
        }
    }

    // 2. the V rotation's 64-point Hadamard (tid bits 0..5)
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * scale64;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
        attn_out_stage_warp(reg, h, lane);
    }
#pragma unroll
    for (int h = warp_size; h < 64; h *= 2) {
        attn_out_stage_smem<NE, NT>(reg, s, h, tid);
    }

    // 3. the output gate
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = attn_out_sigmoid(g[i]) * reg[i];
    }

    // 4. signs + 1024-block Hadamard: lanes, warps (h = 32..128), then registers (h = 256, 512)
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * scale1k;
        reg[i] *= sg[i];
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
        attn_out_stage_warp(reg, h, lane);
    }
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
        attn_out_stage_smem<NE, NT>(reg, s, h, tid);
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k];
                const float b = reg[j + k + step];
                reg[j + k]        = a + b;
                reg[j + k + step] = a - b;
            }
        }
    }

    const int64_t base = (int64_t) r*N;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[base + i*NT + tid] = reg[i];
        if (q8_out != nullptr) {   // uniform per launch: whole warps call it
            ptq1_q8_perm_store(q8_out, base + i*NT + tid, reg[i]);
        }
    }
}

bool ggml_cuda_op_attn_out_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * xyzkv, const ggml_tensor * had,
                                 const ggml_tensor * gate, const ggml_tensor * signs, ggml_tensor * dst, void * q8_out) {
    const ggml_tensor * x         = xyzkv->src[0];
    const ggml_tensor * scale_inv = xyzkv->src[1];
    int direction = 0, group_size = 0;
    memcpy(&direction,  xyzkv->op_params + 0,           sizeof(int));
    memcpy(&group_size, xyzkv->op_params + sizeof(int), sizeof(int));   // op_params[4], as ggml_xyzkv_wht stores it
    if (direction != 1 || group_size != 128 ||
            x->type != GGML_TYPE_F32 || xyzkv->type != GGML_TYPE_F32 || had->type != GGML_TYPE_F32 ||
            gate->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            x->ne[0] != 256 || x->ne[1] != 24 || x->ne[2] < 1 || x->ne[2] > 16 || x->ne[3] != 1 ||
            !ggml_is_contiguous(x) || !ggml_are_same_shape(xyzkv, x) ||
            had->ne[0] != 64 || had->src[0]->ne[0] != 64 || had->src[0]->ne[1] != 64 ||
            ggml_nelements(had) != ggml_nelements(x) ||
            !ggml_are_same_shape(gate, x) || gate->nb[0] != sizeof(float) ||
            gate->nb[1] % sizeof(float) != 0 || gate->nb[2] % sizeof(float) != 0 ||
            !ggml_is_contiguous(signs) || signs->ne[0] != 6144 || ggml_nrows(signs) != 1 ||
            !ggml_is_contiguous(dst) || dst->ne[0] != 1024 || ggml_nelements(dst) != ggml_nelements(x) ||
            (scale_inv != nullptr && (scale_inv->type != GGML_TYPE_F32 || ggml_nelements(scale_inv) < 128))) {
        return false;
    }
    const float scale64 = 1 / sqrtf(64);     // fwht_dispatch's, n = 64
    const float scale1k = 1 / sqrtf(1024);   // fwht_dispatch's, n = 1024
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (x->ne[2] * 6), 1, 1), dim3(FWHT_BLOCK_THREADS, 1, 1), 0,
                                            ctx.stream());
    ggml_cuda_kernel_launch(attn_out_chain_f32, lp, (const float *) x->data, (const float *) gate->data,
        (int64_t) (gate->nb[1] / sizeof(float)), (int64_t) (gate->nb[2] / sizeof(float)),
        scale_inv ? (const float *) scale_inv->data : nullptr, (const float *) signs->data, (float *) dst->data,
        scale64, scale1k, (char *) q8_out);
    return true;
}
