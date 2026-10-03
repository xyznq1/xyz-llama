#pragma once
// The engine's own MMQ: the quantized matmul of the prompt path above 32 rows (the target's PTQ1_0 weights) and of the
// drafter's batches above its MMVQ widths (Q3_K). Derived from our fork's mmq.cuh / mmq-load-tiles.cuh /
// mmq-vec-dot.cuh / quantize.cu at exactly the instances the server runs on sm_89: the Ampere+ config (256 threads,
// I = 128 rows per tile, K_vram 256, stream-K), the int8 tensor-core data layout, the PTQ1_0 ILV16 loader and the Q3_K
// loader, the q8_0 x q8_1 (D4) and q8_0_16 x q8_1 dot products, the stream-K partition and its fixup, and the D4 q8_1
// activation layout -- statement for statement, so every value is the fork's bits (tools/mmq_test.cu). Not carried: the
// other types, AMD / Volta / dp4a / Blackwell paths, MoE (ids, expert bounds), the channel / sample broadcast (one 2-D
// matmul), NVFP4 scales, the tiling (non-stream-K) kernel, the diagnostics.
#include "common.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"   // get_int_b2 / get_int_b4, VDR_Q3_K_Q8_1_MMQ

namespace eng {
namespace mq {

using namespace ggml_cuda_mma;

constexpr int QK8_1_MMQ = 4*QK8_1;   // 128 values per activation block
constexpr int TILE_NE_K = 32;        // MMQ_TILE_NE_K: 32-bit elements of quantized data per SRAM tile half
constexpr int TILE_Y_K  = TILE_NE_K + TILE_NE_K/QI8_1;   // 36: one block_q8_1_mmq in ints
constexpr int ITER_K    = 256;       // K_vram (MMQ_ITER_K)
constexpr int NTHREADS  = 256;
constexpr int I         = 128;       // rows per tile
constexpr int NWARPS    = NTHREADS / WARP_SIZE;

// the activation layout (D4): 128 values per block, one f32 scale per 32
struct block_q8_1_mmq {
    float  d4[4];
    int8_t qs[QK8_1_MMQ];
};
static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");

// the SRAM row stride of the x tile (int32 elements): LAYOUT_Q8_0 for PTQ1_0, LAYOUT_Q3_K for Q3_K
template <ggml_type type> constexpr __host__ __device__ int sram_stride() {
    return type == GGML_TYPE_PTQ1_0 ? 2*TILE_NE_K + 2*TILE_NE_K/QI8_0 + 4 : 2*TILE_NE_K + TILE_NE_K/2 + 4;
}
template <ggml_type type> constexpr __host__ __device__ int qk() {
    return type == GGML_TYPE_PTQ1_0 ? QK_PTQ1_0 : QK_K;
}
// the J (n-tile width) entries of the config table (fallback = false: every engine weight has a multiple of 128 rows)
constexpr __host__ __device__ bool j_valid(const int J) {
    return J == 8 || J == 16 || J == 24 || J == 32 || J == 40 || J == 48 || J == 64 || J == 80 || J == 96 || J == 112 ||
           J == 128;
}
constexpr __host__ __device__ int rows_per_warp(const int J) {
    return J >= 48 && J % 16 == 0 ? 32 : 16;
}
template <ggml_type type> constexpr __host__ __device__ int nbytes_shared(const int J) {
    return J*(int) sizeof(int) + I*sram_stride<type>()*4 + GGML_PAD(J*(int) sizeof(block_q8_1_mmq), NTHREADS*(int) sizeof(int));
}

// ---- the x tiles --------------------------------------------------------------------------------------------------

// mmq-load-tiles.cuh ggml_cuda_mmq_decode_ptq1_0_qs4: 4 bytes of 5 trits -> 5 ints of 4 signed values at dst[t*stride]
static __device__ __forceinline__ void decode_ptq1_0_qs4(uint32_t packed, int * __restrict__ dst, int stride) {
    uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
    uint32_t v_hi = __byte_perm(packed, 0, 0x4342);

#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const uint32_t w_lo = v_lo * 3;
        const uint32_t w_hi = v_hi * 3;
        v_lo                = w_lo & 0x00FF00FF;
        v_hi                = w_hi & 0x00FF00FF;
        dst[t * stride]     = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
    }
}

// ggml_cuda_mmq_load_tiles_ptq1_0 (the int8 MMA layout, ILV16): the trits of I rows x 2 blocks as signed bytes, then
// the blocks' scales, one per 32 values
static __device__ __forceinline__ void load_tiles_ptq1_0(const char * __restrict__ x, int * __restrict__ x_tile,
                                                         const int kbx0, const int i_max, const int stride) {
    constexpr int ss = sram_stride<GGML_TYPE_PTQ1_0>();

    int *   x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * TILE_NE_K);

    constexpr int blocks_per_iter   = ITER_K / QK_PTQ1_0;
    constexpr int threads_per_block = 8;
    constexpr int threads_per_row   = blocks_per_iter * threads_per_block;
    constexpr int nrows             = WARP_SIZE / threads_per_row;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / threads_per_block;
    const int lane = txi % threads_per_block;

    const int kb0   = kbx0 % stride;
    const int rbase = kbx0 - kb0;
    const int nfull = (i_max + 1) & ~15;
    const auto ptq1_block = [&](const int i, const int kb) -> const block_ptq1_0 * {
        return i < nfull ? (const block_ptq1_0 *) x + rbase + (i & ~15)*stride + min(kb0 + kb, stride - 1)*16 + (i & 15)
                         : (const block_ptq1_0 *) x + kbx0 + i*stride + kb;
    };

    // Every lane's global load for every row is issued before any decode. With the loads inside the per-lane branches
    // (lane < 4 / < 6 / == 6) each row paid load -> decode -> store as its own latency chain, I/(nrows*NWARPS) rows x
    // 3 paths in series per warp (SASS: 8 x (LDG STS | LDG STSx5 | LDG STSx5) per 256-value step). One aligned word per
    // lane and row instead -- lanes 0-5 the six qs words, lanes 6-7 bytes 24-27 (qh[0], qh[1], d; block_ptq1_0 is
    // 4-aligned) -- decoded into exactly the words the branches wrote: qs word w lands at row + w (stride 4) for w < 4
    // and at row + 16 + w (stride 2) for w = 4, 5. Bit-identical tiles.
    constexpr int niter = I / (nrows * NWARPS);
    static_assert(I % (nrows * NWARPS) == 0, "whole rows per warp pass");
    uint32_t packed[niter];

#pragma unroll
    for (int it = 0; it < niter; ++it) {
        const int i = it * nrows * NWARPS + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        packed[it] = get_int_b4(ptq1_block(i, kbx)->qs, min(lane, 6));
    }

#pragma unroll
    for (int it = 0; it < niter; ++it) {
        const int i = it * nrows * NWARPS + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        int * row = x_qs + i * ss + kbx * (QK_PTQ1_0 / 4);

        if (lane < 6) {
            decode_ptq1_0_qs4(packed[it], row + (lane < 4 ? lane : 16 + lane), lane < 4 ? 4 : 2);
        } else if (lane == 6) {
            uint32_t v = __byte_perm(packed[it], 0, 0x4140); // qh[0] | qh[1] << 16
#pragma unroll
            for (int t = 0; t < 4; t += 2) {
                const uint32_t w0 = v * 3;
                v                 = w0 & 0x00FF00FF;
                const uint32_t w1 = v * 3;
                v                 = w1 & 0x00FF00FF;
                row[30 + t / 2]   = __vsub4(__byte_perm(w0, w1, 0x7531), 0x01010101);
            }
        }
    }

    constexpr int scale_entries_per_block = QK_PTQ1_0 / QK8_1;
    constexpr int scale_entries_per_row   = blocks_per_iter * scale_entries_per_block;
    constexpr int rows_per_warp_sc        = WARP_SIZE / scale_entries_per_row;
    const int     ksx                     = threadIdx.x % scale_entries_per_row;
    const int     scale_block             = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += NWARPS * rows_per_warp_sc) {
        int i = i0 + threadIdx.y * rows_per_warp_sc + threadIdx.x / scale_entries_per_row;

        const block_ptq1_0 * bxi = ptq1_block(i, scale_block);
        x_df[i * ss + ksx] = bxi->d;
    }
}

// ggml_cuda_mmq_load_tiles_q3_K (the int8 MMA layout): the 3-bit values as signed bytes, then d * scale per 16 values
static __device__ __forceinline__ void load_tiles_q3_K(const char * __restrict__ x, int * __restrict__ x_tile,
                                                       const int kbx0, const int i_max, const int stride) {
    constexpr int ss = sram_stride<GGML_TYPE_Q3_K>();
    GGML_UNUSED(i_max);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + TILE_NE_K*2);

    constexpr int threads_per_row = ITER_K / (4 * QR3_K);
    constexpr int nrows = WARP_SIZE / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

            x_qs[i*ss + k] = x_qs_k;
        }
    }

    constexpr int rows_per_warp_sc = WARP_SIZE / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp_sc) {
        int i = i0 + threadIdx.y*rows_per_warp_sc + threadIdx.x/4;

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*ss + sizeof(int)*ksc + l] = d*sc8[l];
        }
    }
}

// ---- the dot products ---------------------------------------------------------------------------------------------

// ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<.., MMQ_Q8_1_DS_LAYOUT_D4> (the Turing MMA branch): PTQ1_0
template <int J>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1(const int * __restrict__ x, const int * __restrict__ y,
                                                         float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int ss  = sram_stride<GGML_TYPE_PTQ1_0>();
    constexpr int rpw = rows_per_warp(J);
    constexpr int ntx = rpw/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    tile_A A[ntx][TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rpw;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*ss + k0, ss);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*ss + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*TILE_Y_K + k01, TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}

// ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma (the Turing MMA branch): Q3_K
template <int J>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1(const int * __restrict__ x, const int * __restrict__ y,
                                                            float * __restrict__ sum, const int k00) {
    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int ss  = sram_stride<GGML_TYPE_Q3_K>();
    constexpr int rpw = rows_per_warp(J);
    constexpr int ntx = rpw/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*ss + k0, ss);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*ss + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*TILE_Y_K + (k01 + 0),         TILE_Y_K);
            load_generic(B[1], y_qs + j0*TILE_Y_K + (k01 + tile_B::J), TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
}

// ggml_cuda_mmq_write_back_mma (fallback = false: no row bound)
template <int J>
static __device__ __forceinline__ void write_back(const float * __restrict__ sum, const int * __restrict__ ids_dst,
                                                  float * __restrict__ dst, const int stride, const int j_max) {
    typedef tile<16,  8, int> tile_C;

    constexpr int rpw = rows_per_warp(J);
    constexpr int ntx = rpw/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

// ---- the kernel -----------------------------------------------------------------------------------------------------

// mul_mat_q_process_tile: k-blocks [kb0_start, kb0_stop) of one output tile, each K_vram step two 32-int halves of y
template <ggml_type type, int J, bool fixup>
static __device__ __forceinline__ void process_tile(const char * __restrict__ x, const int offset_x,
                                                    const int * __restrict__ y, const int * __restrict__ ids_dst,
                                                    float * __restrict__ dst, float * __restrict__ tmp_fixup,
                                                    const int stride_row_x, const int ncols_y, const int stride_col_dst,
                                                    const int tile_x_max_i, const int tile_y_max_j, const int kb0_start,
                                                    const int kb0_stop) {
    constexpr int QK = qk<type>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J*TILE_Y_K, NWARPS*WARP_SIZE);

    constexpr int ne_block        = QK8_1_MMQ;
    constexpr int blocks_per_iter = ITER_K / QK;

    float sum[J*I / (NWARPS*WARP_SIZE)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        if constexpr (type == GGML_TYPE_PTQ1_0) {
            load_tiles_ptq1_0(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        } else {
            load_tiles_q3_K(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        }
        {
            const int * by0 = y + ncols_y * (kb0 * QK / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * TILE_Y_K; l0 += NWARPS * WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        if constexpr (type == GGML_TYPE_PTQ1_0) {
            vec_dot_q8_0_q8_1<J>(tile_x, tile_y, sum, 0);
        } else {
            vec_dot_q8_0_16_q8_1<J>(tile_x, tile_y, sum, 0);
        }

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * QK / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * TILE_Y_K; l0 += NWARPS * WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        if constexpr (type == GGML_TYPE_PTQ1_0) {
            vec_dot_q8_0_q8_1<J>(tile_x, tile_y, sum, TILE_NE_K);
        } else {
            vec_dot_q8_0_16_q8_1<J>(tile_x, tile_y, sum, TILE_NE_K);
        }

        __syncthreads();
    }

    if (fixup) {
        write_back<J>(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), I, J);
    } else {
        write_back<J>(sum, ids_dst, dst, stride_col_dst, tile_y_max_j);
    }
}

// mul_mat_q, the stream-k branch (one sample, one channel, no ids): CUDA block b takes the contiguous run of the
// flattened (tile, k-block) space [b, b + 1) * total / gridDim.x, snapped to K_vram steps; whole tiles go to dst, the
// last partial one to the fixup buffer
template <ggml_type type, int J>
__launch_bounds__(NTHREADS, 1)
static __global__ void k_mmq(const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
                             float * __restrict__ tmp_fixup, const uint3 blocks_per_ne00, const int nrows_x,
                             const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
                             const uint3 ntx) {
    constexpr int QK = qk<type>();

    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += NWARPS*WARP_SIZE) {
        const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

        if (j0 + NWARPS*WARP_SIZE > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    constexpr int blocks_per_iter = ITER_K / QK;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        const int it = tmp2.x;

        const int offset_y   = (jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        const int offset_dst = jt*J*stride_col_dst + it*I;

        const int tile_x_max_i = nrows_x   - it*I - 1;
        const int tile_y_max_j = ncols_dst - jt*J - 1;

        const int offset_x = it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        process_tile<type, J, fixup>(x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
                                     stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    const int it = tmp2.x;

    const int offset_y   = (jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    const int offset_dst = jt*J*stride_col_dst + it*I;

    const int tile_x_max_i = nrows_x   - it*I - 1;
    const int tile_y_max_j = ncols_dst - jt*J - 1;

    const int offset_x = it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    process_tile<type, J, fixup>(x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
                                 stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

// mul_mat_q_stream_k_fixup: a block that finished a tile adds the partial sums the blocks before it left for that tile,
// in descending block order, into dst
template <ggml_type type, int J>
__launch_bounds__(NTHREADS/2, 1)
static __global__ void k_mmq_fixup(float * __restrict__ dst, const float * __restrict__ tmp_last_tile,
                                   const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
                                   const int stride_col_dst, const uint3 ntx) {
    constexpr int nwarps          = (NTHREADS / 2) / WARP_SIZE;
    constexpr int QK              = qk<type>();
    constexpr int blocks_per_iter = ITER_K / QK;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*WARP_SIZE + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    const int it = tmp2.x;

    const int offset_dst = jt*J*stride_col_dst + it*I;
    dst += offset_dst;

    const int j_max = ncols_dst - jt*J - 1;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[j*stride_col_dst + i] += sum[j0/nwarps];
    }
}

// ---- the activation -------------------------------------------------------------------------------------------------

// quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false> for one 2-D activation: column blockIdx.x (s01 floats apart, ne00
// values, zero-padded to ne0), four values per thread, the D4 scales
__global__ void k_quantize_d4(const float * __restrict__ x, void * __restrict__ vy, const int64_t ne00, const int64_t s01,
                              const int64_t ne0, const int ne1) {
    constexpr int vals_per_scale = 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    const int64_t base_idx = (int64_t) blockIdx.x*s01;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + k_block*ne1 + blockIdx.x;

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (iqs % 32 == 0) {
        y[ib].d4[iqs/32] = d;
    }
}

} // namespace mq
} // namespace eng
