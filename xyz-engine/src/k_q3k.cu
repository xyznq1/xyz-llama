// The engine's drafter matvecs: Q3_K weights (every matrix of the xyz2 drafter) x q8_1 activations at 1..8 columns, with
// the fused epilogues the drafter's step uses (the residual add; gate + up + SwiGLU), and the q8_1 quantizer that feeds
// them. The fork's mul_mat_vec_q<Q3_K, ncols> at sm_89's parameter table (GENERIC): 4 warps at 1..4 columns, 2 at 5..8;
// thread tid takes the super-blocks tid/16 + n*blocks_per_iter and the 16th kqs = tid % 16 of each; the same dp4a,
// scale and float statements; warps 1.. added to warp 0 in order, then the warp butterfly -- bit-identical
// (tools/q3k_test.cu). Rows per CTA: 1 at one column, 2 above; a row's arithmetic does not depend on it.
#include "had.cuh"   // act_silu

#include "kernels.h"

namespace eng {

namespace {

__host__ __device__ constexpr int q3k_nwarps(int ncols) { return ncols <= 4 ? 4 : 2; }
__host__ __device__ constexpr int q3k_rows(int ncols)   { return ncols == 1 ? 1 : 2; }

__device__ __forceinline__ int ld_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int ld_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

// the 16th iqs of super-block kbx against its four q8_1 blocks: per 32-value group i, the 6-bit scale, the 2 low bits
// and the inverted high bit (4 subtracted where it is 0), one dp4a, d8 * (dot * scale); then d * the sum
__device__ __forceinline__ float q3k_dot(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                         const int & kbx, const int & iqs) {
    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset   = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = ld_b2(bq3_K->qs, iqs);
    const int vh = ~ld_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int   u[QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = ld_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    const uint8_t * scales = bq3_K->scales;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// q3k_dot split in two: the q8_1 side (u, d8) of one 16th, loaded once, then any number of rows against it with exactly
// q3k_dot's statements
__device__ __forceinline__ void q8_pre(const block_q8_1 * __restrict__ bq8_1, const int & iqs, int (&u)[QR3_K],
                                       float (&d8)[QR3_K]) {
    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = ld_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
}

__device__ __forceinline__ float q3k_dot_pre(const void * __restrict__ vbq, const int & kbx, const int & iqs,
                                             const int (&u)[QR3_K], const float (&d8)[QR3_K]) {
    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset   = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = ld_b2(bq3_K->qs, iqs);
    const int vh = ~ld_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    const uint8_t * scales = bq3_K->scales;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// up to three matrices of one K against one q8_1 column in one launch: rows [0, end[0]) of w[0] into dst[0], then w[1]'s
// rows, then w[2]'s (Q, K and V of a drafter step); a single matrix sets end[1] = end[2] = end[0]
struct Seg3 {
    const void * w[3];
    float *      dst[3];
    int          end[3];
};

// one column, R rows per CTA: row r's arithmetic is exactly k_q3k_mv<1>'s (logical thread tid takes the same super-blocks
// and 16th in the same order, warps 1..3 join warp 0 in order, then the butterfly), so every output bit is the same; the
// CTA's R rows share each 16th's q8_1 loads. Every segment's rows % R == 0 (every drafter matrix).
// MINB: the register cap as CTAs per SM (a memory-bound matvec wants the occupancy; the arithmetic is the same either way)
template <int R, bool fused, int MINB>
__launch_bounds__(4*WARP_SIZE, MINB)
__global__ void k_q3k_mv1r(const Seg3 seg, const void * __restrict__ vgate, const block_q8_1 * __restrict__ y,
                           const float * __restrict__ x_bias, const int ncols_x, const int stride_row_x) {
    constexpr int nwarps = 4;
    constexpr int blocks_per_iter = nwarps*WARP_SIZE / QI3_K;

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int grow = R*blockIdx.x;
    // selected, not indexed: a runtime index into the parameter struct spills it to the stack
    const bool s0 = grow < seg.end[0], s1 = !s0 && grow < seg.end[1];
    const int row0 = grow - (s0 ? 0 : (s1 ? seg.end[0] : seg.end[1]));
    const void * __restrict__ vx = s0 ? seg.w[0] : (s1 ? seg.w[1] : seg.w[2]);
    float * __restrict__ dst     = s0 ? seg.dst[0] : (s1 ? seg.dst[1] : seg.dst[2]);
    const int blocks_per_row_x = ncols_x / QK_K;

    ggml_cuda_pdl_sync();
    const bool use_gate = fused && vgate != nullptr;
    const bool use_bias = fused && x_bias != nullptr;

    float x_bias_r = 0.0f;                 // k_q3k_mv's x_biases[0] / gate_biases[0]: 0.0f unless loaded, always added
    const float gate_bias_r = 0.0f;
    if constexpr (fused) {
        if (use_bias && threadIdx.x < R && threadIdx.y == 0) {
            x_bias_r = x_bias[row0 + threadIdx.x];
        }
    }

    float tmp[R] = {0.0f};
    float tmp_gate[R] = {0.0f};

    const int kqs = tid % QI3_K;
    for (int kbx = tid / QI3_K; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        int   u[QR3_K];
        float d8[QR3_K];
        q8_pre(&y[kbx * (QK_K/QK8_1)], kqs, u, d8);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int x_index = (row0 + r)*stride_row_x + kbx;
            tmp[r] += q3k_dot_pre(vx, x_index, kqs, u, d8);
            if constexpr (fused) {
                if (use_gate) {
                    tmp_gate[r] += q3k_dot_pre(vgate, x_index, kqs, u, d8);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][R][WARP_SIZE];
    __shared__ float tmp_shared_gate[fused ? nwarps-1 : 1][R][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            tmp_shared[threadIdx.y-1][r][threadIdx.x] = tmp[r];
            if constexpr (fused) {
                if (use_gate) {
                    tmp_shared_gate[threadIdx.y-1][r][threadIdx.x] = tmp_gate[r];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[r] += tmp_shared[l][r][threadIdx.x];
            if constexpr (fused) {
                if (use_gate) {
                    tmp_gate[r] += tmp_shared_gate[l][r][threadIdx.x];
                }
            }
        }
        tmp[r] = warp_reduce_sum<WARP_SIZE>(tmp[r]);
        if constexpr (fused) {
            if (use_gate) {
                tmp_gate[r] = warp_reduce_sum<WARP_SIZE>(tmp_gate[r]);
            }
        }
        if (threadIdx.x == r) {
            float result = tmp[r];
            if constexpr (fused) {
                result += x_bias_r;
                if (use_gate) {
                    float gate_value = tmp_gate[r];
                    gate_value += gate_bias_r;
                    result *= act_silu(gate_value);   // SwiGLU
                }
            }
            dst[row0 + r] = result;
        }
    }
}

// dst[j*stride_col_dst + row] = W[row] . y[j] (+ x_bias; with vgate: * silu(G[row] . y[j]))
template <int ncols, bool fused>
__launch_bounds__(q3k_nwarps(ncols)*WARP_SIZE, 1)
__global__ void k_q3k_mv(const void * __restrict__ vx, const void * __restrict__ vgate, const block_q8_1 * __restrict__ y,
                         const float * __restrict__ x_bias, float * __restrict__ dst, const int ncols_x,
                         const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    constexpr int qk     = QK_K;
    constexpr int qi     = QI3_K;
    constexpr int vdr    = 1;
    constexpr int nwarps = q3k_nwarps(ncols);
    constexpr int rpb    = q3k_rows(ncols);

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = rpb*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    ggml_cuda_pdl_sync();
    const bool use_gate = fused && vgate != nullptr;
    const bool use_bias = fused && x_bias != nullptr;

    float x_biases[ncols] = { 0.0f };
    float gate_biases[ncols] = { 0.0f };
    if constexpr (fused) {
        if (threadIdx.x < rpb && threadIdx.y == 0 && (rpb == 1 || uint32_t(row0 + threadIdx.x) < (uint32_t) stride_col_dst)) {
            if (use_bias) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    x_biases[j] = x_bias[row0 + j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    float tmp[ncols][rpb] = {{0.0f}};
    float tmp_gate[ncols][rpb] = {{0.0f}};

    const int kbx_offset = row0*stride_row_x;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            const int x_index = kbx_offset + i*stride_row_x + kbx;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                tmp[j][i] += q3k_dot(vx, &y[j*stride_col_y + kby], x_index, kqs);
                if constexpr (fused) {
                    if (use_gate) {
                        tmp_gate[j][i] += q3k_dot(vgate, &y[j*stride_col_y + kby], x_index, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][ncols][rpb][WARP_SIZE];
    __shared__ float tmp_shared_gate[fused ? nwarps-1 : 1][ncols][rpb][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (fused) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += row0;
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (fused) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<WARP_SIZE>(tmp[j][i]);
            if constexpr (fused) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<WARP_SIZE>(tmp_gate[j][i]);
                }
            }
            if (threadIdx.x == i && (rpb == 1 || uint32_t(row0 + i) < (uint32_t) stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (fused) {
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        gate_value += gate_biases[j];
                        result *= act_silu(gate_value);   // SwiGLU
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }
}

// row i1 of x (ne00 values, stride s01) into q8_1 blocks, padded with zeros to ne0 values (MATRIX_ROW_PADDING)
__launch_bounds__(256, 1)
__global__ void k_q8_1(const float * __restrict__ x, block_q8_1 * __restrict__ y, const int64_t ne00, const int64_t s01,
                       const int64_t ne0) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i1 = blockIdx.y;
    const int64_t i_cont = i1*ne0 + i0;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i1*s01 + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}

void check_q3k(int type) {
    if (type != GGML_TYPE_Q3_K) {
        fprintf(stderr, "xyz-engine: the drafter matvec is Q3_K; got type %d\n", type);
        abort();
    }
}

void quantize_cols(cudaStream_t st, const float * x, int K, int ncols, char * q8) {
    const int64_t Kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) ((Kp + 255)/256), (unsigned) ncols, 1), dim3(256, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_q8_1, lp, x, (block_q8_1 *) q8, (int64_t) K, (int64_t) K, Kp);
}

template <int ncols, bool fused>
void launch_mv(cudaStream_t st, const void * w, const void * gate, const char * q8, const float * x_bias, float * dst, int K,
               int nrows) {
    const int s01 = K / QK_K;
    const int s11 = (int) (GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1);
    const dim3 grid((unsigned) ((nrows + q3k_rows(ncols) - 1) / q3k_rows(ncols)), 1, 1), block(WARP_SIZE, q3k_nwarps(ncols), 1);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, st);
    ggml_cuda_kernel_launch(k_q3k_mv<ncols, fused>, lp, w, gate, (const block_q8_1 *) q8, x_bias, dst, K, s01, s11, nrows);
}

// The register cap is 4 CTAs per SM (<= 128 registers) or 8 (<= 64). The SwiGLU pair uses one row at cap 8; every other
// call uses eight rows at cap 4.
template <int R, bool fused>
void launch_mv1r(cudaStream_t st, const Seg3 & seg, const void * gate, const char * q8, const float * x_bias, int K, int minb) {
    const dim3 grid((unsigned) (seg.end[2] / R), 1, 1), block(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, st);
    if (minb == 8) {
        ggml_cuda_kernel_launch(k_q3k_mv1r<R, fused, 8>, lp, seg, gate, (const block_q8_1 *) q8, x_bias, K, K / QK_K);
    } else {
        ggml_cuda_kernel_launch(k_q3k_mv1r<R, fused, 4>, lp, seg, gate, (const block_q8_1 *) q8, x_bias, K, K / QK_K);
    }
}

template <int R>
void mv1r(cudaStream_t st, const Seg3 & seg, const void * gate, const char * q8, const float * x_bias, int K, int minb) {
    if (gate != nullptr || x_bias != nullptr) {
        launch_mv1r<R, true>(st, seg, gate, q8, x_bias, K, minb);
    } else {
        launch_mv1r<R, false>(st, seg, nullptr, q8, nullptr, K, minb);
    }
}

void mv1r_rows(cudaStream_t st, int rows, const Seg3 & seg, const void * gate, const char * q8, const float * x_bias, int K,
               int minb = 4) {
    for (int i = 0; i < 3; ++i) {
        if ((seg.end[i] - (i ? seg.end[i - 1] : 0)) % rows != 0) {
            fprintf(stderr, "xyz-engine: a %d-row segment is not a multiple of %d rows\n", seg.end[i] - (i ? seg.end[i - 1] : 0), rows);
            abort();
        }
    }
    switch (rows) {
        case 1: mv1r<1>(st, seg, gate, q8, x_bias, K, minb); break;
        case 2: mv1r<2>(st, seg, gate, q8, x_bias, K, minb); break;
        case 4: mv1r<4>(st, seg, gate, q8, x_bias, K, minb); break;
        case 8: mv1r<8>(st, seg, gate, q8, x_bias, K, minb); break;
        default: fprintf(stderr, "xyz-engine: mmvq_rows at %d rows\n", rows); abort();
    }
}

} // namespace

void mmvq_qkv(cudaStream_t st, int rows, int type, const float * x, int K, char * q8, const void * wq, int nq, float * q,
              const void * wk, int nk, float * k, const void * wv, int nv, float * v) {
    check_q3k(type);
    quantize_cols(st, x, K, 1, q8);
    const Seg3 seg = { { wq, wk, wv }, { q, k, v }, { nq, nq + nk, nq + nk + nv } };
    mv1r_rows(st, rows, seg, nullptr, q8, nullptr, K);
}

void mmvq_fast(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, float * dst, char * q8,
               const void * gate, int glu_op, const float * x_bias) {
    check_q3k(type);
    if (gate != nullptr && glu_op != GGML_GLU_OP_SWIGLU) {
        fprintf(stderr, "xyz-engine: mmvq_fast fuses SwiGLU only; got glu op %d\n", glu_op);
        abort();
    }
    quantize_cols(st, x, K, 1, q8);
    const Seg3 seg = { { w, w, w }, { dst, dst, dst }, { nrows, nrows, nrows } };
    if (gate != nullptr) {
        mv1r_rows(st, 1, seg, gate, q8, x_bias, K, 8);
    } else {
        mv1r_rows(st, 8, seg, nullptr, q8, x_bias, K, 4);
    }
}

void q8_1_quantize(cudaStream_t st, const float * x, int K, int ncols, char * q8) {
    quantize_cols(st, x, K, ncols, q8);
}

void mmvq_n(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, int ncols, float * dst, char * q8) {
    check_q3k(type);
    quantize_cols(st, x, K, ncols, q8);
    switch (ncols) {
        case 1: launch_mv<1, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 2: launch_mv<2, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 3: launch_mv<3, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 4: launch_mv<4, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 5: launch_mv<5, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 6: launch_mv<6, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 7: launch_mv<7, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        case 8: launch_mv<8, false>(st, w, nullptr, q8, nullptr, dst, K, nrows); break;
        default:
            fprintf(stderr, "xyz-engine: mmvq_n at %d columns\n", ncols);
            abort();
    }
}

// the server's routing for Q3_K on Ada (ggml_cuda_should_use_mmvq): the matvec up to 6 columns, the MMQ tiles above
void mm_q(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, int ncols, float * dst, char * q8) {
    if (ncols <= 6) {
        mmvq_n(st, type, w, x, K, nrows, ncols, dst, q8);
    } else {
        mmq(st, type, w, K, nrows, x, K, ncols, dst, nrows);
    }
}

} // namespace eng
