// The engine's rotation kernels: every Walsh-Hadamard transform the verify, the drafter and the prompt path run, with the
// chains folded around them (the residual add + RMS norm in front of a 5120-row, the gated delta-net's per-head norm and
// gate, the attention output's inverse xyzkv + V-rotation + gate) and the q8_1 twin the next PTQ1 matmul reads.
// Statement for statement the arithmetic of the fork's fwht.cu kernels at the engine's shapes (had.cuh's stages in the
// fork's stage order): bit-identical, tools/had_test.cu.
#include "had.cuh"
#include "kernels.h"

namespace eng {

namespace {

constexpr int HN  = 1024;   // the model's Hadamard block
constexpr int HNT = 256;    // threads per 1024-block row

// Rows of N = 64 or 256, one row per warp, 4 rows per CTA, no signs: the drafter's and the attention's head rotations.
template <int N>
__launch_bounds__(4*WARP_SIZE, 1)
__global__ void k_had_rows(const float * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int NE = N / WARP_SIZE;
    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }
    src += r * N;
    dst += r * N;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = src[i * WARP_SIZE + lane] * scale;
    }
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(reg, h, lane);
    }
    had_stages_reg(reg);
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

// 1024-block rows, 256 threads per row: row r of src, or with glu block r % n_blk of token r / n_blk of a
// [2*glu_nc, tokens] SwiGLU input (silu(gate half) * up half in the load); signs: the rotated basis' +/-1 vector of block
// r % n_blk; q8: the twin.
template <bool signed_rows, bool glu>
__launch_bounds__(HNT, 1)
__global__ void k_had_block(const float * src, float * dst, const int64_t n_rows, const float scale, const float * signs,
                            const int n_blk, char * __restrict__ q8, const int64_t glu_nc) {
    __shared__ float s[HN];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    ggml_cuda_pdl_sync();
    had_block_row<HNT, signed_rows, glu>(src, dst, r, scale, signs, n_blk, q8, glu_nc, (int) threadIdx.x, s);
}

// The 256-thread norm/Hadamard kernel processes four virtual 1024-thread lanes per thread. Virtual warp sums land in the
// same slots and the final butterfly preserves the norm reduction order; the Hadamard network is independent of thread
// count and each q8 block still covers the same 32 consecutive values.
template <bool do_add, bool write_norm>
__launch_bounds__(256, 1)
__global__ void k_norm_had_5k_256(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x_out,
                                  const float * __restrict__ w, const float * __restrict__ signs, float * __restrict__ norm_out,
                                  float * __restrict__ dst, const float eps, const float fwht_scale, char * __restrict__ q8) {
    constexpr int VB    = 1024;              // the virtual block (k_norm_had_5k's threads)
    constexpr int NT    = 256;
    constexpr int NV    = VB / NT;
    constexpr int NCOLS = 5120;
    constexpr int NPT   = NCOLS / VB;
    constexpr int NBLK  = NCOLS / HN;
    static_assert(VB / WARP_SIZE == WARP_SIZE, "the final butterfly reads one virtual warp sum per lane");

    __shared__ float s_sum[32];
    __shared__ float s[HN];

    const int     row  = blockIdx.x / NBLK;
    const int     blk  = blockIdx.x % NBLK;
    const int     tid  = threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int64_t off  = (int64_t) row*NCOLS;

    float wv[NV], sg[NV];
#pragma unroll
    for (int m = 0; m < NV; ++m) {   // weights: not written by the previous kernel
        wv[m] = w[blk*HN + tid + m*NT];
        sg[m] = signs[blk*HN + tid + m*NT];
    }

    ggml_cuda_pdl_sync();
    float xb[NV], tmp[NV];
#pragma unroll
    for (int m = 0; m < NV; ++m) {
        const int vt = tid + m*NT;
        float xv[NPT];
        if constexpr (do_add) {
            float av[NPT], bv[NPT];
#pragma unroll
            for (int j = 0; j < NPT; ++j) {
                av[j] = a[off + vt + j*VB];
                bv[j] = b[off + vt + j*VB];
            }
#pragma unroll
            for (int j = 0; j < NPT; ++j) {
                xv[j] = av[j] + bv[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < NPT; ++j) {
                xv[j] = a[off + vt + j*VB];
            }
        }
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            t += xv[j] * xv[j];
        }
        tmp[m] = t;
        xb[m]  = xv[blk];
        if constexpr (do_add) {
            x_out[off + blk*HN + vt] = xv[blk];
        }
    }
#pragma unroll
    for (int m = 0; m < NV; ++m) {
        tmp[m] = warp_reduce_sum(tmp[m]);
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < NV; ++m) {
            s_sum[tid/WARP_SIZE + m*(NT/WARP_SIZE)] = tmp[m];
        }
    }
    __syncthreads();
    const float total = warp_reduce_sum(s_sum[lane]);
    const float mean  = total / NCOLS;
    const float scale = rsqrtf(mean + eps);

    float v[NV];
#pragma unroll
    for (int m = 0; m < NV; ++m) {
        const float y = scale * xb[m] * wv[m];
        if constexpr (write_norm) {
            norm_out[off + blk*HN + tid + m*NT] = y;
        }
        v[m] = y * fwht_scale;
        v[m] *= sg[m];
    }
    had_all<HN, NT>(v, s, tid);

    const int64_t base = ((int64_t) row*NBLK + blk)*HN;
#pragma unroll
    for (int m = 0; m < NV; ++m) {
        dst[base + tid + m*NT] = v[m];
        if (q8 != nullptr) {
            q8_twin_store(q8, base + tid + m*NT, v[m]);
        }
    }
}

// The gated delta-net's output chain per (token, 1024-block): 8 heads of 128, each RMS-normed, * w * silu(gate), gathered
// from the recurrence's [16 key heads x 3] order (block g's source head is g/3 + 16*(g%3)), then signs + 1024-block
// Hadamard + twin. The fork normed the 8 heads one after another, each a 256-thread block_reduce (threads 0..127 hold x^2,
// 128..255 zero): warp j's butterfly over elements 32j..32j+31 -> W_j, then one warp's butterfly over [W0..W3, 0 ...],
// which is (W0 + W2) + (W1 + W3). Here warp lh owns head lh: lane l holds elements l + 32j, the same four butterflies give
// the same W_j, and the same two adds give the same sum -- bit-identical (tools/had_test.cu), 8 serial reductions -> 1.
__launch_bounds__(HNT, 1)
__global__ void k_gdn_out(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ gate,
                          const float * __restrict__ signs, float * __restrict__ dst, const float eps, const float fwht_scale,
                          char * __restrict__ q8) {
    constexpr int HD   = 128;
    constexpr int NK   = 16;
    constexpr int REP  = 3;
    constexpr int NE   = HN / HNT;
    constexpr int NBLK = HD*NK*REP/HN;
    constexpr int NH   = HN/HD;
    static_assert(NH == HNT/WARP_SIZE && HD == 4*WARP_SIZE, "one warp per head, four elements per lane");

    __shared__ float s_values[HN];

    const int r     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int token = r / NBLK;
    const int blk   = r % NBLK;
    const int lh    = tid / WARP_SIZE;
    const int lane  = tid % WARP_SIZE;

    const int g           = blk*NH + lh;
    const int source_head = g / REP + NK*(g % REP);
    const int64_t off     = ((int64_t) token*(NK*REP) + source_head)*HD;

    ggml_cuda_pdl_sync();
    float xs[4], ws[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xs[j] = x[off + lane + j*WARP_SIZE];
        ws[j] = warp_reduce_sum(xs[j]*xs[j]);
    }
    const float sum   = (ws[0] + ws[2]) + (ws[1] + ws[3]);
    const float scale = rsqrtf(sum/HD + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j*WARP_SIZE;
        s_values[lh*HD + i] = scale*xs[j]*w[i]*act_silu(gate[off + i]);
    }
    __syncthreads();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int c = i*HNT + tid;
        reg[i] = s_values[c]*fwht_scale;
        reg[i] *= signs[blk*HN + c];
    }
    had_all<HN, HNT>(reg, s_values, tid);

    const int64_t base = (int64_t) r*HN;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[base + i*HNT + tid] = reg[i];
        if (q8 != nullptr) {
            q8_twin_store(q8, base + i*HNT + tid, reg[i]);
        }
    }
}

// The attention layer's output chain per (token, 1024-block = 4 heads of 256; register i of thread tid is element tid of
// head blk*4 + i): xyzkv's inverse 128-point WHT with the InnerQ scale, the V rotation's 64-point Hadamard, the
// output gate sigmoid(g) * x, then signs + 1024-block Hadamard + twin.
__launch_bounds__(HNT, 1)
__global__ void k_attn_out(const float * __restrict__ x, const float * __restrict__ gate, const int64_t gate_s1,
                           const int64_t gate_s2, const float * __restrict__ scale_inv, const float * __restrict__ signs,
                           float * __restrict__ dst, const float scale64, const float scale1k, char * __restrict__ q8) {
    constexpr int HD   = 256;
    constexpr int NC   = 6144;
    constexpr int NE   = HN / HNT;
    constexpr int NBLK = NC / HN;
    static_assert(HNT == HD && NE == 4, "register i of thread tid is element tid of head blk*NE + i");

    __shared__ float s[HN];

    const int r     = blockIdx.x;
    const int token = r / NBLK;
    const int blk   = r % NBLK;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int t     = tid % 128;

    ggml_cuda_pdl_sync();
    float reg[NE], g[NE], sg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int head = blk*NE + i;
        reg[i] = x[(int64_t) token*NC + head*HD + tid];
        g[i]   = gate[(int64_t) token*gate_s2 + (int64_t) head*gate_s1 + tid];
        sg[i]  = signs[blk*HN + i*HNT + tid];
    }
    const float si = scale_inv != nullptr ? scale_inv[t] : 1.0f;
    const float s1 = xyzkv_s1[t];
    const float s2 = xyzkv_s2[t];

    // 1. xyzkv's inverse WHT on each 128-group
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] *= s2;
    }
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(reg, h, lane);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < 128; h *= 2) {
        had_stage_smem<NE, HNT>(reg, s, h, tid);
    }
    constexpr float inv_sqrt = 0.08838834764831845f;   // 1/sqrt(128)
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * inv_sqrt * s1;
        if (scale_inv != nullptr) {
            reg[i] *= si;
        }
    }

    // 2. the V rotation's 64-point Hadamard
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * scale64;
    }
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(reg, h, lane);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < 64; h *= 2) {
        had_stage_smem<NE, HNT>(reg, s, h, tid);
    }

    // 3. the output gate
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = act_sigmoid(g[i]) * reg[i];
    }

    // 4. signs + 1024-block Hadamard
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = reg[i] * scale1k;
        reg[i] *= sg[i];
    }
    had_all<HN, HNT>(reg, s, tid);

    const int64_t base = (int64_t) r*HN;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[base + i*HNT + tid] = reg[i];
        if (q8 != nullptr) {
            q8_twin_store(q8, base + i*HNT + tid, reg[i]);
        }
    }
}

} // namespace

void norm_fwht(cudaStream_t st, const float * a, const float * b, float * x_out, const float * w, const float * signs,
               float * norm_out, float * dst, char * q8, int nrows, float eps) {
    const float fwht_scale = 1 / sqrtf(1024);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (nrows*5), 1, 1), dim3(256, 1, 1), 0, st);
#define ENG_RNF(ADD, NORM) \
    ggml_cuda_kernel_launch(k_norm_had_5k_256<ADD, NORM>, lp, a, b, x_out, w, signs, norm_out, dst, eps, fwht_scale, q8)
    if (b != nullptr) {
        if (norm_out) { ENG_RNF(true, true); } else { ENG_RNF(true, false); }
    } else {
        if (norm_out) { ENG_RNF(false, true); } else { ENG_RNF(false, false); }
    }
#undef ENG_RNF
}

void fwht_block(cudaStream_t st, const float * src, float * dst, int64_t rows, const float * signs, int n_blk, char * q8) {
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) rows, 1, 1), dim3(HNT, 1, 1), 0, st);
    if (signs) {
        ggml_cuda_kernel_launch(k_had_block<true, false>, lp, src, dst, rows, 1 / sqrtf(1024), signs, n_blk, q8, (int64_t) 0);
    } else {
        ggml_cuda_kernel_launch(k_had_block<false, false>, lp, src, dst, rows, 1 / sqrtf(1024), (const float *) nullptr, 1,
            q8, (int64_t) 0);
    }
}

void fwht_glu(cudaStream_t st, const float * glu_src, float * dst, int64_t rows, const float * signs, int n_blk, char * q8,
              int64_t glu_nc) {
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) rows, 1, 1), dim3(HNT, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_had_block<true, true>, lp, glu_src, dst, rows, 1 / sqrtf(1024), signs, n_blk, q8, glu_nc);
}

void gdn_out_chain(cudaStream_t st, const float * x, const float * w, const float * gate, const float * signs, float * dst,
                   float eps, char * q8, int n_tokens) {
    const float fwht_scale = 1/sqrtf(1024.0f);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_tokens*6), 1, 1), dim3(HNT, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_gdn_out, lp, x, w, gate, signs, dst, eps, fwht_scale, q8);
}

void attn_out_chain(cudaStream_t st, const float * x, const float * gate, int64_t gate_s1, int64_t gate_s2,
                    const float * scale_inv, const float * signs, float * dst, char * q8, int n_tokens) {
    const float scale64 = 1 / sqrtf(64);
    const float scale1k = 1 / sqrtf(1024);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_tokens*6), 1, 1), dim3(HNT, 1, 1), 0, st);
    ggml_cuda_kernel_launch(k_attn_out, lp, x, gate, gate_s1, gate_s2, scale_inv, signs, dst, scale64, scale1k, q8);
}

void fwht_rows(cudaStream_t st, const float * src, float * dst, int n, int64_t rows) {
    const dim3 g((unsigned) ((rows + 3) / 4), 1, 1), b(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params lp(g, b, 0, st);
    if (n == 256) {
        ggml_cuda_kernel_launch(k_had_rows<256>, lp, src, dst, rows, 1 / sqrtf(256));
    } else if (n == 64) {
        ggml_cuda_kernel_launch(k_had_rows<64>, lp, src, dst, rows, 1 / sqrtf(64));
    } else {
        fprintf(stderr, "fwht_rows: n %d not instantiated\n", n);
        abort();
    }
}

} // namespace eng
