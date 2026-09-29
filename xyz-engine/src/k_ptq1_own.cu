// The engine's own PTQ1_0 x q8_1 tensor-core matmul for the verify's widths (1..8 columns): the arithmetic of the fork's
// mul_mat_ptq1_mma<1, 4, false, ILV, false, *, isum> -- the same fragments, the same IMMA inputs and -sum(q) starts, the
// same float products and sums in the same order (per k-block: blk += d8 * C over its 4 slices; per warp: acc += d * blk
// over its k-blocks kb = w, w + 4, ...; then warps 1..3 added to warp 0 in order) -- so every output is bit-identical.
// What is its own: the schedule. A warp takes U k-blocks per iteration (kb, kb + 4, ...): their weight words, activation
// records, decodes and IMMAs are independent, only the U acc updates stay in order, so one block's load and decode
// latency hides under the other's. The long-K one-wave shapes (FFN down, K 17408: 34 blocks per warp, 320 tiles on 66
// SMs) are this chain's latency; the many-wave shapes keep the bus full either way.
#include "common.cuh"
#include "mma.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#include "mmvq-ptq1-mma.cuh"

#include "had.cuh"
#include "kernels.h"

namespace eng {

namespace {

// one lane's activation words for k-block kb (the ptq1_perm record: its B column's 8 fragment words, its two C
// columns' (d, -sum q) pairs), exactly as mul_mat_ptq1_mma_tile's ACT_PERM loads them
struct Act {
    int   bw[8];
    half2 dsa[4];
    half2 dsb[4];
};

// A lane's three activation streams (its B column's fragment words, its two C columns' (d, -sum q) pairs) at k-block 0,
// set up once per kernel. A column past ncols reads column 0 of y instead of branching around its loads: its B and C
// columns only reach outputs the store loop skips (col >= ncols), so every stored value is unchanged.
struct ActPtr {
    const char * b;
    const char * c0;
    const char * c1;
};

// col0: the n-tile's first column; the fallback is column 0 of y itself, never a column past the buffer
static __device__ __forceinline__ ActPtr act_ptr(const block_q8_1 * __restrict__ y, const int stride_col_y, const int col0,
                                                 const int g, const int c, const int ncols) {
    const int colB = col0 + g;
    const int colC = col0 + 2*c;
    ActPtr p;
    p.b  = (const char *) (y + (colB     < ncols ? colB     : 0)*stride_col_y) + 32*c;
    p.c0 = (const char *) (y + (colC     < ncols ? colC     : 0)*stride_col_y) + 4*QK8_1;
    p.c1 = (const char *) (y + (colC + 1 < ncols ? colC + 1 : 0)*stride_col_y) + 4*QK8_1;
    return p;
}

static __device__ __forceinline__ void load_act(Act & a, const ActPtr & p, const int kb) {
    const size_t off = (size_t) kb*((QK_PTQ1_0/QK8_1)*sizeof(block_q8_1));   // one ptq1_perm record per k-block
    const int4 lo = *(const int4 *) (p.b + off);
    const int4 hi = *(const int4 *) (p.b + off + 16);
    const int4 za = *(const int4 *) (p.c0 + off);
    const int4 zb = *(const int4 *) (p.c1 + off);
    a.bw[0] = lo.x; a.bw[1] = lo.y; a.bw[2] = lo.z; a.bw[3] = lo.w;
    a.bw[4] = hi.x; a.bw[5] = hi.y; a.bw[6] = hi.z; a.bw[7] = hi.w;
    memcpy(a.dsa, &za, sizeof(za));
    memcpy(a.dsb, &zb, sizeof(zb));
}

// one k-block's contribution: blk[l] = sum over its 4 slices of d8 * IMMA (the tile's inner loop, ntiles 1)
template <bool isum>
static __device__ __forceinline__ void block_mma(const ptq1_frag & fx, const Act & a, float (&blk)[4]) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    blk[0] = blk[1] = blk[2] = blk[3] = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        tile_A A;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            A.x[l] = fx.s[s][l];
        }
        tile_B B;
        B.x[0] = a.bw[2*s];
        B.x[1] = a.bw[2*s + 1];
        const float d8_0 = __low2float(a.dsa[s]);
        const float d8_1 = __low2float(a.dsb[s]);
        tile_C C;
        if constexpr (isum) {
            const int s8_0 = ptq1_neg_isum(a.dsa[s]);
            const int s8_1 = ptq1_neg_isum(a.dsb[s]);
            C.x[0] = s8_0;
            C.x[1] = s8_1;
            C.x[2] = s8_0;
            C.x[3] = s8_1;
        }
        mma(C, A, B);
        blk[0] += d8_0 * (float) C.x[0];
        blk[1] += d8_1 * (float) C.x[1];
        blk[2] += d8_0 * (float) C.x[2];
        blk[3] += d8_1 * (float) C.x[3];
    }
}

static __device__ __forceinline__ int glu_ld_relaxed(const int32_t * p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ int glu_ld_acquire(const int32_t * p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// GLU (the FFN's gate_up, merged [gate; up] rows = 2 x glu_nc): the SwiGLU + signed 1024-point Hadamard + q8_1 twin that
// fwht_cuda_block<1024, 256, true, true> runs after the matmul, done by the matmul's own last CTAs. Block b's rows (one
// per column: row r = col*n_blk + b) read gate rows [b*1024, +1024) and up rows [glu_nc + b*1024, +1024): 128 tiles. Each
// tile counts itself into cnt[b] after its outputs (release); the last ncols tiles of block b each take one row, wait for
// the count to reach 128 (acquire) and run had.cuh's had_block_row on 128 threads (its butterfly network does not depend
// on the thread count: bit-identical). The last row of a block resets its counters (a graph replays the launch as is).
struct GluArgs {
    float *       dst;      // the Hadamard's output rows [ncols*n_blk][1024]
    const float * signs;    // [glu_nc]
    char *        q8;       // the down matmul's q8_1 twin (a buffer the matmul does NOT read)
    int32_t *     cnt;      // [n_blk] tiles done
    int32_t *     fin;      // [n_blk] rows done
    int           glu_nc;
    float         scale;
};

template <bool isum, int U, int MINB, bool GLU = false>
__launch_bounds__(128, MINB)
static __global__ void k_ptq1_own(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int K, const int nrows, const int ncols, const int dst_stride,
                                  float * __restrict__ hi, const int hi_row0, const int hi_stride,
                                  float * __restrict__ hi2, const int hi2_row0, const int hi2_stride, const GluArgs ga) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int nwarps = 4;
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int c    = lane & 3;
    const int row0 = blockIdx.x*16;

    const int nblocks      = K / QK_PTQ1_0;
    const int stride_row_x = nblocks;
    const int stride_col_y = K / QK8_1;
    const int ra = min(row0 + g,     nrows - 1);
    const int rb = min(row0 + g + 8, nrows - 1);
    const bool tile_ilv = row0 + 16 <= (nrows & ~15);
    const int  kstep    = tile_ilv ? 16 : 1;
    const int  oa       = tile_ilv ? row0*stride_row_x + g     : ra*stride_row_x;
    const int  ob       = tile_ilv ? row0*stride_row_x + g + 8 : rb*stride_row_x;
    const block_ptq1_0 * xa = (const block_ptq1_0 *) vx + oa;
    const block_ptq1_0 * xb = (const block_ptq1_0 *) vx + ob;
    const block_q8_1 *   y  = (const block_q8_1 *) vy;
    const ActPtr         ap = act_ptr(y, stride_col_y, 0, g, c, ncols);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    // U blocks per iteration: kb = kb0 + u*nwarps
    for (int kb0 = w; kb0 < nblocks; kb0 += U*nwarps) {
        ptq1_words wa[U], wb[U];
        Act        act[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kb = kb0 + u*nwarps;
            if (kb < nblocks) {
                wa[u] = ptq1_load_words(xa + kb*kstep, c);
                wb[u] = ptq1_load_words(xb + kb*kstep, c);
                load_act(act[u], ap, kb);
            }
        }
        float blk[U][4];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (kb0 + u*nwarps < nblocks) {
                ptq1_frag fx;
                ptq1_decode_words<isum>(wa[u], wb[u], c, fx);
                block_mma<isum>(fx, act[u], blk[u]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {   // in k-block order: acc += d * blk
            if (kb0 + u*nwarps < nblocks) {
                const float da = ptq1_word_scale(wa[u].hd);
                const float db = ptq1_word_scale(wb[u].hd);
                acc[0] += da * blk[u][0];
                acc[1] += da * blk[u][1];
                acc[2] += db * blk[u][2];
                acc[3] += db * blk[u][3];
            }
        }
    }

    // warps 1..3 hand their partial sums to warp 0, which adds them in warp order
    __shared__ float red[nwarps - 1][4][WARP_SIZE];
    if (w > 0) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            red[w - 1][l][lane] = acc[l];
        }
    }
    __syncthreads();
    if (w == 0) {
#pragma unroll
        for (int v = 0; v < nwarps - 1; ++v) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[l] += red[v][l][lane];
            }
        }
        float * d        = dst;
        int     d_stride = dst_stride;
        int     d_row0   = 0;
        if (hi2 != nullptr && row0 >= hi2_row0) {
            d = hi2; d_stride = hi2_stride; d_row0 = hi2_row0;
        } else if (hi != nullptr && row0 >= hi_row0) {
            d = hi; d_stride = hi_stride; d_row0 = hi_row0;
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = row0 + g + (l >> 1)*8;
            const int col = 2*c + (l & 1);
            if (row >= nrows || col >= ncols) {
                continue;
            }
            d[col*d_stride + row - d_row0] = acc[l];
        }
    }
    if constexpr (GLU) {
        __shared__ float s_row[1024];
        __shared__ int   s_k;
        const int tid  = w*WARP_SIZE + lane;
        const int half = row0 >= ga.glu_nc ? 1 : 0;
        const int b    = (row0 - half*ga.glu_nc) / 1024;
        const int n_blk = ga.glu_nc / 1024;
        __syncthreads();   // warp 0's outputs are written
        if (tid == 0) {
            __threadfence();
            const int v = atomicAdd(&ga.cnt[b], 1);
            s_k = v >= 128 - ncols ? v - (128 - ncols) : -1;
        }
        __syncthreads();
        const int k = s_k;
        if (k >= 0) {
            if (tid == 0) {
                while (glu_ld_relaxed(&ga.cnt[b]) < 128) {
                    __nanosleep(64);
                }
                (void) glu_ld_acquire(&ga.cnt[b]);
            }
            __syncthreads();
            had_block_row<128, true, true>(dst, ga.dst, (int64_t) k*n_blk + b, ga.scale, ga.signs, n_blk, ga.q8, ga.glu_nc,
                                           tid, s_row);
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                if (atomicAdd(&ga.fin[b], 1) == ncols - 1) {
                    ga.cnt[b] = 0;
                    ga.fin[b] = 0;
                }
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// NT 8-column tiles (9..32 columns: the prompt chunks), one k-block per warp iteration as the fork's ntiles > 1 launch
// runs it. An element's arithmetic does not depend on the tiles beside it -- blk[nt] over the 4 slices, acc[nt] += d * blk
// over the warp's k-blocks, warps 1..3 added to warp 0 in order -- so every column is the one-tile kernel's.
template <bool isum, int NT>
static __device__ __forceinline__ void block_mma_nt(const ptq1_frag & fx, const Act (&a)[NT], float (&blk)[NT][4]) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        blk[nt][0] = blk[nt][1] = blk[nt][2] = blk[nt][3] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        tile_A A;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            A.x[l] = fx.s[s][l];
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            tile_B B;
            B.x[0] = a[nt].bw[2*s];
            B.x[1] = a[nt].bw[2*s + 1];
            const float d8_0 = __low2float(a[nt].dsa[s]);
            const float d8_1 = __low2float(a[nt].dsb[s]);
            tile_C C;
            if constexpr (isum) {
                const int s8_0 = ptq1_neg_isum(a[nt].dsa[s]);
                const int s8_1 = ptq1_neg_isum(a[nt].dsb[s]);
                C.x[0] = s8_0;
                C.x[1] = s8_1;
                C.x[2] = s8_0;
                C.x[3] = s8_1;
            }
            mma(C, A, B);
            blk[nt][0] += d8_0 * (float) C.x[0];
            blk[nt][1] += d8_1 * (float) C.x[1];
            blk[nt][2] += d8_0 * (float) C.x[2];
            blk[nt][3] += d8_1 * (float) C.x[3];
        }
    }
}

// quantize_q8_1<true>: column i1 of x (ne00 values, stride s01), zero-padded to ne0, in the ptq1_perm record layout the
// tile kernels read (had.cuh q8_twin_store: the same bytes)
__launch_bounds__(256, 1)
static __global__ void k_q8_perm(const float * __restrict__ x, char * __restrict__ vy, const int64_t ne00, const int64_t s01,
                                 const int64_t ne0) {
    const int64_t i0 = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i1     = blockIdx.y;
    const int64_t i_cont = i1*ne0 + i0;
    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i1*s01 + i0] : 0.0f;
    q8_twin_store(vy, i_cont, xi);
}

template <bool isum, int NT>
__launch_bounds__(128, 2)
static __global__ void k_ptq1_own_nt(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                     const int K, const int nrows, const int ncols, const int dst_stride,
                                     float * __restrict__ hi, const int hi_row0, const int hi_stride,
                                     float * __restrict__ hi2, const int hi2_row0, const int hi2_stride) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int nwarps = 4;
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int c    = lane & 3;
    const int row0 = blockIdx.x*16;

    const int nblocks      = K / QK_PTQ1_0;
    const int stride_row_x = nblocks;
    const int stride_col_y = K / QK8_1;
    const int ra = min(row0 + g,     nrows - 1);
    const int rb = min(row0 + g + 8, nrows - 1);
    const bool tile_ilv = row0 + 16 <= (nrows & ~15);
    const int  kstep    = tile_ilv ? 16 : 1;
    const int  oa       = tile_ilv ? row0*stride_row_x + g     : ra*stride_row_x;
    const int  ob       = tile_ilv ? row0*stride_row_x + g + 8 : rb*stride_row_x;
    const block_ptq1_0 * xa = (const block_ptq1_0 *) vx + oa;
    const block_ptq1_0 * xb = (const block_ptq1_0 *) vx + ob;
    const block_q8_1 *   y  = (const block_q8_1 *) vy;
    ActPtr ap[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        ap[nt] = act_ptr(y, stride_col_y, nt*8, g, c, ncols);
    }

    float acc[NT][4] = {{0.0f}};
    for (int kb = w; kb < nblocks; kb += nwarps) {
        const ptq1_words wa = ptq1_load_words(xa + kb*kstep, c);
        const ptq1_words wb = ptq1_load_words(xb + kb*kstep, c);
        Act act[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            load_act(act[nt], ap[nt], kb);
        }
        ptq1_frag fx;
        ptq1_decode_words<isum>(wa, wb, c, fx);
        float blk[NT][4];
        block_mma_nt<isum, NT>(fx, act, blk);
        const float da = ptq1_word_scale(wa.hd);
        const float db = ptq1_word_scale(wb.hd);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            acc[nt][0] += da * blk[nt][0];
            acc[nt][1] += da * blk[nt][1];
            acc[nt][2] += db * blk[nt][2];
            acc[nt][3] += db * blk[nt][3];
        }
    }

    __shared__ float red[nwarps - 1][NT*4][WARP_SIZE];
    if (w > 0) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                red[w - 1][nt*4 + l][lane] = acc[nt][l];
            }
        }
    }
    __syncthreads();
    if (w > 0) {
        return;
    }
#pragma unroll
    for (int v = 0; v < nwarps - 1; ++v) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[nt][l] += red[v][nt*4 + l][lane];
            }
        }
    }
    float * d        = dst;
    int     d_stride = dst_stride;
    int     d_row0   = 0;
    if (hi2 != nullptr && row0 >= hi2_row0) {
        d = hi2; d_stride = hi2_stride; d_row0 = hi2_row0;
    } else if (hi != nullptr && row0 >= hi_row0) {
        d = hi; d_stride = hi_stride; d_row0 = hi_row0;
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = row0 + g + (l >> 1)*8;
            const int col = nt*8 + 2*c + (l & 1);
            if (row >= nrows || col >= ncols) {
                continue;
            }
            d[col*d_stride + row - d_row0] = acc[nt][l];
        }
    }
#else
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

} // namespace

// variant: 0 = one block per iteration, 1 = two, 2 = two at 6 CTAs/SM, 3 = four
void ptq1_own(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst, int dst_stride,
              int variant, float * hi, int hi_row0, int hi_stride, float * hi2, int hi2_row0, int hi2_stride) {
    const dim3 grid((nrows + 15) / 16, 1, 1), block(WARP_SIZE, 4, 1);
    if (ncols > 8) {   // the prompt chunks: 2 tiles to 16 columns, 4 above (the fork's (ncols + 7)/8 switch)
        if (ncols > 32) {
            fprintf(stderr, "ptq1_own: %d columns (the tile kernel takes 32)\n", ncols);
            abort();
        }
#define OWN_NT(NT) k_ptq1_own_nt<true, NT><<<grid, block, 0, st>>>(w, q8, dst, K, nrows, ncols, dst_stride, hi, hi_row0, \
        hi_stride, hi2, hi2_row0, hi2_stride)
        if (ncols <= 16) {
            OWN_NT(2);
        } else {
            OWN_NT(4);
        }
#undef OWN_NT
        return;
    }
    const GluArgs ga = {};
#define OWN_LAUNCH(U, MINB) k_ptq1_own<true, U, MINB><<<grid, block, 0, st>>>(w, q8, dst, K, nrows, ncols, dst_stride, \
        hi, hi_row0, hi_stride, hi2, hi2_row0, hi2_stride, ga)
    switch (variant) {
        case 0:  OWN_LAUNCH(1, 6); break;
        case 1:  OWN_LAUNCH(2, 5); break;
        case 2:  OWN_LAUNCH(2, 6); break;
        default: OWN_LAUNCH(4, 4); break;
    }
#undef OWN_LAUNCH
}

// the fork's v2 kernel (mmvq-ptq1-mma.cuh ptq1_mma_version): 32 columns on >= 16384 rows, more than 16 on >= 65536
static bool ptq1_v2(const int ncols, const int nrows) {
    return (ncols >= 32 && nrows >= 16384) || (ncols > 16 && nrows >= 65536);
}

namespace {

// mul_mat_ptq1_mma2<ntiles, nwr, nwk, false> (mmvq-ptq1-mma.cuh), one channel and sample: nwr warps own different 16-row
// tiles and share one k-block's activations staged in shared memory, nwk warp-groups split K; the plain q8_1 layout
// (quantize_q8_1<false>), signed trits (no -sum(q) start), acc += d * blk per k-block in each group's order, groups 1..
// added to group 0 in order; the sibling merge (dst_hi) as the fork routes it
template <int ntiles, int nwr, int nwk>
__launch_bounds__(nwr*nwk*WARP_SIZE, 1)
static __global__ void k_ptq1_v2(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                 const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
                                 const int stride_col_y, const int stride_col_dst, float * __restrict__ dst_hi,
                                 const int dst_hi_row0, const int dst_hi_stride_col) {
#if defined(TURING_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    constexpr int ncp  = ntiles*8;                               // padded columns
    constexpr int nacc = ntiles*4;
    constexpr int ys_bytes  = nwk*4*ncp*(int) sizeof(block_q8_1);
    constexpr int red_bytes = nwk > 1 ? (nwk - 1)*nwr*nacc*WARP_SIZE*(int) sizeof(float) : 0;
    __shared__ __align__(16) char smem[ys_bytes > red_bytes ? ys_bytes : red_bytes];
    block_q8_1 (*ys)[4][ncp] = (block_q8_1 (*)[4][ncp]) smem;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int wr   = w % nwr;                                    // this warp's row tile
    const int wk   = w / nwr;                                    // this warp's k-group
    const int g    = lane >> 2;
    const int c    = lane & 3;
    const int row0 = (blockIdx.x*nwr + wr)*16;

    ggml_cuda_pdl_sync();

    const int nblocks = ncols_x / QK_PTQ1_0;
    const int ra = min(row0 + g,     nrows_x - 1);
    const int rb = min(row0 + g + 8, nrows_x - 1);
    const bool tile_ilv = row0 + 16 <= (nrows_x & ~15);
    const int  kstep    = tile_ilv ? 16 : 1;
    const int  oa       = tile_ilv ? row0*stride_row_x + g     : ra*stride_row_x;
    const int  ob       = tile_ilv ? row0*stride_row_x + g + 8 : rb*stride_row_x;
    const block_ptq1_0 * xa = (const block_ptq1_0 *) vx + oa;
    const block_ptq1_0 * xb = (const block_ptq1_0 *) vx + ob;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float acc[ntiles][4] = {{0.0f}};

    const int niter = (nblocks + nwk - 1) / nwk;                  // every k-group runs the same trip count
    ptq1_words nxa = {0, 0, 0}, nxb = {0, 0, 0};
    if (wk < nblocks) {
        nxa = ptq1_load_words(xa + wk*kstep, c);
        nxb = ptq1_load_words(xb + wk*kstep, c);
    }
    for (int it = 0; it < niter; ++it) {
        const int  kb     = it*nwk + wk;
        const bool active = kb < nblocks;
        const ptq1_words cxa = nxa, cxb = nxb;
        const int kn = kb + nwk;
        if (kn < nblocks) {
            nxa = ptq1_load_words(xa + kn*kstep, c);
            nxb = ptq1_load_words(xb + kn*kstep, c);
        }
        {
            constexpr int nwords = ncp*36;
            for (int i = wr*WARP_SIZE + lane; i < nwords; i += nwr*WARP_SIZE) {
                const int col  = i / 36;
                const int r    = i - col*36;
                const int s    = r / 9;
                const int word = r - s*9;
                int v = 0;
                if (active && col < ncols_dst) {
                    v = ((const int *) (y + col*stride_col_y + kb*(QK_PTQ1_0/QK8_1)))[r];
                }
                ((int *) &ys[wk][s][col])[word] = v;
            }
        }
        __syncthreads();
        if (active) {
            ptq1_frag fx;
            ptq1_decode_words(cxa, cxb, c, fx);
            const float da = ptq1_word_scale(cxa.hd);
            const float db = ptq1_word_scale(cxb.hd);
            float blk[ntiles][4] = {{0.0f}};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                tile_A A;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    A.x[l] = fx.s[s][l];
                }
#pragma unroll
                for (int nt = 0; nt < ntiles; ++nt) {
                    const block_q8_1 & yb = ys[wk][s][nt*8 + g];     // column g of this n-tile (zeros past ncols)
                    tile_B B;
                    B.x[0] = get_int_b4(yb.qs, c);
                    B.x[1] = get_int_b4(yb.qs, 4 + c);
                    const float d8_0 = __low2float(ys[wk][s][nt*8 + 2*c    ].ds);
                    const float d8_1 = __low2float(ys[wk][s][nt*8 + 2*c + 1].ds);
                    tile_C C;
                    mma(C, A, B);
                    blk[nt][0] += d8_0 * (float) C.x[0];
                    blk[nt][1] += d8_1 * (float) C.x[1];
                    blk[nt][2] += d8_0 * (float) C.x[2];
                    blk[nt][3] += d8_1 * (float) C.x[3];
                }
            }
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
                acc[nt][0] += da * blk[nt][0];
                acc[nt][1] += da * blk[nt][1];
                acc[nt][2] += db * blk[nt][2];
                acc[nt][3] += db * blk[nt][3];
            }
        }
        __syncthreads();                                         // the next stage overwrites ys
    }

    if constexpr (nwk > 1) {
        float (*red)[nwr][nacc][WARP_SIZE] = (float (*)[nwr][nacc][WARP_SIZE]) smem;
        if (wk > 0) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    red[wk - 1][wr][nt*4 + l][lane] = acc[nt][l];
                }
            }
        }
        __syncthreads();
        if (wk > 0) {
            return;
        }
#pragma unroll
        for (int v = 0; v < nwk - 1; ++v) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[nt][l] += red[v][wr][nt*4 + l][lane];
                }
            }
        }
    }

    float * d        = dst;
    int     d_stride = stride_col_dst;
    int     d_row0   = 0;
    if (dst_hi != nullptr && row0 >= dst_hi_row0) {   // sibling merge: this 16-row tile is B's
        d        = dst_hi;
        d_stride = dst_hi_stride_col;
        d_row0   = dst_hi_row0;
    }
#pragma unroll
    for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = row0 + g + (l >> 1)*8;
            const int col = nt*8 + 2*c + (l & 1);
            if (row >= nrows_x || col >= ncols_dst) {
                continue;
            }
            d[col*d_stride + row - d_row0] = acc[nt][l];
        }
    }
#else
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// ptq1_mma2_nwr: as many row tiles per CTA as keep >= 2 CTAs per SM (the server sets no override)
int v2_nwr(const int nrows_x) {
    static int nsm = 0;
    if (nsm == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    }
    const int tiles = (nrows_x + 15) / 16;
    for (const int nwr : {8, 4, 2}) {
        if (tiles / nwr >= 2*nsm) {
            return nwr;
        }
    }
    return 1;
}

template <int ntiles>
void v2_launch_nt(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst, int dst_stride,
                  float * hi, int hi_row0, int hi_stride) {
    const int nwr = v2_nwr(nrows);
    const int s01 = K / QK_PTQ1_0;   // src0->nb[1] / type_size
    const int s11 = (int) (GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1);
#define ENG_V2(NWR) k_ptq1_v2<ntiles, NWR, 8/NWR><<<dim3((unsigned) ((nrows + 16*NWR - 1) / (16*NWR)), 1, 1), \
        dim3(WARP_SIZE, 8, 1), 0, st>>>(w, q8, dst, K, nrows, ncols, s01, s11, dst_stride, hi, hi_row0, hi_stride)
    switch (nwr) {
        case 8:  ENG_V2(8); break;
        case 4:  ENG_V2(4); break;
        case 2:  ENG_V2(2); break;
        default: ENG_V2(1); break;
    }
#undef ENG_V2
}

// the v2 kernel on the plain q8_1 twin q8 (17..32 columns)
void ptq1_v2_launch(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst,
                    int dst_stride, float * hi, int hi_row0, int hi_stride) {
    switch ((ncols + 7) / 8) {
        case 3:  v2_launch_nt<3>(st, w, q8, K, nrows, ncols, dst, dst_stride, hi, hi_row0, hi_stride); break;
        case 4:  v2_launch_nt<4>(st, w, q8, K, nrows, ncols, dst, dst_stride, hi, hi_row0, hi_stride); break;
        default:
            fprintf(stderr, "ptq1 v2: %d columns (the engine instantiates 17..32)\n", ncols);
            abort();
    }
}

} // namespace

// The engine's PTQ1 matmul, routed as the fork routes it: its one-to-four-tile kernel unless ptq1_mma_version picks v2 --
// here the selected one-tile variant per shape and the multi-tile kernel to 32
// columns; the v2 kernel where the server runs that. Above 32 columns the server takes MMQ (eng::mmq_ptq1).
void ptq1_matmul(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst, int dst_stride,
                 float * hi, int hi_row0, int hi_stride, float * hi2, int hi2_row0, int hi2_stride) {
    if (ncols < 1 || ncols > 32) {
        fprintf(stderr, "ptq1_matmul: %d columns (1..32; wider batches take MMQ)\n", ncols);
        abort();
    }
    if (ptq1_v2(ncols, nrows)) {   // q8: the plain q8_1 layout, as ptq1_mma's v2 reads it; no second sibling there
        if (hi2 != nullptr) {
            fprintf(stderr, "ptq1_matmul: the v2 kernel carries one sibling output, not two\n");
            abort();
        }
        ptq1_v2_launch(st, w, q8, K, nrows, ncols, dst, dst_stride, hi, hi_row0, hi_stride);
        return;
    }
    const int variant = (nrows == 5120 && K == 17408) ? 1 : (nrows == 14336 || nrows > 65536 ? 1 : 2);
    ptq1_own(st, w, q8, K, nrows, ncols, dst, dst_stride, variant, hi, hi_row0, hi_stride, hi2, hi2_row0, hi2_stride);
}

// a prompt chunk of 17..32 rows with no q8 twin: the activation quantized in the layout the kernel reads, then the matmul
void ptq1_mm_f32(cudaStream_t st, const void * w, const float * x, int K, int nrows, int ncols, float * dst, int dst_stride,
                 char * q8) {
    if (ptq1_v2(ncols, nrows)) {   // the v2 kernel reads the plain q8_1 layout (quantize_q8_1<false>)
        q8_1_quantize(st, x, K, ncols, q8);
        ptq1_v2_launch(st, w, q8, K, nrows, ncols, dst, dst_stride, nullptr, 0, 0);
        return;
    }
    const int64_t Kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    k_q8_perm<<<dim3((unsigned) ((Kp + 255)/256), (unsigned) ncols, 1), 256, 0, st>>>(x, q8, K, K, Kp);
    ptq1_matmul(st, w, q8, K, nrows, ncols, dst, dst_stride);
}

} // namespace eng
