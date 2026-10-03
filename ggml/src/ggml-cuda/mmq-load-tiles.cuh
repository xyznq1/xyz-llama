#pragma once

#include "vecdotq.cuh"

// Diagnostics for the fragment-pipelined PTQ1 MMA path (mmq-vec-dot.cuh); see mmq.cuh for how they are filled.
static __device__ __constant__ bool g_ptq1_mmq_skip_mma;
static __device__ __constant__ bool g_ptq1_mmq_sync_frag;

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q1_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int blocks_per_iter = MMQ_ITER_K / QK1_0;
    constexpr int threads_per_row = blocks_per_iter * QI1_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK1_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI1_0;
    const int kqsx = txi % QI1_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *) x + kbx0 + i*stride + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 2;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int q  = qxi[j];

            // unpack crumbs into nibble indices
            const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
            const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
            // unpack nibbles into byte values
            const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
            const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
            const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
            const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
            // unshuffle values
            const int v0 = __byte_perm(s0, s1, 0x5410);
            const int v1 = __byte_perm(s0, s1, 0x7632);
            const int v2 = __byte_perm(s2, s3, 0x5410);
            const int v3 = __byte_perm(s2, s3, 0x7632);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + dst_offset + j*4+0] = v0;
            x_qs[i*sram_stride           + dst_offset + j*4+1] = v1;
            x_qs[i*sram_stride           + dst_offset + j*4+2] = v2;
            x_qs[i*sram_stride           + dst_offset + j*4+3] = v3;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+0] = v0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+1] = v1;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+2] = v2;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+3] = v3;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *) x + kbx0 + i*stride + scale_block;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int blocks_per_iter = MMQ_ITER_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iter * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 4;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qxi[j];

#if defined(GGML_USE_HIP)
            const uint32_t qx_indices = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
            const uint32_t qy_bits    = q >> 8;
            const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
            const int qx = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qx_indices);
            const int qy = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qy_indices);
#else
            // unpack even and odd crumbs into byte values
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            // unshuffle values
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);
#endif // defined(GGML_USE_HIP)

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + dst_offset + j*2+0] = qx;
            x_qs[i*sram_stride           + dst_offset + j*2+1] = qy;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+0] = qx;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+1] = qy;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + scale_block;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

#if !defined(GGML_USE_HIP)
// Eight threads share a 128-weight PTQ1_0 block and write its 32 words in dequantize_row_ptq1_0's order: the qs words
// 0-3 at stages 0-4 (elements t*16 + m), the qs words 4-5 at stages 0-4 (80 + t*8 + m), the qh pair (120 + 2t + h). The
// trits land in the tile as signed bytes, so from here on the tile is an ordinary q8_0 tile and the shared q8_0 x q8_1
// dot (dp4a or int8 tensor cores) runs unchanged -- the decode is paid ONCE PER TILE, not once per output column, which
// is the property the mmvq path lacks.
//
// One output word of the PTQ1_0 decode at any stage, from two 16-bit-lane byte pairs va, vb at stages ta, tb (ma = 3^ta,
// mb = 3^tb): byte k of the result is the trit of pair byte k at its stage, minus 1 -- a qs word's stage-t word
// (va = bytes 0-1, vb = bytes 2-3, ta = tb = t) and the qh pair's word (va = vb, tb = ta + 1).
// (v * 3^t) & 0x00FF00FF is the stage's remainder byte; * 3 puts the trit in each 16-bit lane's high byte, and
// + 0x7F007F00 biases it to trit + 0x7F without carrying out of the lane (765 + 0x7F00 < 0x10000), so after the permute
// ^ 0x80808080 is the - 1 that __vsub4 spends three instructions on.
static __device__ __forceinline__ uint32_t ggml_cuda_mmq_ptq1_0_trit_word(const uint32_t va, const uint32_t vb,
                                                                          const uint32_t ma, const uint32_t mb) {
    const uint32_t wa = ((va * ma) & 0x00FF00FF) * 3 + 0x7F007F00;
    const uint32_t wb = ((vb * mb) & 0x00FF00FF) * 3 + 0x7F007F00;
    return __byte_perm(wa, wb, 0x7531) ^ 0x80808080;
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_ptq1_0(const char * __restrict__ x,
                                                                       int * __restrict__ x_tile,
                                                                       const int kbx0,
                                                                       const int i_max,
                                                                       const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(TURING_MMA_AVAILABLE)
    int *   x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs  = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int *                  x_qs = (int *) x_tile;
    float *                x_df = (float *) (x_qs + txs.qs);
#endif // defined(TURING_MMA_AVAILABLE)

    // From the CONFIG, not the MMQ_ITER_K macro: the kernel iterates k in K_vram-sized steps, so a loader that
    // sizes itself from the macro loads the wrong amount the moment the two differ.
    constexpr int blocks_per_iter   = ggml_cuda_mmq_get_K_vram(type, J, fallback) / QK_PTQ1_0;
    constexpr int threads_per_block = 8;
    constexpr int threads_per_row   = blocks_per_iter * threads_per_block;
    constexpr int nrows             = warp_size / threads_per_row;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / threads_per_block;
    const int lane = txi % threads_per_block;

    // ILV16 (common.cuh). kbx0 = this tile's row offset (a whole number of rows, 16-aligned: I is) + the k-block kb0 <
    // stride, so the two split with one modulo. Rows below nfull sit in whole 16-row tiles of the matrix; the k-block
    // index is clamped to the row (an odd block count makes the last iteration read one past it -- against zero
    // activations, as in the row-major path, but here the address would leave the tile).
    const int kb0   = kbx0 % stride;
    const int rbase = kbx0 - kb0;
    const int nfull = (i_max + 1) & ~15;
    const auto ptq1_block = [&](const int i, const int kb) -> const block_ptq1_0 * {
        return i < nfull ? (const block_ptq1_0 *) x + rbase + (i & ~15)*stride + min(kb0 + kb, stride - 1)*16 + (i & 15)
                         : (const block_ptq1_0 *) x + kbx0 + i*stride + kb;
    };

    // Every lane's global load for every row is issued before any decode. With the loads inside the per-lane branches
    // (lane < 4 / < 6 / == 6) each row paid load -> decode -> store as its own latency chain, I/(nrows*nwarps) rows x
    // 3 paths in series per warp (SASS: 8 x (LDG STS | LDG STSx5 | LDG STSx5) per 256-value step). One aligned word per
    // lane and row instead -- lanes 0-5 the six qs words, lanes 6-7 bytes 24-27 (qh[0], qh[1], d; block_ptq1_0 is
    // 4-aligned) -- decoded into exactly the words the branches wrote: qs word w lands at row + w (stride 4) for w < 4
    // and at row + 16 + w (stride 2) for w = 4, 5. Bit-identical tiles (test-backend-ops output hashes).
    // The blocks' scales are loaded here too, with the trits, and stored after the decode: their loop used to follow
    // the decode, so its loads waited for it (the scales' half->float conversion was the kernel's second
    // long-scoreboard stall in Nsight Compute).
    constexpr int scale_entries_per_block = QK_PTQ1_0 / QK8_1;
    constexpr int scale_entries_per_row   = blocks_per_iter * scale_entries_per_block;
    constexpr int rows_per_warp           = warp_size / scale_entries_per_row;
    const int     ksx                     = threadIdx.x % scale_entries_per_row;
    const int     scale_block             = ksx / scale_entries_per_block;
    constexpr int nsc = (I + nwarps * rows_per_warp - 1) / (nwarps * rows_per_warp);
    ggml_half dsc[nsc];
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / scale_entries_per_row;
        if (fallback) {
            i = min(i, i_max);
        }
        dsc[i0 / (nwarps * rows_per_warp)] = ptq1_block(i, scale_block)->d;
    }

    constexpr int niter = I / (nrows * nwarps);
    static_assert(I % (nrows * nwarps) == 0, "whole rows per warp pass");
    uint32_t packed[niter];

#pragma unroll
    for (int it = 0; it < niter; ++it) {
        int i = it * nrows * nwarps + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (fallback) {
            i = min(i, i_max);
        }
        packed[it] = get_int_b4(ptq1_block(i, kbx)->qs, min(lane, 6));
    }

    // ONE code path for all eight lanes of a block (per-lane branches would run lanes 0-5 and lane 6 one after the
    // other while lane 7 idles): lane l writes words l, l+8, l+16 and l+24 of the block's 32. Word w is qs word w%4 at
    // stage w/4 for w < 20, qs word 4 + w%2 at stage (w-20)/2 for w < 30, the qh pair at stages 2(w-30) and 2(w-30)+1
    // for w = 30, 31. The lanes trade their loaded words with shuffles; ggml_cuda_mmq_ptq1_0_trit_word computes any
    // stage directly.
    static_assert(threads_per_block == 8, "an 8-lane group per block");
    // 3^t for t = 0..5 as the bytes of one constant: a per-lane index into an array would put it in local memory
    const auto pw = [](const int t) -> uint32_t { return (uint32_t) (0x0000F3511B090301ull >> (8 * t)) & 0xFF; };
    const int      gbase = threadIdx.x & ~(threads_per_block - 1);
    const uint32_t m0    = pw(lane >> 2);
    const uint32_t m1    = pw(2 + (lane >> 2));
    const uint32_t m2    = lane < 4 ? pw(4) : pw((lane - 4) >> 1);
    const uint32_t m3a   = lane < 6 ? pw((lane + 4) >> 1) : pw(2 * (lane - 6));
    const uint32_t m3b   = lane < 6 ? m3a                 : pw(2 * (lane - 6) + 1);

#pragma unroll
    for (int it = 0; it < niter; ++it) {
        int i = it * nrows * nwarps + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (fallback) {
            i = min(i, i_max);
        }

#if defined(TURING_MMA_AVAILABLE)
        int * row = x_qs + i * sram_stride + kbx * (QK_PTQ1_0 / 4);
#else
        int * row = x_qs + i * (2 * MMQ_TILE_NE_K + 1) + kbx * (QK_PTQ1_0 / 4);
#endif // defined(TURING_MMA_AVAILABLE)

        const uint32_t wa  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + (lane & 3), warp_size);
        const uint32_t wb  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + 4 + (lane & 1), warp_size);
        const uint32_t wq  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + 6, warp_size);
        const uint32_t alo = __byte_perm(wa, 0, 0x4140);
        const uint32_t ahi = __byte_perm(wa, 0, 0x4342);
        const uint32_t blo = __byte_perm(wb, 0, 0x4140);
        const uint32_t bhi = __byte_perm(wb, 0, 0x4342);
        const uint32_t q   = __byte_perm(wq, 0, 0x4140); // qh[0] | qh[1] << 16

        row[lane]      = ggml_cuda_mmq_ptq1_0_trit_word(alo, ahi, m0, m0);
        row[lane + 8]  = ggml_cuda_mmq_ptq1_0_trit_word(alo, ahi, m1, m1);
        row[lane + 16] = ggml_cuda_mmq_ptq1_0_trit_word(lane < 4 ? alo : blo, lane < 4 ? ahi : bhi, m2, m2);
        row[lane + 24] = ggml_cuda_mmq_ptq1_0_trit_word(lane < 6 ? blo : q, lane < 6 ? bhi : q, m3a, m3b);
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / scale_entries_per_row;
        if (fallback) {
            i = min(i, i_max);
        }

#if defined(TURING_MMA_AVAILABLE)
        x_df[i * sram_stride + ksx] = dsc[i0 / (nwarps * rows_per_warp)];
#else
        x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + ksx] = dsc[i0 / (nwarps * rows_per_warp)];
#endif // defined(TURING_MMA_AVAILABLE)
    }
}

// The PTQ1_0 pipeline (mul_mat_q_process_tile, ggml_cuda_mmq_ptq1_0_pipelined): the next K step's weight tile goes to a
// raw staging buffer in shared memory by cp.async while the current step's MMAs run, and is decoded from there. ILV16
// keeps a 16-row group's blocks of one k-block contiguous (16 x 28 = 448 B, 16-byte aligned), so a tile is
// I/16 x blocks_per_iter runs of 28 16-byte copies; the scales ride along in the blocks.
// Staging order [16-row group][k-block][row][28 B]: the decode's two k-blocks of a row are 448 B (16 banks) apart, so its
// reads are conflict-free, and a group's two k-blocks are one contiguous 896 B run in global memory. (k-block outermost
// put them I/16 x 448 B apart, a multiple of 128 B: every decode read a 2-way bank conflict.)
// (cp-async.cuh has these helpers but no include guard, and fattn-mma-f16.cuh includes it.)
static __device__ __forceinline__ void ggml_cuda_mmq_cp_async_16(void * dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int dst_s = (unsigned int) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst_s), "l"(src) : "memory");
#else
    GGML_UNUSED_VARS(dst, src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mmq_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;" : : : "memory");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// waits until at most n of this thread's most recently committed groups are still in flight
template <int n>
static __device__ __forceinline__ void ggml_cuda_mmq_cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n) : "memory");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ int ggml_cuda_mmq_ptq1_0_raw_offset(const int kb, const int i, const int blocks_per_iter) {
    return ((i >> 4) * blocks_per_iter + kb) * (16 * (int) sizeof(block_ptq1_0)) + (i & 15) * (int) sizeof(block_ptq1_0);
}

// the next K step's weight tile (kbx0 as ggml_cuda_mmq_load_tiles_ptq1_0 takes it) into the staging buffer, by cp.async
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_ptq1_0_raw_copy(const char * __restrict__ x, char * __restrict__ raw,
                                                                     const int kbx0, const int stride) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads        = ggml_cuda_mmq_get_nthreads(type, J, fallback);
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int blocks_per_iter = ggml_cuda_mmq_get_K_vram(type, J, fallback) / QK_PTQ1_0;
    constexpr int chunk           = 16 * (int) sizeof(block_ptq1_0); // 448 B: one k-block of one 16-row group
    constexpr int per_chunk       = chunk / 16;                       // 28 copies
    constexpr int ncopy           = (I / 16) * blocks_per_iter * per_chunk;
    static_assert(I % 16 == 0 && chunk % 16 == 0, "whole ILV16 groups in 16-byte copies");
    static_assert(!fallback, "full row tiles only: every row is in a whole 16-row ILV group");

    const int kb0   = kbx0 % stride;
    const int rbase = kbx0 - kb0;
    const int tid   = threadIdx.y * warp_size + threadIdx.x;

#pragma unroll
    for (int c0 = 0; c0 < ncopy; c0 += nthreads) {
        const int c = c0 + tid;
        if (ncopy % nthreads == 0 || c < ncopy) {
            const int p    = c / per_chunk; // g * blocks_per_iter + kb
            const int g    = p / blocks_per_iter;
            const int kb   = p % blocks_per_iter;
            const int part = c % per_chunk;
            const block_ptq1_0 * src = (const block_ptq1_0 *) x + rbase + g * 16 * stride + min(kb0 + kb, stride - 1) * 16;
            ggml_cuda_mmq_cp_async_16(raw + p * chunk + part * 16, (const char *) src + part * 16);
        }
    }
#else
    GGML_UNUSED_VARS(x, raw, kbx0, stride);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

// ggml_cuda_mmq_load_tiles_ptq1_0's decode, reading the staging buffer instead of global memory: the same words in the
// same slots
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_ptq1_0_raw(const char * __restrict__ raw, int * __restrict__ x_tile) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int *   x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int blocks_per_iter   = ggml_cuda_mmq_get_K_vram(type, J, fallback) / QK_PTQ1_0;
    constexpr int threads_per_block = 8;
    constexpr int threads_per_row   = blocks_per_iter * threads_per_block;
    constexpr int nrows             = warp_size / threads_per_row;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / threads_per_block;
    const int lane = txi % threads_per_block;

    constexpr int scale_entries_per_block = QK_PTQ1_0 / QK8_1;
    constexpr int scale_entries_per_row   = blocks_per_iter * scale_entries_per_block;
    constexpr int rows_per_warp           = warp_size / scale_entries_per_row;
    const int     ksx                     = threadIdx.x % scale_entries_per_row;
    const int     scale_block             = ksx / scale_entries_per_block;

    constexpr int niter = I / (nrows * nwarps);
    static_assert(I % (nrows * nwarps) == 0, "whole rows per warp pass");
    uint32_t packed[niter];
#pragma unroll
    for (int it = 0; it < niter; ++it) {
        const int i = it * nrows * nwarps + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        packed[it] = *(const uint32_t *) (raw + ggml_cuda_mmq_ptq1_0_raw_offset(kbx, i, blocks_per_iter) + 4 * min(lane, 6));
    }

    const auto pw = [](const int t) -> uint32_t { return (uint32_t) (0x0000F3511B090301ull >> (8 * t)) & 0xFF; };
    const int      gbase = threadIdx.x & ~(threads_per_block - 1);
    const uint32_t m0    = pw(lane >> 2);
    const uint32_t m1    = pw(2 + (lane >> 2));
    const uint32_t m2    = lane < 4 ? pw(4) : pw((lane - 4) >> 1);
    const uint32_t m3a   = lane < 6 ? pw((lane + 4) >> 1) : pw(2 * (lane - 6));
    const uint32_t m3b   = lane < 6 ? m3a                 : pw(2 * (lane - 6) + 1);

#pragma unroll
    for (int it = 0; it < niter; ++it) {
        const int i = it * nrows * nwarps + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        int * row = x_qs + i * sram_stride + kbx * (QK_PTQ1_0 / 4);

        const uint32_t wa  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + (lane & 3), warp_size);
        const uint32_t wb  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + 4 + (lane & 1), warp_size);
        const uint32_t wq  = __shfl_sync(0xFFFFFFFF, packed[it], gbase + 6, warp_size);
        const uint32_t alo = __byte_perm(wa, 0, 0x4140);
        const uint32_t ahi = __byte_perm(wa, 0, 0x4342);
        const uint32_t blo = __byte_perm(wb, 0, 0x4140);
        const uint32_t bhi = __byte_perm(wb, 0, 0x4342);
        const uint32_t q   = __byte_perm(wq, 0, 0x4140); // qh[0] | qh[1] << 16

        row[lane]      = ggml_cuda_mmq_ptq1_0_trit_word(alo, ahi, m0, m0);
        row[lane + 8]  = ggml_cuda_mmq_ptq1_0_trit_word(alo, ahi, m1, m1);
        row[lane + 16] = ggml_cuda_mmq_ptq1_0_trit_word(lane < 4 ? alo : blo, lane < 4 ? ahi : bhi, m2, m2);
        row[lane + 24] = ggml_cuda_mmq_ptq1_0_trit_word(lane < 6 ? blo : q, lane < 6 ? bhi : q, m3a, m3b);
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        const int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / scale_entries_per_row;
        x_df[i * sram_stride + ksx] = *(const ggml_half *) (raw + ggml_cuda_mmq_ptq1_0_raw_offset(scale_block, i, blocks_per_iter) +
                                                             offsetof(block_ptq1_0, d));
    }
#else
    GGML_UNUSED_VARS(raw, x_tile);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}
#endif // !defined(GGML_USE_HIP)

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_0;
    const int kqsx = txi % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                     + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_1(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI4_1) + kqsx + 0]     = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + kbx*(2*QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride                     + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_0;
    const int kqsx = txi % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                     + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_1(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_1;
    const int kqsx = txi % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride                     + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + kbxd] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride         + kqsx] = x_dm_ik;
#else
        x_dm[i*(MMQ_TILE_NE_K + 1) + kqsx] = x_dm_ik;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q3_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*sram_stride + sizeof(int)*ksc + l] = d*sc8[l];
        }
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = sc;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE))
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        x_df[i] = bxi->d;
    }
#endif // !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)) || defined(AMD_WMMA_AVAILABLE)
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < I) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kq0] = ql0 | qh0;
        x_qs[i*sram_stride + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = ql0 | qh0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = ql1 | qh1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < I) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
            for (int l = 0; l < int(sizeof(int)); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride]                     = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + threadIdx.x%(MMQ_TILE_NE_K/8)] = get_int_b2(bxi->scales, threadIdx.x%(QI6_K/8));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_S);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

    #pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_ds[i*sram_stride             + kqsx] = make_half2(d1q, d1q*delta);
#else
        x_ds[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = make_half2(d1q, d1q*delta);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = (ls*d + d/2)/2;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = (ls*d + d/2)/2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = ls*d;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = ls*d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = d * (ls - 32);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * (ls - 32);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0]      = v.x;
        x_qs[i*sram_stride + k0 + QI4_NL] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]      = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI4_NL] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                       + kbxd] = __half2float(bxi->d);
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_NL) + i/QI4_NL + kbxd] = __half2float(bxi->d);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP4, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR_MXFP4);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI_MXFP4;
    const int kqsx = txi % QI_MXFP4;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b1(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0]        = v.x;
        x_qs[i*sram_stride + k0 + QI_MXFP4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]        = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI_MXFP4;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#else
        x_df[i*(MMQ_TILE_NE_K/QI_MXFP4) + i/QI_MXFP4 + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4_fp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int *      x_qs = (int *) x_tile;
    uint32_t * x_sc = (uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);

    const int txi = threadIdx.x;

    constexpr int iter_k = ggml_cuda_mmq_get_K_vram(type, J, fallback);

    constexpr int threads_per_row = iter_k / QK_MXFP4;  // each thread processes 1 block
    constexpr int rows_per_warp   = warp_size / threads_per_row;
    const int     kbx             = txi % threads_per_row;
    const int     row_in_warp     = txi / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i * stride + kbx;

        // quantize_mxfp4_mmq permutes nibbles to match the quantized format
        const int k0 = kbx * 4;
        memcpy(x_qs + i*sram_stride + k0, bxi->qs, 16);

        // Load E8M0 scales: pack 2 consecutive scales into one uint32
        if (kbx % 2 == 0) {
            uint32_t e = bxi->e;
            e |= ((bxi + 1)->e << 8);
            x_sc[i*sram_stride + kbx / 2] = e;
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kb0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, I);
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;
    constexpr int rows_per_warp = warp_size / threads_per_row;
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = (const block_nvfp4 *) x + kb0 + i * stride + kbx;
        const uint32_t * __restrict__ src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
        const int kqs = 16 * kbx;
        const int ksc = 4 * kbx;

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], kvalues_mxfp4);
            const int2 q1 = get_int_from_table_16(src_qs[2 * sub + 1], kvalues_mxfp4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqs + 4 * sub + 0] = q0.x;
            x_qs[i*sram_stride + kqs + 4 * sub + 1] = q1.x;
            x_qs[i*sram_stride + kqs + 4 * sub + 2] = q0.y;
            x_qs[i*sram_stride + kqs + 4 * sub + 3] = q1.y;
            x_df[i*sram_stride + ksc + sub] = ggml_cuda_ue4m3_to_fp32(bxi->d[sub]);
#else
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 0] = q0.x;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 1] = q1.x;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 2] = q0.y;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 3] = q1.y;
            x_df[i * (2 * MMQ_TILE_NE_K * 2 / QI_NVFP4) + i / (QK_NVFP4_SUB / QI_NVFP4) + ksc + sub] = ggml_cuda_ue4m3_to_fp32(bxi->d[sub]);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int iter_k          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int threads_per_row = iter_k / QK_NVFP4; // each thread processes 1 block
    constexpr int rows_per_warp   = warp_size / threads_per_row;
    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    uint32_t * x_u32 = (uint32_t *) x_tile;

    const int txi = threadIdx.x;
    const int kbx = txi % threads_per_row;
    const int row_in_warp = txi / threads_per_row;

    const block_nvfp4 * bxi_base = (const block_nvfp4 *) x + kbx0 + kbx;
    uint32_t * x_u32_scale = x_u32 + 64 + kbx;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = bxi_base + i * stride;

        const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            x_u32[i*sram_stride + 8*kbx + 2 * sub + 0] = src_qs[2 * sub + 0];
            x_u32[i*sram_stride + 8*kbx + 2 * sub + 1] = src_qs[2 * sub + 1];
        }

        x_u32_scale[i*sram_stride] = get_int_b4(bxi->d, 0);
    }
}
