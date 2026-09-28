#pragma once

#include "common.cuh"
#include "xyzkv-quant.cuh"
#include "fattn-swizzle.cuh"

// Native xyzkv2_0 and q4_0 KV tile loaders for the MMA flash-attention kernel.
// Reading quantized bytes directly into shared memory avoids materializing the whole cache as f16.
//
// DKQ = 256 uses a bank-aligned, swizzled tile. Writers and ldmatrix readers must use the same address map.

// Four consecutive half2 = one 16-byte chunk (col_h2 % 4 == 0). The swizzle XORs only bits 4-6 of the byte offset
// (fattn-swizzle.cuh bytes_rc), i.e. it permutes 16-byte chunks and never splits one, so the chunk stays contiguous and
// 16-byte aligned; unswizzled rows are 16-byte aligned when the stride is a multiple of 4 half2.
template <int stride_tile, bool swz>
static __device__ __forceinline__ void flash_attn_ext_xyzkv_tile_put4(
        half2 * __restrict__ tile, const int row, const int col_h2, const uint4 val) {
    if constexpr (swz) {
        const int off = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(row, col_h2);
        *(uint4 *) ((char *) tile + off) = val;
    } else {
        static_assert(stride_tile % 4 == 0, "16-byte tile stores need a stride that is a multiple of 4 half2");
        *(uint4 *) (tile + row * stride_tile + col_h2) = val;
    }
}

// Write one half2 into the tile, honouring the swizzle when it is active.
template <int stride_tile, bool swz>
static __device__ __forceinline__ void flash_attn_ext_xyzkv_tile_put(
        half2 * __restrict__ tile, const int row, const int col_h2, const half2 val) {
    if constexpr (swz) {
        // bytes_rc static_asserts bank alignment, so it is only instantiated on the swizzled branch.
        const int off = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(row, col_h2);
        *(half2 *) ((char *) tile + off) = val;
    } else {
        tile[row * stride_tile + col_h2] = val;
    }
}

// Native cache types are xyzkv2_0 for the target and q4_0 for the drafter.
//
// A block is 18 bytes -- f16 d, then 16 bytes of nibbles, byte j holding value j (low nibble) and value j + 16 (high) --
// so with 4-byte-aligned rows every other block starts two bytes past a word. An item is one whole block: its aligned
// 20-byte window read as five u32 and realigned with PRMT. That is nbatch_fa*D/32 items of four 16-byte tile stores,
// exactly the item and store count of xyzkv2's quarter-blocks, so the fused loop's decode units map one to one.
//
// The values are the reference dequant's (convert.cu dequantize_block_q4_0_f16) bit for bit: (1024 + n) - 1032 is exact
// in f16, and one f16 FMA with a +0 addend rounds d*(n - 8) once, as __floats2half2_rn rounds the f32 d*n - 8d (exact in
// f32); the +0 addend also gives n = 8 the reference's +0 where a plain multiply by a negative d would give -0.
template <ggml_type type>
static constexpr __host__ __device__ bool flash_attn_ext_native_kv() {
    return type == GGML_TYPE_XYZKV2_0 || type == GGML_TYPE_Q4_0;
}

// d (low 16 bits) and the 16 qs bytes, as four little-endian words, of the q4_0 block at blk (2-byte aligned, inside a
// 4-byte-aligned row with an even block count, so the window never leaves the row).
static __device__ __forceinline__ void flash_attn_ext_q4_0_load_block(const char * blk, uint32_t & d, uint32_t (&qs)[4]) {
    const uint32_t   mis = (uint32_t) (uintptr_t) blk & 2u;   // 0: the block starts on a word; 2: two bytes past one
    const uint32_t * w   = (const uint32_t *) (blk - mis);
    uint32_t W[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        W[i] = w[i];
    }
    d = __byte_perm(W[0], 0u, mis ? 0x4432u : 0x4410u);
    const uint32_t sel = mis ? 0x7654u : 0x5432u;             // the 4 bytes that start mis + 2 bytes into W[k]
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        qs[k] = __byte_perm(W[k], W[k + 1], sel);
    }
}

// Values 8c .. 8c+7 of a loaded block as four half2 (one 16-byte tile chunk): the low (c < 2) or high (c >= 2) nibbles
// of qs bytes 8*(c % 2) .. 8*(c % 2) + 7.
static __device__ __forceinline__ uint4 flash_attn_ext_q4_0_chunk(const uint32_t d, const uint32_t (&qs)[4], const int c) {
    const uint32_t dd = __byte_perm(d, 0u, 0x1010u);          // {d, d}
    const uint32_t a  = qs[2*(c % 2) + 0] >> (c >= 2 ? 4 : 0);
    const uint32_t b  = qs[2*(c % 2) + 1] >> (c >= 2 ? 4 : 0);
    uint32_t v[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t t = __byte_perm(p < 2 ? a : b, 0u, p % 2 ? 0x4342u : 0x4140u);   // two bytes -> the two halves
        const uint32_t n = (t & 0x000F000Fu) | 0x64006400u;                              // {1024 + n0, 1024 + n1}
        half2 x, s, m;
        const uint32_t k1032 = 0x64086408u;
        memcpy(&x, &n, 4);
        memcpy(&s, &k1032, 4);
        memcpy(&m, &dd, 4);
        const half2 r = __hfma2(__hsub2(x, s), m, make_half2(0.0f, 0.0f));
        memcpy(&v[p], &r, 4);
    }
    return make_uint4(v[0], v[1], v[2], v[3]);
}

// flash_attn_ext_xyzkv2_load_tile for q4_0: same interface, same batched / plain / oob structure.
template <int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check, bool swz, bool batch_ok>
static __device__ __forceinline__ void flash_attn_ext_q4_0_load_tile(
        const char * __restrict__ raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    static_assert(D % 64 == 0, "an even block count per row keeps every 20-byte window inside its row");
    constexpr int items_per_row = D / QK4_0;
    constexpr int n_items       = nbatch_fa * items_per_row;
    const int tid = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;

    if constexpr (batch_ok && !oob_check && n_items % nthreads == 0) {
        constexpr int per_thread = n_items / nthreads;
        uint32_t d[per_thread];
        uint32_t qs[per_thread][4];
#pragma unroll
        for (int it = 0; it < per_thread; ++it) {
            const int item = tid + it*nthreads;
            flash_attn_ext_q4_0_load_block(raw + (int64_t) (item / items_per_row) * stride_bytes
                + (item % items_per_row) * (int) sizeof(block_q4_0), d[it], qs[it]);
        }
#pragma unroll
        for (int it = 0; it < per_thread; ++it) {
            const int item = tid + it*nthreads;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, item / items_per_row,
                    (item % items_per_row)*(QK4_0/2) + 4*c, flash_attn_ext_q4_0_chunk(d[it], qs[it], c));
            }
        }
        return;
    }
    for (int item = tid; item < n_items; item += nthreads) {
        const int row = item / items_per_row;
        const int h2  = (item % items_per_row) * (QK4_0/2);
        if (oob_check && row >= i_sup) {   // zero-fill past the end of the cache, as the f16 path pads
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, row, h2 + 4*c, make_uint4(0u, 0u, 0u, 0u));
            }
            continue;
        }
        uint32_t d;
        uint32_t qs[4];
        flash_attn_ext_q4_0_load_block(raw + (int64_t) row * stride_bytes + (item % items_per_row) * (int) sizeof(block_q4_0), d, qs);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, row, h2 + 4*c, flash_attn_ext_q4_0_chunk(d, qs, c));
        }
    }
}

// Load nbatch_fa rows of xyzkv2_0-quantized K or V directly into a half2 shmem tile (q4_0: the loader above).
//
// One block holds QK_XYZKV2 (128) values as a 2-byte norm plus 32 bytes of 2-bit indices, so a row
// of width D holds D/128 blocks. Threads stride over rows; each thread dequantises a whole row so
// the four centroids are computed once per block rather than once per element.
//
// oob_check zero-fills rows past the end of the cache, matching the f16 path's padding semantics --
// without it the softmax would see uninitialised shmem for the ragged last tile.
template <ggml_type type, int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check, bool swz, bool batch_ok = true>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_load_tile(
        const char * __restrict__ raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    if constexpr (type == GGML_TYPE_Q4_0) {
        flash_attn_ext_q4_0_load_tile<D, stride_tile, nbatch_fa, nthreads, oob_check, swz, batch_ok>(raw, tile, stride_bytes, i_sup);
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    // Split each row into block quarters so all threads share the dequantization work.
    // Threads in one block reuse its norm while each thread reads only its quarter of qs.
    constexpr int blocks_per_row = D / QK_XYZKV2;
    constexpr int quarters       = 4;
    constexpr int h2_per_q       = QK_XYZKV2 / (2*quarters);   // 16 half2 written per item
    constexpr int items_per_row  = blocks_per_row * quarters;
    constexpr int n_items        = nbatch_fa * items_per_row;

    if constexpr (batch_ok && !oob_check && n_items % nthreads == 0) {
        // Load all items before writing the shared-memory tile to avoid alias serialization.
        constexpr int per_thread = n_items / nthreads;
        constexpr int qs_u16     = QK_XYZKV2 / (8*quarters);
        uint32_t nbits[per_thread];
        uint32_t words[per_thread][qs_u16];
#pragma unroll
        for (int it = 0; it < per_thread; ++it) {
            const int item = tid + it*nthreads;
            const int rem  = item % items_per_row;
            const volatile uint16_t * blk16 = (const volatile uint16_t *)
                (raw + (int64_t) (item / items_per_row) * stride_bytes + (rem / quarters) * (int) sizeof(block_xyzkv2_0));
            nbits[it] = blk16[0];
#pragma unroll
            for (int u = 0; u < qs_u16; ++u) {
                words[it][u] = blk16[1 + (rem % quarters)*qs_u16 + u];
            }
        }
#pragma unroll
        for (int it = 0; it < per_thread; ++it) {
            const int item = tid + it*nthreads;
            const int row  = item / items_per_row;
            const int rem  = item % items_per_row;
            ggml_half norm_h;
            const uint16_t nb = (uint16_t) nbits[it];
            memcpy(&norm_h, &nb, sizeof(norm_h));
            const float norm = __half2float(norm_h);
            const uint32_t lut_lo = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[0] * norm)) |
                                   ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[1] * norm)) << 16);
            const uint32_t lut_hi = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[2] * norm)) |
                                   ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[3] * norm)) << 16);
#pragma unroll
            for (int u = 0; u < qs_u16; ++u) {
                const uint32_t w = words[it][u];
                uint32_t v4[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
                    v4[p] = __byte_perm(lut_lo, lut_hi, t*0x22u + 0x1010u);
                }
                flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, row,
                    (rem / quarters)*(QK_XYZKV2/2) + (rem % quarters)*h2_per_q + 4*u, make_uint4(v4[0], v4[1], v4[2], v4[3]));
            }
        }
        return;
    }
    for (int item = tid; item < n_items; item += nthreads) {
        const int row     = item / items_per_row;
        const int rem     = item - row*items_per_row;
        const int blk_idx = rem / quarters;
        const int q       = rem - blk_idx*quarters;

        // Base half2 index in the tile for this quarter.
        const int h2_base = blk_idx*(QK_XYZKV2/2) + q*h2_per_q;

        if (oob_check && row >= i_sup) {
            // Each item zero-fills only its own quarter; every quarter is visited, so the row is
            // still fully padded, matching the f16 path's semantics for the ragged last tile.
#pragma unroll
            for (int b = 0; b < h2_per_q; ++b) {
                flash_attn_ext_xyzkv_tile_put<stride_tile, swz>(tile, row, h2_base + b, make_half2(0.0f, 0.0f));
            }
            continue;
        }

        const char * row_ptr = raw + (int64_t) row * stride_bytes;

        {
            // A xyzkv2 block is 34 bytes, so only 2-byte loads are aligned for every block.
            // Volatile keeps the compiler from merging them into a wider access.
            const volatile uint16_t * blk16 =
                (const volatile uint16_t *) (row_ptr + blk_idx * (int) sizeof(block_xyzkv2_0));

            ggml_half norm_h;
            const uint16_t norm_bits = blk16[0];
            memcpy(&norm_h, &norm_bits, sizeof(norm_h));
            const float norm = __half2float(norm_h);

            // Pack the four f16 centroids into two words for byte-permute lookup.
            // Only THIS quarter's bytes. A quarter covers values [q*32, q*32+32), which is qs bytes [q*8, q*8+8) --
            // four uint16_t, not sixteen; the other three quarters are other threads' work. Word u holds values
            // 8u..8u+7 of the quarter, value j at bits 2j..2j+1, so half2 b = 4u + p takes the codes at bits 4p and
            // 4p + 2.
            constexpr int qs_u16_per_q = QK_XYZKV2 / (8*quarters);
            static_assert(h2_per_q == 4*qs_u16_per_q, "one uint16_t of codes = four half2 of the tile");
            const uint32_t lut_lo = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[0] * norm)) |
                                   ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[1] * norm)) << 16);
            const uint32_t lut_hi = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[2] * norm)) |
                                   ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[3] * norm)) << 16);
#pragma unroll
            for (int u = 0; u < qs_u16_per_q; ++u) {
                const uint32_t w = blk16[1 + q*qs_u16_per_q + u];
                // the word's four half2 are one aligned 16-byte chunk of the tile (h2_base is a multiple of 16)
                uint32_t v4[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
                    v4[p] = __byte_perm(lut_lo, lut_hi, t*0x22u + 0x1010u);
                }
                flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, row, h2_base + 4*u, make_uint4(v4[0], v4[1], v4[2], v4[3]));
            }
        }
    }
}

// The loader above reads its rows through a pointer + pitch, so it does not care whether the rows live in global
// memory or in a packed shared-memory stage. Staging overlaps the next tile's copies with the current tile's MMA work.
//
// The helpers are self-contained on purpose: cp-async.cuh has no include guard, and this header must not depend on
// being included after it.

static __device__ __forceinline__ void flash_attn_ext_xyzkv2_cp_async_4(char * dst_smem, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int dst = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst_smem);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Each thread waits for ITS copies; a __syncthreads() after this makes every thread's rows visible to all.
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Bytes of one packed row of D values: xyzkv2 = D/128 blocks of 34 bytes (68 for D = 256), q4_0 = D/32 blocks of 18
// (144); both a whole number of words. (Non-native types never reach a caller that uses the value.)
template <ggml_type type, int D>
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_row_bytes() {
    return type == GGML_TYPE_Q4_0 ? (D / QK4_0) * (int) sizeof(block_q4_0) : (D / QK_XYZKV2) * (int) sizeof(block_xyzkv2_0);
}

// Which instantiations stage: both caches the same native type (xyzkv2 or q4_0), GQA-packed (ncols2 > 1, the
// non-ragged branch), head size 256 -- the shape ggml_cuda_fattn_mma_use_native_xyzkv2() admits. The kernel
// additionally needs CP_ASYNC_AVAILABLE; the host allocates the stage for every instantiation this returns true for,
// so a device without cp.async just wastes the few KB.
template <ggml_type type_K, ggml_type type_V, int ncols2, int DKQ, int DV>
static constexpr __host__ __device__ bool flash_attn_ext_xyzkv2_stage_enabled() {
    return type_K == type_V && flash_attn_ext_native_kv<type_K>() && ncols2 > 1 && DKQ == 256 && DV == 256;
}

// Byte offset of the stage inside the dynamic shared memory: right after the KV+mask tiles when Q lives in registers --
// the Q tile is dead once process_tile has moved Q into registers, behind the barrier before the first stage issue, so
// the stage may reuse its upper part -- after
// Q + KV + mask otherwise. Host and kernel both call this.
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_stage_off(
        const bool q_in_reg, const int ncols, const int dkq, const int nbatch_fa, const int stride_tile_kv_max, const int ncols1) {
    const int q_h2    = ncols * (dkq/2 + 4);
    const int kvm_h2  = nbatch_fa * stride_tile_kv_max + ncols1 * (nbatch_fa/2 + 4);
    const int used_h2 = q_in_reg ? kvm_h2 : q_h2 + kvm_h2;
    return ((used_h2 * (int) sizeof(half2)) + 15) & ~15;
}

// Two staged KV slots keep the next tile in flight while the current tile computes.
static constexpr int FATTN_XYZKV2_RING = 2;

// The verify tiles (>= 32 columns: n >= 3 with pair packing) run the ring. The 8/16-column tiles keep S = 1: a 64-row V
// tile at occupancy 2-4 leaves no shared memory for more slots without losing a CTA per SM.
template <ggml_type type_K, ggml_type type_V, int ncols2, int DKQ, int DV, int ncols>
static constexpr __host__ __device__ bool flash_attn_ext_xyzkv2_ring_enabled() {
    return ncols >= 32 && flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>();
}

template <ggml_type type_K, ggml_type type_V, int ncols2, int DKQ, int DV, int ncols>
static constexpr __host__ __device__ bool flash_attn_ext_xyzkv2_fuse_enabled() {
    return ncols <= 32 && flash_attn_ext_xyzkv2_ring_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>();
}

// f16 KV buffers the staged xyzkv2 path uses: 2 with the overlap or the fused loop (tile_V separate from tile_K), else 1
// (aliased). ncols matters: the fused loop's second buffer is only affordable on the ring tiles.
template <ggml_type type_K, ggml_type type_V, int ncols2, int DKQ, int DV, int ncols>
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_kv_bufs() {
    return flash_attn_ext_xyzkv2_fuse_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>() ? 2 : 1;
}

// Decode one unit of a packed stage. The caller can distribute units across MMA steps.
template <int D, int nbatch_fa, int nthreads>
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_decode_units() {
    return (nbatch_fa * (D / QK_XYZKV2) * 4 / nthreads) * 4;
}

// Load each thread's raw words before the fused MMA loop to avoid shared-memory alias serialization.
template <int D, int nbatch_fa, int nthreads>
struct flash_attn_ext_xyzkv2_decode_regs {
    static constexpr int items = flash_attn_ext_xyzkv2_decode_units<D, nbatch_fa, nthreads>() >= 4 ?
        flash_attn_ext_xyzkv2_decode_units<D, nbatch_fa, nthreads>() / 4 : 1;   // >= 1: also instantiated for non-xyzkv heads
    uint32_t nbits[items];
    uint32_t words[items][4];
};

// q4_0: nbits = d, words = the block's qs realigned (flash_attn_ext_q4_0_load_block) -- the same 5 registers per item.
// Pitch is the stage row pitch. Stage points at row zero's first byte.
template <ggml_type type, int D, int nbatch_fa, int nthreads, int pitch = flash_attn_ext_xyzkv2_row_bytes<type, D>()>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_decode_load(
        const char * __restrict__ stage, flash_attn_ext_xyzkv2_decode_regs<D, nbatch_fa, nthreads> & r) {
    const int tid = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;
    if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int items_per_row = D / QK4_0;
#pragma unroll
        for (int it = 0; it < r.items; ++it) {
            const int item = tid + it * nthreads;
            flash_attn_ext_q4_0_load_block(stage + (item / items_per_row) * pitch
                + (item % items_per_row) * (int) sizeof(block_q4_0), r.nbits[it], r.words[it]);
        }
        return;
    }
    constexpr int items_per_row = (D / QK_XYZKV2) * 4;
#pragma unroll
    for (int it = 0; it < r.items; ++it) {
        const int item = tid + it * nthreads;
        const volatile uint16_t * blk16 = (const volatile uint16_t *) (stage + (item / items_per_row) * pitch
            + ((item % items_per_row) / 4) * (int) sizeof(block_xyzkv2_0));
        r.nbits[it] = blk16[0];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            r.words[it][u] = blk16[1 + (item % 4)*4 + u];
        }
    }
}

// Build four byte-permute selectors from one 16-bit code word.
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_decode_word_fast(const uint32_t w, const uint32_t (&lut)[2], uint32_t (&v4)[4]) {
    const uint32_t x0 = w & 0xFFu;
    const uint32_t x1 = (w >> 8) & 0xFFu;
    const uint32_t t0 = (x0 & 0x33u) | ((x0 & 0xCCu) << 6);        // codes 0/2 at bits 0/4, 1/3 at bits 8/12
    const uint32_t t1 = (x1 & 0x33u) | ((x1 & 0xCCu) << 6);
    const uint32_t y0 = (t0 & 0x0303u) | ((t0 & 0x3030u) << 12);   // code k of the byte at byte k
    const uint32_t y1 = (t1 & 0x0303u) | ((t1 & 0x3030u) << 12);
    const uint32_t s0 = y0*0x22u + 0x10101010u;
    const uint32_t s1 = y1*0x22u + 0x10101010u;
    v4[0] = __byte_perm(lut[0], lut[1], s0);
    v4[1] = __byte_perm(lut[0], lut[1], s0 >> 16);
    v4[2] = __byte_perm(lut[0], lut[1], s1);
    v4[3] = __byte_perm(lut[0], lut[1], s1 >> 16);
}

template <ggml_type type, int D, int stride_tile, int nbatch_fa, int nthreads, bool swz, bool fast_sel = false>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_decode_unit_regs(
        const flash_attn_ext_xyzkv2_decode_regs<D, nbatch_fa, nthreads> & r, half2 * __restrict__ tile, const int u,
        uint32_t (&lut)[2]) {
    const int tid  = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;
    const int item = tid + (u / 4) * nthreads;
    if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int items_per_row = D / QK4_0;
        flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, item / items_per_row,
            (item % items_per_row)*(QK4_0/2) + 4*(u % 4), flash_attn_ext_q4_0_chunk(r.nbits[u / 4], r.words[u / 4], u % 4));
        return;
    }
    constexpr int items_per_row = (D / QK_XYZKV2) * 4;
    if (u % 4 == 0) {
        ggml_half norm_h;
        const uint16_t nb = (uint16_t) r.nbits[u / 4];
        memcpy(&norm_h, &nb, sizeof(norm_h));
        const float norm = __half2float(norm_h);
        lut[0] = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[0] * norm)) |
                ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[1] * norm)) << 16);
        lut[1] = (uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[2] * norm)) |
                ((uint32_t) __half_as_ushort(__float2half(XYZKV_CENTROIDS_2BIT[3] * norm)) << 16);
    }
    const uint32_t w = r.words[u / 4][u % 4];
    uint32_t v4[4];
    if constexpr (fast_sel) {
        flash_attn_ext_xyzkv2_decode_word_fast(w, lut, v4);
    } else {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
            v4[p] = __byte_perm(lut[0], lut[1], t*0x22u + 0x1010u);
        }
    }
    flash_attn_ext_xyzkv_tile_put4<stride_tile, swz>(tile, item / items_per_row,
        ((item % items_per_row) / 4)*(QK_XYZKV2/2) + (item % 4)*16 + 4*(u % 4), make_uint4(v4[0], v4[1], v4[2], v4[3]));
}

// Wide ring copies store each row at its original 4-byte alignment inside a padded 16-byte window.
// Rows whose length is already a 16-byte multiple use their unpadded length.
template <ggml_type type, int D, bool wide>
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_stage_pitch() {
    return !wide || flash_attn_ext_xyzkv2_row_bytes<type, D>() % 16 == 0 ? flash_attn_ext_xyzkv2_row_bytes<type, D>() :
        ((flash_attn_ext_xyzkv2_row_bytes<type, D>() + 12 + 15) / 16) * 16;
}

// Bytes of one ring slot: nbatch_fa raw K rows, nbatch_fa raw V rows, then the mask tile as flash_attn_ext_f16_load_mask
// lays it out (mask_rows rows of nbatch_fa halves + 16 bytes). A 16-byte multiple, so each slot's mask stays aligned for
// the mask's 16-byte copies. Wide rows use their padded pitch.
template <ggml_type type, int DKQ, int DV, bool wide = false>
static constexpr __host__ __device__ int flash_attn_ext_xyzkv2_slot_bytes(const int nbatch_fa, const int mask_rows) {
    return (nbatch_fa * (flash_attn_ext_xyzkv2_stage_pitch<type, DKQ, wide>() + flash_attn_ext_xyzkv2_stage_pitch<type, DV, wide>())
            + mask_rows * (nbatch_fa * (int) sizeof(half) + 16) + 15) & ~15;
}

static __device__ __forceinline__ void flash_attn_ext_xyzkv2_cp_async_16(char * dst_smem, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int dst = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst_smem);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Copy one tile's rows into the padded stage. raw_aligned is the tile's first row minus its
// misalignment o (16-byte aligned); stride_bytes a multiple of 16. Chunk c = row * (pitch/16) + col lands at stage + 16c.
template <ggml_type type, int D, int nbatch_fa, int nthreads>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_stage_issue16(
        const char * __restrict__ raw_aligned, char * __restrict__ stage, const int stride_bytes) {
    constexpr int chunks = flash_attn_ext_xyzkv2_stage_pitch<type, D, true>() / 16;   // per row
    constexpr int n      = nbatch_fa * chunks;
    const int tid = threadIdx.y * ggml_cuda_get_physical_warp_size() + threadIdx.x;
#pragma unroll
    for (int c0 = 0; c0 < n; c0 += nthreads) {
        const int c = c0 + tid;
        if (n % nthreads == 0 || c < n) {
            const int row = c / chunks;
            flash_attn_ext_xyzkv2_cp_async_16(stage + 16*c, raw_aligned + (int64_t) row * stride_bytes + 16*(c - row*chunks));
        }
    }
}

static __device__ __forceinline__ void flash_attn_ext_xyzkv2_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Each thread waits until at most n of ITS newest copy groups are pending; a __syncthreads() after it publishes the rows.
template <int n>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Issue the cp.async copies for one tile of raw xyzkv2 rows into `stage` (packed, pitch = row bytes), row by row in
// 4-byte words. Completion is the CALLER's job: wait_all, then __syncthreads(), before anything reads the stage.
//
template <ggml_type type, int D, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_stage_issue(
        const char * __restrict__ raw,
        char       * __restrict__ stage,
        const int stride_bytes,
        const int i_sup) {
    constexpr int row_bytes     = flash_attn_ext_xyzkv2_row_bytes<type, D>();
    static_assert(row_bytes % 4 == 0, "a native row must be a whole number of 4-byte words");
    constexpr int words_per_row = row_bytes / 4;
    constexpr int n_words       = nbatch_fa * words_per_row;
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int w = tid; w < n_words; w += nthreads) {
        const int row = w / words_per_row;
        const int col = w - row * words_per_row;
        if (oob_check && row >= i_sup) {
            continue;   // the dequant loader zero-fills rows past the end of the cache
        }
        flash_attn_ext_xyzkv2_cp_async_4(stage + row * row_bytes + col * 4, raw + (int64_t) row * stride_bytes + col * 4);
    }
}
