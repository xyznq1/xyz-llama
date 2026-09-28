#pragma once
// The engine's attention: MMA flash attention for head size 256 over the target's native xyzkv2 cache and the drafter's
// native q4_0 cache -- KQ and PV on the tensor cores (f16 in, f32 KQ, f16 VKQ accumulators), the online softmax, and
// stream-K over (KV tiles x output tiles) with the partial results recombined by a fixup launch. A tile is ncols1 x 8
// columns: 8 query-head slots of a KV head's GQA group per token, or on the 32/64-column tiles the (token, head) PAIRS of
// the real 6-head group (5 / 10 tokens per tile). The KV bytes are staged raw in shared memory (cp.async) and decoded to
// f16 tiles in the kernel; on the 32-column verify tile the staging is a 2-slot ring refilled in 16-byte copies and the
// decode is threaded between the MMAs (V(t) under KQ(t), K(t+1) under PV(t)).
//
// Derived from the fork's fattn-mma-f16.cuh + fattn-xyzkv-tiles.cuh + fattn-swizzle.cuh native xyzkv2/q4_0 path at the
// exact instances the server runs on sm_89, statement for statement: every output is
// bit-identical. Other head sizes, cache types, and experimental instances are not included.
#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"

#include "../xyzkv2.cuh"

namespace eng {
namespace fa {

using namespace ggml_cuda_mma;

constexpr int   D           = 256;              // head size (K and V)
constexpr int   NCOLS2      = 8;                // query-head slots per KV head in a tile
constexpr int   RING        = 2;                // ring slots of the 32/64-column tiles
constexpr int   REF_TOKENS  = 32;               // above: tiles pack as the f16 kernel's (no pair packing)
constexpr int   BATCH_MAX_NCOLS = 64;           // the decode's batched loads below this many columns
// The VKQ rescale skip runs on every tile except the 32-column verify tile, where the warp vote and branch disrupt
// softmax interleaving.
constexpr int   SKIP_EXCLUDE_NCOLS = 32;
constexpr float SOFTMAX_FTZ = -20.0f;           // exps of values below this flush to zero
constexpr float KQ_MAX_OFFSET = 3.0f*0.6931f;

enum : int { F_WIDE = 16, F_FAST_SEL = 32, F_POS_MASK = 512 };   // instance features (the fork's l2pf bits)

// ---- configuration (head size 256, sm_89) --------------------------------------------------------------------------
struct Config {
    int  nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine;
    bool Q_in_reg;
};

__host__ __device__ constexpr Config config(const int ncols) {
    return ncols ==  8 ? Config{128, 2, 64, 128, 128, 128, true} :
           ncols == 16 ? Config{ 64, 4, 32, 128, 128, 128, true} :
           ncols == 32 ? Config{128, 2, 32, 128, 128, 128, true} :
           ncols == 64 ? Config{128, 2, 32, 128, 128, 128, true} :
                         Config{ 32, 1,  0,   0,   0,   0, false};
}

template <int ncols> struct Tiles {
    using T_A_KQ  = tile<16,  8, half2>;   // row-major
    using T_B_KQ  = tile<16,  8, half2>;   // column-major
    using T_C_KQ  = tile<16, 16, float>;   // column-major
    using T_A_VKQ = tile<16,  8, half2>;   // row-major
    using T_B_VKQ = tile<16,  8, half2>;   // column-major
    using T_C_VKQ = tile<16,  8, half2>;   // column-major
};
template <> struct Tiles<8> {
    using T_A_KQ  = tile<16,  8, half2>;   // row-major
    using T_B_KQ  = tile< 8,  8, half2>;   // column-major
    using T_C_KQ  = tile<16,  8, float>;   // row-major
    using T_A_VKQ = tile<16,  8, half2>;   // row-major
    using T_B_VKQ = tile< 8,  8, half2>;   // column-major
    using T_C_VKQ = tile<16,  4, half2>;   // row-major
};

// ---- pair packing ---------------------------------------------------------------------------------------------------
__host__ __device__ constexpr bool pair_capable(const int ncols1) {
    return ncols1*NCOLS2 >= 32;
}
// the column group size d of a tile: the real query heads per KV head on a pair-capable tile (6 < 8), else the 8 slots
__host__ __device__ __forceinline__ int pair_d(const int ncols1, const int gqa_ratio, const int n_tokens) {
    return pair_capable(ncols1) && n_tokens <= REF_TOKENS && gqa_ratio > 1 && gqa_ratio < NCOLS2 ? gqa_ratio : NCOLS2;
}
// rows of the mask tile = tokens per tile
__host__ __device__ constexpr int mask_rows(const int ncols1) {
    return pair_capable(ncols1) ? (ncols1*NCOLS2)/2 : ncols1;
}

// ---- the shared-memory swizzle of the K/V tiles (XOR on 16-byte chunks; strides that are multiples of 32 half2) ----
namespace swz {

__host__ __device__ constexpr bool enabled(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

__host__ __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

template <int stride_h2>
__device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(enabled(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(int * xi, const half2 * addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(int * xi, const half2 * addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
}

template <int stride_h2>
__device__ __forceinline__ const half2 * lane_addr(const half2 * tile_base, const int base_row, const int base_col_h2,
                                                   const int I, const int J) {
    static_assert(enabled(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    uint32_t byte_off = (uint32_t) ((base_row + lane_row)*stride_h2 + base_col_h2 + lane_col) * (uint32_t) sizeof(half2);
    byte_off ^= (uint32_t) (((base_row + lane_row) & 7) << 4);
    return (const half2 *) ((const char *) tile_base + byte_off);
}

template <int stride_h2, bool swz, typename TileT>
__device__ __forceinline__ void load_ldmatrix(TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, tile<16, 8, half2>>, "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template <int stride_h2, bool swz, typename TileT>
__device__ __forceinline__ void load_ldmatrix_trans(TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, tile<16, 8, half2>>, "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4_trans((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template <int stride_h2, bool swz, typename TileT>
__device__ __forceinline__ void load_ldmatrix_trans(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix_trans<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + off_h2, stride_h2);
    }
}

} // namespace swz

// ---- the native caches: staging and decode ----------------------------------------------------------------------------

// bytes of one packed row of D values: xyzkv2 = D/128 blocks of 34 (68), q4_0 = D/32 blocks of 18 (144)
template <ggml_type type>
__host__ __device__ constexpr int row_bytes() {
    return type == GGML_TYPE_Q4_0 ? (D / QK4_0) * (int) sizeof(block_q4_0) : (D / QK_XYZKV2) * (int) sizeof(block_xyzkv2_0);
}

template <int ncols>
__host__ __device__ constexpr bool ring_enabled() {
    return RING > 1 && ncols >= 32;
}

template <int ncols>
__host__ __device__ constexpr bool fuse_enabled() {
    return ncols <= 32 && ring_enabled<ncols>();
}

template <int ncols>
__host__ __device__ constexpr int kv_bufs() {
    return fuse_enabled<ncols>() ? 2 : 1;
}

// the stage's byte offset in the dynamic shared memory: after the KV + mask tiles (Q lives in registers, so the stage may
// reuse the Q tile's upper part once Q is in registers)
__host__ __device__ constexpr int stage_off(const int nbatch_fa, const int stride_tile_kv_max, const int nrows_mask) {
    const int kvm_h2 = nbatch_fa * stride_tile_kv_max + nrows_mask * (nbatch_fa/2 + 4);
    return ((kvm_h2 * (int) sizeof(half2)) + 15) & ~15;
}

// the ring's row pitch: wide (16-byte copies) pads a xyzkv2 row to 80 bytes; q4_0 rows are already a 16-byte multiple
template <ggml_type type, bool wide>
__host__ __device__ constexpr int stage_pitch() {
    return !wide || row_bytes<type>() % 16 == 0 ? row_bytes<type>() : ((row_bytes<type>() + 12 + 15) / 16) * 16;
}

// one ring slot: nbatch_fa raw K rows, nbatch_fa raw V rows, the mask tile (nrows_mask rows of nbatch_fa halves + 16 bytes)
template <ggml_type type, bool wide = false>
__host__ __device__ constexpr int slot_bytes(const int nbatch_fa, const int nrows_mask) {
    return (nbatch_fa * (stage_pitch<type, wide>() + stage_pitch<type, wide>()) + nrows_mask * (nbatch_fa * (int) sizeof(half) + 16)
            + 15) & ~15;
}

template <int nbatch_fa, int nthreads>
__host__ __device__ constexpr int decode_units() {
    return (nbatch_fa * (D / QK_XYZKV2) * 4 / nthreads) * 4;
}

template <int nbatch_fa, int nthreads>
struct DecodeRegs {
    static constexpr int items = decode_units<nbatch_fa, nthreads>() >= 4 ? decode_units<nbatch_fa, nthreads>() / 4 : 1;
    uint32_t nbits[items];
    uint32_t words[items][4];
};

// cp.async is sm_80+: below that the kernels compile to NO_DEVICE_CODE (xe_create refuses those GPUs), as cp-async.cuh does
__device__ __forceinline__ void cp_async_4(char * dst_smem, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int dst = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst_smem);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

__device__ __forceinline__ void cp_async_16(char * dst_smem, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int dst = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst_smem);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

__device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

__device__ __forceinline__ void cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// four consecutive half2 (one 16-byte chunk) into the tile, through the swizzle when it is on
template <int stride_tile, bool swz>
__device__ __forceinline__ void tile_put4(half2 * __restrict__ tile, const int row, const int col_h2, const uint4 val) {
    if constexpr (swz) {
        const int off = swz::bytes_rc<stride_tile>(row, col_h2);
        *(uint4 *) ((char *) tile + off) = val;
    } else {
        static_assert(stride_tile % 4 == 0, "16-byte tile stores need a stride that is a multiple of 4 half2");
        *(uint4 *) (tile + row * stride_tile + col_h2) = val;
    }
}

// q4_0: d (low 16 bits) and the 16 qs bytes as four words, of the block at blk (2-byte aligned, 4-byte-aligned rows)
__device__ __forceinline__ void q4_0_load_block(const char * blk, uint32_t & d, uint32_t (&qs)[4]) {
    const uint32_t   mis = (uint32_t) (uintptr_t) blk & 2u;
    const uint32_t * w   = (const uint32_t *) (blk - mis);
    uint32_t W[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        W[i] = w[i];
    }
    d = __byte_perm(W[0], 0u, mis ? 0x4432u : 0x4410u);
    const uint32_t sel = mis ? 0x7654u : 0x5432u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        qs[k] = __byte_perm(W[k], W[k + 1], sel);
    }
}

// values 8c .. 8c+7 of a loaded q4_0 block as four half2: d*(n - 8) in one f16 FMA with a +0 addend
__device__ __forceinline__ uint4 q4_0_chunk(const uint32_t d, const uint32_t (&qs)[4], const int c) {
    const uint32_t dd = __byte_perm(d, 0u, 0x1010u);
    const uint32_t a  = qs[2*(c % 2) + 0] >> (c >= 2 ? 4 : 0);
    const uint32_t b  = qs[2*(c % 2) + 1] >> (c >= 2 ? 4 : 0);
    uint32_t v[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t t = __byte_perm(p < 2 ? a : b, 0u, p % 2 ? 0x4342u : 0x4140u);
        const uint32_t n = (t & 0x000F000Fu) | 0x64006400u;
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

// the two words of a xyzkv2 block's centroid LUT: {c0, c1} and {c2, c3} times the block norm, each rounded to f16 once
__device__ __forceinline__ void xyzkv2_lut(const uint32_t nbits, uint32_t (&lut)[2]) {
    ggml_half norm_h;
    const uint16_t nb = (uint16_t) nbits;
    memcpy(&norm_h, &nb, sizeof(norm_h));
    const float norm = __half2float(norm_h);
    lut[0] = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[0] * norm)) |
            ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[1] * norm)) << 16);
    lut[1] = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[2] * norm)) |
            ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[3] * norm)) << 16);
}

// nbatch_fa rows of raw K or V (pitch stride_bytes) decoded into a half2 tile. Items: (row, 128-block, quarter) for xyzkv2,
// (row, 32-block) for q4_0 -- nbatch_fa * 8 of them either way. batch_ok: every load of a thread's items, then every store.
template <ggml_type type, int stride_tile, int nbatch_fa, int nthreads, bool swz, bool batch_ok>
__device__ __forceinline__ void load_tile(const char * __restrict__ raw, half2 * __restrict__ tile, const int stride_bytes) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int items_per_row = D / QK4_0;
        constexpr int n_items       = nbatch_fa * items_per_row;
        if constexpr (batch_ok && n_items % nthreads == 0) {
            constexpr int per_thread = n_items / nthreads;
            uint32_t d[per_thread];
            uint32_t qs[per_thread][4];
#pragma unroll
            for (int it = 0; it < per_thread; ++it) {
                const int item = tid + it*nthreads;
                q4_0_load_block(raw + (int64_t) (item / items_per_row) * stride_bytes
                    + (item % items_per_row) * (int) sizeof(block_q4_0), d[it], qs[it]);
            }
#pragma unroll
            for (int it = 0; it < per_thread; ++it) {
                const int item = tid + it*nthreads;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    tile_put4<stride_tile, swz>(tile, item / items_per_row,
                        (item % items_per_row)*(QK4_0/2) + 4*c, q4_0_chunk(d[it], qs[it], c));
                }
            }
            return;
        }
        for (int item = tid; item < n_items; item += nthreads) {
            const int row = item / items_per_row;
            const int h2  = (item % items_per_row) * (QK4_0/2);
            uint32_t d;
            uint32_t qs[4];
            q4_0_load_block(raw + (int64_t) row * stride_bytes + (item % items_per_row) * (int) sizeof(block_q4_0), d, qs);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                tile_put4<stride_tile, swz>(tile, row, h2 + 4*c, q4_0_chunk(d, qs, c));
            }
        }
    } else {
        constexpr int blocks_per_row = D / QK_XYZKV2;
        constexpr int quarters       = 4;
        constexpr int h2_per_q       = QK_XYZKV2 / (2*quarters);
        constexpr int items_per_row  = blocks_per_row * quarters;
        constexpr int n_items        = nbatch_fa * items_per_row;
        constexpr int qs_u16         = QK_XYZKV2 / (8*quarters);
        if constexpr (batch_ok && n_items % nthreads == 0) {
            constexpr int per_thread = n_items / nthreads;
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
                const uint32_t lut_lo = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[0] * norm)) |
                                       ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[1] * norm)) << 16);
                const uint32_t lut_hi = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[2] * norm)) |
                                       ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[3] * norm)) << 16);
#pragma unroll
                for (int u = 0; u < qs_u16; ++u) {
                    const uint32_t w = words[it][u];
                    uint32_t v4[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
                        v4[p] = __byte_perm(lut_lo, lut_hi, t*0x22u + 0x1010u);
                    }
                    tile_put4<stride_tile, swz>(tile, row,
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
            const int h2_base = blk_idx*(QK_XYZKV2/2) + q*h2_per_q;
            const char * row_ptr = raw + (int64_t) row * stride_bytes;
            {
                const volatile uint16_t * blk16 =
                    (const volatile uint16_t *) (row_ptr + blk_idx * (int) sizeof(block_xyzkv2_0));
                ggml_half norm_h;
                const uint16_t norm_bits = blk16[0];
                memcpy(&norm_h, &norm_bits, sizeof(norm_h));
                const float norm = __half2float(norm_h);
                const uint32_t lut_lo = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[0] * norm)) |
                                       ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[1] * norm)) << 16);
                const uint32_t lut_hi = (uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[2] * norm)) |
                                       ((uint32_t) __half_as_ushort(__float2half(xyzkv2_centroid[3] * norm)) << 16);
#pragma unroll
                for (int u = 0; u < qs_u16; ++u) {
                    const uint32_t w = blk16[1 + q*qs_u16 + u];
                    uint32_t v4[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
                        v4[p] = __byte_perm(lut_lo, lut_hi, t*0x22u + 0x1010u);
                    }
                    tile_put4<stride_tile, swz>(tile, row, h2_base + 4*u, make_uint4(v4[0], v4[1], v4[2], v4[3]));
                }
            }
        }
    }
}

// the raw words of this thread's decode units, loaded before the MMA loop they are threaded through
template <ggml_type type, int nbatch_fa, int nthreads, int pitch>
__device__ __forceinline__ void decode_load(const char * __restrict__ stage, DecodeRegs<nbatch_fa, nthreads> & r) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int items_per_row = D / QK4_0;
#pragma unroll
        for (int it = 0; it < r.items; ++it) {
            const int item = tid + it * nthreads;
            q4_0_load_block(stage + (item / items_per_row) * pitch + (item % items_per_row) * (int) sizeof(block_q4_0),
                            r.nbits[it], r.words[it]);
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

// the four byte-permute selectors of one 16-bit word of codes from two spreads and two IMADs (the same selectors)
__device__ __forceinline__ void decode_word_fast(const uint32_t w, const uint32_t (&lut)[2], uint32_t (&v4)[4]) {
    const uint32_t x0 = w & 0xFFu;
    const uint32_t x1 = (w >> 8) & 0xFFu;
    const uint32_t t0 = (x0 & 0x33u) | ((x0 & 0xCCu) << 6);
    const uint32_t t1 = (x1 & 0x33u) | ((x1 & 0xCCu) << 6);
    const uint32_t y0 = (t0 & 0x0303u) | ((t0 & 0x3030u) << 12);
    const uint32_t y1 = (t1 & 0x0303u) | ((t1 & 0x3030u) << 12);
    const uint32_t s0 = y0*0x22u + 0x10101010u;
    const uint32_t s1 = y1*0x22u + 0x10101010u;
    v4[0] = __byte_perm(lut[0], lut[1], s0);
    v4[1] = __byte_perm(lut[0], lut[1], s0 >> 16);
    v4[2] = __byte_perm(lut[0], lut[1], s1);
    v4[3] = __byte_perm(lut[0], lut[1], s1 >> 16);
}

// decode unit u (one word of codes -> one 16-byte tile store) from the preloaded raw words; the item's LUT is built at
// its first word into lut, so units 0 .. units-1 in order do exactly load_tile's work
template <ggml_type type, int stride_tile, int nbatch_fa, int nthreads, bool swz, bool fast_sel>
__device__ __forceinline__ void decode_unit_regs(const DecodeRegs<nbatch_fa, nthreads> & r, half2 * __restrict__ tile,
                                                 const int u, uint32_t (&lut)[2]) {
    const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int item = tid + (u / 4) * nthreads;
    if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int items_per_row = D / QK4_0;
        tile_put4<stride_tile, swz>(tile, item / items_per_row,
            (item % items_per_row)*(QK4_0/2) + 4*(u % 4), q4_0_chunk(r.nbits[u / 4], r.words[u / 4], u % 4));
        return;
    }
    constexpr int items_per_row = (D / QK_XYZKV2) * 4;
    if (u % 4 == 0) {
        xyzkv2_lut(r.nbits[u / 4], lut);
    }
    const uint32_t w = r.words[u / 4][u % 4];
    uint32_t v4[4];
    if constexpr (fast_sel) {
        decode_word_fast(w, lut, v4);
    } else {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t t = ((w >> (4*p)) & 0x3u) | (((w >> (4*p + 2)) & 0x3u) << 8);
            v4[p] = __byte_perm(lut[0], lut[1], t*0x22u + 0x1010u);
        }
    }
    tile_put4<stride_tile, swz>(tile, item / items_per_row,
        ((item % items_per_row) / 4)*(QK_XYZKV2/2) + (item % 4)*16 + 4*(u % 4), make_uint4(v4[0], v4[1], v4[2], v4[3]));
}

// one tile's raw rows into the padded stage in 16-byte copies (raw_aligned: the first row minus its misalignment)
template <ggml_type type, int nbatch_fa, int nthreads>
__device__ __forceinline__ void stage_issue16(const char * __restrict__ raw_aligned, char * __restrict__ stage, const int stride_bytes) {
    constexpr int chunks = stage_pitch<type, true>() / 16;
    constexpr int n      = nbatch_fa * chunks;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int c0 = 0; c0 < n; c0 += nthreads) {
        const int c = c0 + tid;
        if (n % nthreads == 0 || c < n) {
            const int row = c / chunks;
            cp_async_16(stage + 16*c, raw_aligned + (int64_t) row * stride_bytes + 16*(c - row*chunks));
        }
    }
}

// one tile's raw rows into the packed stage in 4-byte copies
template <ggml_type type, int nbatch_fa, int nthreads>
__device__ __forceinline__ void stage_issue(const char * __restrict__ raw, char * __restrict__ stage, const int stride_bytes) {
    constexpr int rb            = row_bytes<type>();
    static_assert(rb % 4 == 0, "a native row must be a whole number of 4-byte words");
    constexpr int words_per_row = rb / 4;
    constexpr int n_words       = nbatch_fa * words_per_row;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int w = tid; w < n_words; w += nthreads) {
        const int row = w / words_per_row;
        const int col = w - row * words_per_row;
        cp_async_4(stage + row * rb + col * 4, raw + (int64_t) row * stride_bytes + col * 4);
    }
}

// ---- the mask tile ----------------------------------------------------------------------------------------------------
// use_cp_async: 16-byte copies (the ring); pos_mask: the positional vector [qpos | kpos] -> +0 / -inf
template <int nrows, int nwarps, int nbatch_fa, bool use_cp_async, bool pos_mask>
__device__ __forceinline__ void load_mask(const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
                                          const int stride_mask, const int k_VKQ_0, const int j0, const uint3 ne01) {
    if constexpr (pos_mask) {
        const float * const qpos = (const float *) mask_h;
        const float * const kpos = qpos + (-stride_mask);
#pragma unroll
        for (int j1 = 0; j1 < nrows; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            if (j1 + nwarps > nrows && j_sram >= nrows) {
                break;
            }
            const float qp = qpos[fastmodulo(j0 + j_sram, ne01)];
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > nbatch_fa && i >= nbatch_fa) {
                    break;
                }
                tile_mask[j_sram*(nbatch_fa + 8) + i] = kpos[k_VKQ_0 + i] < qp ? half(0.0f) : half(-INFINITY);
            }
        }
        return;
    }
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*WARP_SIZE && nbatch_fa % 8 == 0, "bad nbatch_fa");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);
#pragma unroll
        for (int j1 = 0; j1 < nrows; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
            if (j1 + stride_j > nrows && j_sram >= nrows) {
                break;
            }
            const int i = 8 * (threadIdx.x % (nbatch_fa/8));
            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half),
                                    mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (nbatch_fa < 2*WARP_SIZE) {
        constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < nrows; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
            if (j1 + stride_j > nrows && j_sram >= nrows) {
                break;
            }
            const int i = threadIdx.x % (WARP_SIZE/cols_per_warp);
            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < nrows; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
            if (j1 + nwarps > nrows && j_sram >= nrows) {
                break;
            }
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*WARP_SIZE) {
                const int i = i0 + 2*threadIdx.x;
                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

// ---- the ring: slot kb % RING <- tile kb's K rows; V rows + mask ------------------------------------------------------
template <ggml_type type, int nbatch_fa, int nwarps, int nrows_mask, bool wide = false>
__device__ __forceinline__ void ring_issue_K(char * const stage, const int kb, const half2 * const __restrict__ K_h2,
                                             const int stride_K) {
    constexpr int nthreads = nwarps * WARP_SIZE;
    char * const slot = stage + (kb % RING) * slot_bytes<type, wide>(nbatch_fa, nrows_mask);
    const int stride_K_bytes = stride_K * (int) sizeof(half2);
    const char * src = (const char *) K_h2 + (int64_t) kb*nbatch_fa*stride_K_bytes;
    if constexpr (wide) {
        stage_issue16<type, nbatch_fa, nthreads>(src - ((uintptr_t) src & 15), slot, stride_K_bytes);
    } else {
        stage_issue<type, nbatch_fa, nthreads>(src, slot, stride_K_bytes);
    }
}

// kb_vis: tiles below it lie wholly in the prefix every row sees -- their mask is all zeros, so it is neither copied nor
// added (x + 0 is x but for -0 -> +0, and the max, the exp and the P tile cannot tell those apart: bit-identical)
template <ggml_type type, int nbatch_fa, int nwarps, int nrows_mask, bool wide = false, bool pos_mask = false>
__device__ __forceinline__ void ring_issue_V(char * const stage, const int kb, const half2 * const __restrict__ V_h2,
                                             const int stride_V, const half * const __restrict__ mask_h,
                                             const int stride_mask, const int j0, const uint3 ne01, const int kb_vis) {
    constexpr int nthreads = nwarps * WARP_SIZE;
    constexpr int rowK     = stage_pitch<type, wide>();
    constexpr int rowV     = stage_pitch<type, wide>();
    char * const slot = stage + (kb % RING) * slot_bytes<type, wide>(nbatch_fa, nrows_mask);
    const int stride_V_bytes = stride_V * (int) sizeof(half2);
    const char * src = (const char *) V_h2 + (int64_t) kb*nbatch_fa*stride_V_bytes;
    if constexpr (wide) {
        stage_issue16<type, nbatch_fa, nthreads>(src - ((uintptr_t) src & 15), slot + nbatch_fa*rowK, stride_V_bytes);
    } else {
        stage_issue<type, nbatch_fa, nthreads>(src, slot + nbatch_fa*rowK, stride_V_bytes);
    }
    if (kb >= kb_vis) {
        load_mask<nrows_mask, nwarps, nbatch_fa, true, pos_mask>
            (mask_h, (half *) (slot + nbatch_fa*(rowK + rowV)), stride_mask, kb*nbatch_fa, j0, ne01);
    }
}

// ---- one KV tile: KQ, the online softmax, PV --------------------------------------------------------------------------
template <int ncols1, int nwarps, bool last_iter, ggml_type type, int feat>
__device__ __forceinline__ void iter(
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const float slope,
        const uint3 ne01,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        typename Tiles<ncols1*NCOLS2>::T_B_KQ  * const __restrict__ Q_B,
        typename Tiles<ncols1*NCOLS2>::T_C_VKQ * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int kb0_stop,
        const int k_VKQ_sup,
        const int d,
        const int tpt,
        const int kb_vis) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int ncols = ncols1 * NCOLS2;
    using T_A_KQ  = typename Tiles<ncols>::T_A_KQ;
    using T_B_KQ  = typename Tiles<ncols>::T_B_KQ;
    using T_C_KQ  = typename Tiles<ncols>::T_C_KQ;
    using T_A_VKQ = typename Tiles<ncols>::T_A_VKQ;
    using T_B_VKQ = typename Tiles<ncols>::T_B_VKQ;
    using T_C_VKQ = typename Tiles<ncols>::T_C_VKQ;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2;
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols;
    constexpr int  nbatch_fa       = config(ncols).nbatch_fa;
    constexpr int  nbatch_K2       = config(ncols).nbatch_K2;
    constexpr int  nbatch_V2       = config(ncols).nbatch_V2;
    constexpr int  stride_tile_K   = swz::tile_stride(nbatch_K2);
    constexpr int  stride_tile_V   = swz::tile_stride(nbatch_V2);
    constexpr bool swz_K           = swz::enabled(nbatch_K2);
    constexpr bool swz_V           = swz::enabled(nbatch_V2);
    constexpr int  nrows_mask      = mask_rows(ncols1);
    constexpr bool ring            = ring_enabled<ncols>();
    constexpr bool fuse            = fuse_enabled<ncols>();
    constexpr bool wide            = fuse && (feat & F_WIDE) != 0;
    constexpr int  PK              = stage_pitch<type, wide>();
    constexpr int  PV              = stage_pitch<type, wide>();
    constexpr bool fast_sel        = wide && (feat & F_FAST_SEL) != 0;
    constexpr bool pos_mask        = (feat & F_POS_MASK) != 0;
    constexpr int  nthreads_fuse   = nwarps * WARP_SIZE;
    constexpr int  st_off          = stage_off(nbatch_fa,
        kv_bufs<ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), nrows_mask);

    const int k_VKQ_0 = kb0 * nbatch_fa;
    [[maybe_unused]] const int oK = wide ? (int) ((uintptr_t) K_h2 & 15) : 0;
    [[maybe_unused]] const int oV = wide ? (int) ((uintptr_t) V_h2 & 15) : 0;
    [[maybe_unused]] uint32_t lut_fuse[2];
    [[maybe_unused]] const char * V_stage_fuse = nullptr;
    [[maybe_unused]] DecodeRegs<nbatch_fa, nthreads_fuse> V_regs_fuse;
    [[maybe_unused]] DecodeRegs<nbatch_fa, nthreads_fuse> K_regs_fuse;
    half * tile_mask_it = tile_mask;

    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];

    const bool mask_add = kb0 >= kb_vis;   // below kb_vis the tile's mask is all zeros (ring_issue_V)
    if constexpr (!ring) {
        if (mask_add) {
            load_mask<nrows_mask, nwarps, nbatch_fa, false, pos_mask>(mask_h, tile_mask, stride_mask, k_VKQ_0, jt*tpt, ne01);
        }
    }

#pragma unroll
    for (int k0_start = (D/2-1) - (D/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < D/2 ? k0_start + nbatch_K2 : D/2;

        static_assert(nbatch_K2 == D/2, "the native K tile loader requires unbatched K");
        const int stride_K_bytes = stride_K * (int) sizeof(half2);
        constexpr int nthreads_tile = nwarps * WARP_SIZE;
        const char * K_raw   = (const char *) K_h2 + (int64_t) k_VKQ_0 * stride_K_bytes;
        int          K_pitch = stride_K_bytes;
        char * const raw_K = (char *) tile_Q + st_off;
        if constexpr (ring) {
            if constexpr (!fuse) {
                // land ALL of tile kb0 (its K group, its V + mask group); the newer groups stay in flight
                cp_async_wait_group<2*RING - 2>();
                __syncthreads();
            }   // fused: tile kb0 landed and was published by the previous barrier A (or the prologue's barrier)
            char * const slot = raw_K + (kb0 % RING) * slot_bytes<type, wide>(nbatch_fa, nrows_mask);
            K_raw        = slot + oK;
            K_pitch      = PK;
            V_stage_fuse = slot + nbatch_fa*PK + oV;
            if constexpr (fuse) {
                decode_load<type, nbatch_fa, nthreads_fuse, PV>(V_stage_fuse, V_regs_fuse);
            }
            tile_mask_it = (half *) (slot + nbatch_fa*(PK + PV));
        } else {
            cp_async_wait_all();
            __syncthreads();
            K_raw   = raw_K;
            K_pitch = row_bytes<type>();
        }
        if constexpr (!fuse) {   // fused: K(kb0) was decoded under the previous tile's PV (or the prologue)
            load_tile<type, stride_tile_K, nbatch_fa, nthreads_tile, swz_K, (ncols < BATCH_MAX_NCOLS)>(K_raw, tile_K, K_pitch);
            __syncthreads();
        }
        if constexpr (ring) {
            if constexpr (!fuse) {   // this slot's K half is consumed: refill it with tile kb0 + RING (one commit, always)
                if (kb0 + RING < kb0_stop) {
                    ring_issue_K<type, nbatch_fa, nwarps, nrows_mask>(raw_K, kb0 + RING, K_h2, stride_K);
                }
                cp_async_commit();
            }
        } else if constexpr (!last_iter) {
            stage_issue<type, nbatch_fa, nthreads_tile>
                ((const char *) K_h2 + (int64_t) (k_VKQ_0 + nbatch_fa) * stride_K_bytes, raw_K, stride_K_bytes);
        }

        // KQ (Q in registers)
#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                T_A_KQ K_A;
                swz::load_ldmatrix<stride_tile_K, swz_K>(K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);
                if constexpr (cols_per_warp == 8) {
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                } else {
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);   // wide KQ_C is column-major
                }
                if constexpr (fuse) {   // one unit of V(kb0)'s decode between this step's HMMAs and the next step's
                    constexpr int ksteps = (D/2)/T_A_KQ::J;
                    constexpr int steps  = (nbatch_fa/(np*T_A_KQ::I)) * ksteps;
                    constexpr int units  = decode_units<nbatch_fa, nthreads_fuse>();
                    static_assert(steps % units == 0, "the V decode units must spread evenly over the KQ steps");
                    const int step = (i_KQ_00/(np*T_A_KQ::I))*ksteps + (k_KQ_0 - k0_start)/T_A_KQ::J;
                    if (step % (steps/units) == 0) {
                        decode_unit_regs<type, stride_tile_V, nbatch_fa, nthreads_fuse, swz_V, fast_sel>(
                            V_regs_fuse, tile_V, step/(steps/units), lut_fuse);
                    }
                }
            }
        }

        if constexpr (fuse) {
            // barrier A: tile kb0+1 has landed for this thread; the barrier publishes it and V(kb0), and puts every warp
            // past its KQ reads of tile_K
            cp_async_wait_group<(RING >= 2 ? RING - 2 : 0)>();
            __syncthreads();
        } else {
            __syncthreads();   // tile_K == tile_V
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (mask_add) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / d;
                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask_it[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int KQ_idx = l % 2;
                KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + KQ_MAX_OFFSET);
            }
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int KQ_idx = l % 2;
                KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
            }
        }
    } else {
        if (mask_add) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / d;
                    const float2 tmp = __half22float2(((const half2 *)tile_mask_it)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int KQ_idx = (l/2) % 2;
                KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + KQ_MAX_OFFSET);
            }
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int KQ_idx = (l/2) % 2;
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ;

            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        // every scale == 1.0 -> the rescale is an identity; skipped warp-uniformly (not on the 32-column tile)
        bool rescale = false;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            rescale |= KQ_max_scale[col] != 1.0f;
        }
        if (ncols == SKIP_EXCLUDE_NCOLS || __any_sync(0xFFFFFFFF, rescale))
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < D/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (D/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // KQ C tiles -> B tiles for PV
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    // PV
#pragma unroll
    for (int i0_start = 0; i0_start < D; i0_start += 2*nbatch_V2) {
        static_assert(D % (2*nbatch_V2) == 0, "bad loop size");
        static_assert(2*nbatch_V2 == D, "the native V tile loader requires unbatched V");
        const int i0_stop = i0_start + 2*nbatch_V2;

        const int stride_V_bytes = stride_V * (int) sizeof(half2);
        constexpr int nthreads_tile = nwarps * WARP_SIZE;
        const char * V_raw   = (const char *) V_h2 + (int64_t) k_VKQ_0 * stride_V_bytes;
        int          V_pitch = stride_V_bytes;
        char * const raw_V = (char *) tile_Q + st_off + nbatch_fa * row_bytes<type>();
        if constexpr (ring) {
            V_raw   = (const char *) tile_Q + st_off + nbatch_fa * row_bytes<type>()
                    + (kb0 % RING) * slot_bytes<type>(nbatch_fa, nrows_mask);
            V_pitch = row_bytes<type>();
        } else {
            V_raw   = raw_V;
            V_pitch = row_bytes<type>();
        }
        if constexpr (kv_bufs<ncols>() == 1) {
            load_tile<type, stride_tile_V, nbatch_fa, nthreads_tile, swz_V, (ncols < BATCH_MAX_NCOLS)>(V_raw, tile_V, V_pitch);
        }
        if constexpr (!fuse) {   // fused: barrier A already published V(kb0)
            __syncthreads();
        }
        if constexpr (ring) {
            if constexpr (!fuse) {   // this slot's V half and mask are consumed: refill them with tile kb0 + RING
                if (kb0 + RING < kb0_stop) {
                    ring_issue_V<type, nbatch_fa, nwarps, nrows_mask, false, pos_mask>((char *) tile_Q + st_off,
                        kb0 + RING, V_h2, stride_V, mask_h, stride_mask, jt*tpt, ne01, kb_vis);
                }
                cp_async_commit();
            }
        } else if constexpr (!last_iter) {
            stage_issue<type, nbatch_fa, nthreads_tile>
                ((const char *) V_h2 + (int64_t) (k_VKQ_0 + nbatch_fa) * stride_V_bytes, raw_V, stride_V_bytes);
        }
        const half2 * tile_V_i = tile_V;
        if constexpr (fuse && !last_iter) {   // K(kb0+1)'s raw words, before the PV loop
            decode_load<type, nbatch_fa, nthreads_fuse, PK>((const char *) tile_Q + st_off
                + ((kb0 + 1) % RING) * slot_bytes<type, wide>(nbatch_fa, nrows_mask) + oK, K_regs_fuse);
        }

#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A;   // transposed in shared memory, transposed on load
                swz::load_ldmatrix_trans<stride_tile_V, swz_V>(A, tile_V, (int)(tile_V_i - tile_V) + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);   // wide VKQ_C is column-major
                }
                if constexpr (fuse && !last_iter) {   // one unit of K(kb0+1)'s decode between the PV steps
                    constexpr int ksteps = (nbatch_fa/2)/(np*T_A_VKQ::J);
                    constexpr int steps  = ((2*nbatch_V2)/T_A_VKQ::I) * ksteps;
                    constexpr int units  = decode_units<nbatch_fa, nthreads_fuse>();
                    static_assert(steps % units == 0, "the K decode units must spread evenly over the PV steps");
                    const int step = ((i_VKQ_0 - i0_start)/T_A_VKQ::I)*ksteps + k00/(np*T_A_VKQ::J);
                    if (step % (steps/units) == 0) {
                        decode_unit_regs<type, stride_tile_K, nbatch_fa, nthreads_fuse, swz_K, fast_sel>(
                            K_regs_fuse, tile_K, step/(steps/units), lut_fuse);
                    }
                }
            }
        }

        if constexpr (kv_bufs<ncols>() == 1) {
            __syncthreads();   // tile_K == tile_V
        }
        if constexpr (fuse) {
            // barrier B: K(kb0+1) published; every warp is past its PV reads of tile_V and its softmax reads of this
            // slot's mask -- refill the whole slot with tile kb0 + RING (one group per tile, always committed)
            __syncthreads();
            if (kb0 + RING < kb0_stop) {
                ring_issue_K<type, nbatch_fa, nwarps, nrows_mask, wide>((char *) tile_Q + st_off, kb0 + RING, K_h2, stride_K);
                ring_issue_V<type, nbatch_fa, nwarps, nrows_mask, wide, pos_mask>((char *) tile_Q + st_off, kb0 + RING, V_h2,
                    stride_V, mask_h, stride_mask, jt*tpt, ne01, kb_vis);
            }
            cp_async_commit();
        }
    }
#else
    GGML_UNUSED_VARS(K_h2, V_h2, mask_h, slope, ne01, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt, kb_vis);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// ---- one output tile over KV tiles [kb0_start, kb0_stop): Q in, the iterations, the combine, the write-out ----------
template <int ncols1, int nwarps, bool needs_fixup, bool is_fixup, ggml_type type, int feat>
__device__ __forceinline__ void process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        const int d,
        const int tpt,
        const int kb_vis) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int ncols = ncols1 * NCOLS2;
    using T_B_KQ  = typename Tiles<ncols>::T_B_KQ;
    using T_C_VKQ = typename Tiles<ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2;
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols;
    constexpr int  nbatch_fa       = config(ncols).nbatch_fa;
    constexpr int  nbatch_K2       = config(ncols).nbatch_K2;
    constexpr int  nbatch_V2       = config(ncols).nbatch_V2;
    constexpr int  nbatch_combine  = config(ncols).nbatch_combine;
    constexpr bool Q_in_reg        = config(ncols).Q_in_reg;
    constexpr int  nrows_mask      = mask_rows(ncols1);
    static_assert(Q_in_reg, "every tile here keeps Q in registers");
    static_assert(nwarps * (cols_per_warp/NCOLS2) % ncols1 == 0, "bad nwarps");

    constexpr int  stride_tile_Q      = D/2 + 4;
    constexpr int  stride_tile_K      = swz::tile_stride(nbatch_K2);
    constexpr int  stride_tile_V      = swz::tile_stride(nbatch_V2);
    constexpr int  stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;
    constexpr bool swz_K              = swz::enabled(nbatch_K2);
    constexpr bool swz_V              = swz::enabled(nbatch_V2);

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = tile_Q;
    constexpr bool sep_V = kv_bufs<ncols>() == 2;
    half2 * tile_V    = sep_V ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (sep_V ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ  Q_B[D/(2*T_B_KQ::J)];
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? D/T_C_VKQ::I : D/(2*T_C_VKQ::J)];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Q into tile_Q (scaled, f16), then into registers
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4, WARP_SIZE/8}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            // column jc = (token, head) pair with group size d; the last ncols - tpt*d columns of a packed tile are dead
            const int j = jc / d;
            const int c = jc % d;

            if (jc < tpt*d && (ncols1 == 1 || jt*tpt + j < int(ne01.z)) && (zt_gqa*d + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*tpt + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;
#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    {
        // the stage: the first tile(s) in flight before the loop; the first iteration lands them
        constexpr int nthreads_tile = nwarps * WARP_SIZE;
        constexpr int st_off        = stage_off(nbatch_fa, kv_bufs<ncols>()*stride_tile_KV_max, nrows_mask);
        constexpr bool pos_mask     = (feat & F_POS_MASK) != 0;
        char * const raw_K = (char *) tile_Q + st_off;
        static_assert((nbatch_fa * row_bytes<type>()) % 16 == 0, "raw_V must stay 16-byte aligned");
        if constexpr (fuse_enabled<ncols>()) {
            // the first RING tiles in flight, ONE group per tile; land tile kb0 and decode its K
            constexpr bool wide = (feat & F_WIDE) != 0;
            constexpr int slot = slot_bytes<type, wide>(nbatch_fa, nrows_mask);
            const int oK = wide ? (int) ((uintptr_t) K_h2 & 15) : 0;
#pragma unroll
            for (int s = 0; s < RING; ++s) {
                if (kb0 + s < kb0_stop) {
                    ring_issue_K<type, nbatch_fa, nwarps, nrows_mask, wide>(raw_K, kb0 + s, K_h2, stride_K);
                    ring_issue_V<type, nbatch_fa, nwarps, nrows_mask, wide, pos_mask>(raw_K, kb0 + s, V_h2, stride_V,
                        mask_h, stride_mask, jt*tpt, ne01, kb_vis);
                }
                cp_async_commit();
            }
            cp_async_wait_group<RING - 1>();
            __syncthreads();
            load_tile<type, stride_tile_K, nbatch_fa, nthreads_tile, swz_K, (ncols < BATCH_MAX_NCOLS)>
                (raw_K + (kb0 % RING)*slot + oK, tile_K, stage_pitch<type, wide>());
            __syncthreads();
        } else if constexpr (ring_enabled<ncols>()) {
            // per tile a K group, then a V + mask group (empty past kb0_stop): the same two groups every iteration commits
#pragma unroll
            for (int s = 0; s < RING; ++s) {
                if (kb0 + s < kb0_stop) {
                    ring_issue_K<type, nbatch_fa, nwarps, nrows_mask>(raw_K, kb0 + s, K_h2, stride_K);
                }
                cp_async_commit();
                if (kb0 + s < kb0_stop) {
                    ring_issue_V<type, nbatch_fa, nwarps, nrows_mask, false, pos_mask>(raw_K, kb0 + s, V_h2, stride_V,
                        mask_h, stride_mask, jt*tpt, ne01, kb_vis);
                }
                cp_async_commit();
            }
        } else {
            char * const raw_V = raw_K + nbatch_fa * row_bytes<type>();
            stage_issue<type, nbatch_fa, nthreads_tile>
                ((const char *) K_h2 + (int64_t) (kb0*nbatch_fa) * (stride_K * (int) sizeof(half2)), raw_K,
                 stride_K * (int) sizeof(half2));
            stage_issue<type, nbatch_fa, nthreads_tile>
                ((const char *) V_h2 + (int64_t) (kb0*nbatch_fa) * (stride_V * (int) sizeof(half2)), raw_V,
                 stride_V * (int) sizeof(half2));
        }

        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr int k_VKQ_sup = nbatch_fa;
            iter<ncols1, nwarps, false, type, feat>(K_h2, V_h2, mask_h, slope, ne01, stride_K, stride_V, stride_mask,
                tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt, kb_vis);
        }
        constexpr int k_VKQ_sup = nbatch_fa;
        iter<ncols1, nwarps, true, type, feat>(K_h2, V_h2, mask_h, slope, ne01, stride_K, stride_V, stride_mask,
            tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt, kb_vis);
    }

    // the partial KQ rowsums, spread across 8 (8-column tile) / 4 threads
    {
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // combine the VKQ accumulators of the np parallel warps (and write column-major through shared memory either way)
    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((D/2) % nbatch_combine == 0, "bad nbatch_combine");
    constexpr bool combine_needs_sync = swz_K || swz_V;

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J;
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo;
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]);

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1 && threadIdx.y % np == 0) {
        // the meta data of the parallel warps via shared memory; warps with threadIdx.y % np != 0 must not return early
        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x;   // the max between all parallel warps
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta];   // the max scale per warp
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y;   // the scaled rowsum of all parallel warps
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        __syncthreads();   // the if branch syncs once: so must every other warp
    }

#pragma unroll
    for (int k00 = 0; k00 < D/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1);
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]);   // C to B puts it in column-major format

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // the first 2*2*gridDim.x*ncols floats of dstk_fixup: the max values and row sums; then the blocks' partials
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4, WARP_SIZE/8}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / d;
                    const int c_dst = jc_dst % d;

                    if (!is_fixup && (jc_dst >= tpt*d || (ncols1 > 1 && jt*tpt + j_dst >= int(ne01.z)) || (zt_gqa*d + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(D/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*tpt + j_dst)*ne02 + c_dst)*(D/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        // close the pass when another follows (the next pass overwrites the combine buffer other warps may still read)
        if (np > 1 || k00 + nbatch_combine < D/2) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, ne01, ne02, gqa_ratio, stride_Q1, stride_Q2,
        stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, d, tpt, kb_vis);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// ---- the kernel: stream-K over (KV tile, output tile) work items --------------------------------------------------------
template <int ncols1, ggml_type type, int feat>
__launch_bounds__(config(ncols1*NCOLS2).nthreads, config(ncols1*NCOLS2).occupancy)
__global__ void k_fa(const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
                     const char * __restrict__ mask, float * __restrict__ dst, float2 * __restrict__ dst_meta,
                     const float scale, const uint3 ne01, const int32_t ne02, const int32_t nb01, const int32_t nb02,
                     const int32_t ne11, const int32_t ne12, const int32_t nb11, const int32_t nb12,
                     const int32_t nb21, const int32_t nb22, const int32_t nb31, const int * __restrict__ vis_pos) {
    ggml_cuda_pdl_sync();
#if defined(TURING_MMA_AVAILABLE)
    constexpr int ncols     = ncols1 * NCOLS2;
    constexpr int nbatch_fa = config(ncols).nbatch_fa;
    constexpr int nthreads  = config(ncols).nthreads;
    constexpr int nwarps    = nthreads / WARP_SIZE;

    // vis_pos (optional): the first row's position -- every row sees cells 0 .. *vis_pos, so the KV tiles below kb_vis
    // carry an all-zero mask (ring_issue_V)
    const int kb_vis = vis_pos != nullptr ? (*vis_pos + 1) / nbatch_fa : 0;

    const int gqa_ratio = ne02 / ne12;
    const int d   = pair_d(ncols1, gqa_ratio, (int) ne01.z);
    const int tpt = ncols / d;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / (int) sizeof(half);   // signed: the positional mask passes -n_q halves here
    const int stride_V    = nb21 / sizeof(half2);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (tpt       - 1)) / tpt;
    const int iter_z_gqa = (gqa_ratio + (d         - 1)) / d;

    // kbc: the current index in the continuous (KV tile, token tile, head group, KV head) space
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12) / gridDim.x;

    // a seam inside an output tile: the block that starts it needs_fixup, the one that finishes it is_fixup
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    const float slope = 1.0f;

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int z_KV   = kbc/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa = (kbc - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt     = (kbc - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*d;   // the global Q head start

        const float2 * Q_f2   = (const float2 *) (Q + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb12*z_KV);
        const half   * mask_h = (const half   *) mask;
        float2       * dstk   = ((float2 *) dst) + zt_Q * (D/2);
        const half2  * V_h2   = (const half2  *) (V + nb22*z_KV);

        if (kb0_start == 0) {
            process_tile<ncols1, nwarps, false, false, type, feat>(Q_f2, K_h2, V_h2, mask_h, dstk, dst_meta, scale, slope,
                ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start,
                kb0_stop, d, tpt, kb_vis);
        } else {
            process_tile<ncols1, nwarps, true, false, type, feat>(Q_f2, K_h2, V_h2, mask_h, dstk, dst_meta, scale, slope,
                ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start,
                kb0_stop, d, tpt, kb_vis);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int z_KV   = kbc/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa = (kbc - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt     = (kbc - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*d;

    const float2 * Q_f2   = (const float2 *) (Q + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb12*z_KV);
    const half   * mask_h = (const half   *) mask;
    float2       * dstk   = ((float2 *) dst) + zt_Q * (D/2);
    const half2  * V_h2   = (const half2  *) (V + nb22*z_KV);

    // the last item writes its data to the fixup buffer (no race with the other blocks of the tile)
    process_tile<ncols1, nwarps, false, true, type, feat>(Q_f2, K_h2, V_h2, mask_h, dstk, dst_meta, scale, slope,
        ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, d, tpt,
        kb_vis);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst, dst_meta, scale, ne01, ne02, nb01, nb02, ne11, ne12, nb11, nb12, nb21, nb22, nb31,
        vis_pos);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// ---- the stream-K fixups: the partial results of a tile's blocks, combined in order ------------------------------------
template <int ncols1>
__launch_bounds__(D, 1)
__global__ void k_fixup_uniform(float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01,
                                const int ne02, const int nblocks_stream_k, const int gqa_ratio, const int blocks_per_tile,
                                const uint3 fd_iter_j_z_ne12, const uint3 fd_iter_j_z, const uint3 fd_iter_j,
                                const int tile_tokens, const int tile_heads) {
    constexpr int ncols = ncols1*NCOLS2;

    const int tile_idx = blockIdx.x;   // one block per output tile
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*tile_heads + c;
    const int tid      = threadIdx.x;

    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*tile_heads;

    if (jt*tile_tokens + j >= ne01 || zt_gqa*tile_heads + c >= gqa_ratio) {
        return;
    }

    dst += jt*ne02*(tile_tokens*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    *dst = dst_val / rowsum;
}

template <int ncols1>
__launch_bounds__(D, 1)
__global__ void k_fixup_general(float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01,
                                const int ne02, const int gqa_ratio, const int total_work, const uint3 fd_iter_k_j_z_ne12,
                                const uint3 fd_iter_k_j_z, const uint3 fd_iter_k_j, const uint3 fd_iter_k,
                                const int tile_tokens, const int tile_heads) {
    constexpr int ncols = ncols1*NCOLS2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*tile_heads + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*tile_heads;

    if (jt*tile_tokens + j >= ne01 || zt_gqa*tile_heads + c >= gqa_ratio) {
        return;
    }

    dst += jt*ne02*(tile_tokens*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while (true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) {   // did not have any data
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    *dst = dst_val / rowsum;
}

} // namespace fa
} // namespace eng
