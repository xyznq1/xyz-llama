// The engine's attention launches (src/fa/fa.cuh): the tile each width takes as the server dispatches it, the stream-K
// grid from the kernel's occupancy exactly as launch_fattn sizes it (the seams, and so the fixup's sums, fall where
// the server's do), and the fixup. fa_init() (outside any capture) raises the shared-memory limits, measures the
// occupancies once and allocates the fixup buffer for the widest tile.
#include "fa/fa.cuh"

#include "kernels.h"

#include <algorithm>

namespace eng {

namespace {

using namespace fa;

constexpr int N_HEAD_KV = 4;

struct Inst {
    const void * kernel = nullptr;
    size_t       smem   = 0;
    int          occ    = 0;
};

float2 * g_meta     = nullptr;   // the stream-K fixup buffer
size_t   g_meta_cap = 0;         // in float2
int      g_nsm      = 0;
int      g_twin_occ = 0;         // the (F16, F16) 64-column twin's occupancy: the prompt tile's grid above 32 rows
Inst     g_pos;                  // the positional-mask 64-column xyzkv2 instance (the prompt path)

template <int ncols1, ggml_type type, bool wide>
size_t smem_bytes() {
    constexpr int    ncols  = ncols1*NCOLS2;
    constexpr Config cfg    = config(ncols);
    constexpr int    nwarps = cfg.nthreads / WARP_SIZE;
    constexpr int    cols_per_warp = ncols < 16 ? ncols : 16;
    constexpr int    sK = swz::tile_stride(cfg.nbatch_K2);
    constexpr int    sV = swz::tile_stride(cfg.nbatch_V2);
    constexpr size_t kv_1stage = (size_t) cfg.nbatch_fa * std::max(sK, sV) * sizeof(half2);
    constexpr size_t kv_2stage = (size_t) cfg.nbatch_fa * (sK + sV) * sizeof(half2);
    constexpr size_t q_bytes   = (size_t) ncols * (D/2 + 4) * sizeof(half2);
    constexpr size_t mask      = (size_t) mask_rows(ncols1) * (cfg.nbatch_fa/2 + 4) * sizeof(half2);
    constexpr size_t combine   = (size_t) nwarps*cols_per_warp * (cfg.nbatch_combine + 4) * sizeof(half2);
    constexpr size_t kv        = kv_bufs<ncols>() == 1 ? kv_1stage : kv_2stage;
    size_t total = std::max(combine, cfg.Q_in_reg ? std::max(q_bytes, kv + mask) : q_bytes + kv + mask);
    const int off = stage_off(cfg.nbatch_fa, kv_bufs<ncols>()*std::max(sK, sV), mask_rows(ncols1));
    const size_t stage = ring_enabled<ncols>() ? (size_t) RING * slot_bytes<type>(cfg.nbatch_fa, mask_rows(ncols1)) :
                                                 (size_t) cfg.nbatch_fa * (row_bytes<type>() + row_bytes<type>());
    total = std::max(total, (size_t) off + stage);
    if constexpr (wide) {
        total = std::max(total, (size_t) off + (size_t) RING * slot_bytes<type, true>(cfg.nbatch_fa, mask_rows(ncols1)));
    }
    return total;
}

template <int ncols1, ggml_type type, bool wide>
Inst & inst() {
    static Inst i;
    return i;
}

template <int ncols1, ggml_type type, bool wide>
void init_inst() {
    Inst & i = inst<ncols1, type, wide>();
    const auto kernel = k_fa<ncols1, type, wide ? (int) (F_WIDE | F_FAST_SEL) : 0>;
    i.kernel = (const void *) kernel;
    i.smem   = smem_bytes<ncols1, type, wide>();
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) i.smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&i.occ, kernel, config(ncols1*NCOLS2).nthreads, i.smem));
    GGML_ASSERT(i.occ > 0);
    const size_t need = (size_t) i.occ * g_nsm * (ncols1*NCOLS2) * (2 + D/2);
    g_meta_cap = std::max(g_meta_cap, need);
}

// The occupancy ggml_cuda_flash_attn_ext_mma_f16_case sizes a grid by above REF_TOKENS rows (occupancy_ref): the (F16,
// F16) 64-column kernel's -- its shared memory by the fork's formula (config 64 with 2 stages on sm_89: the K and V tiles,
// Q, an ncols1-row mask, the combine buffer) on 128 threads under __launch_bounds__(128, 2), so shared memory decides.
// This kernel carries the same launch bounds, so its occupancy at that shared memory is the twin's (tools/fa_test.cu
// checks it against the fork's own computation).
int twin_occupancy() {
    constexpr int    ncols1  = 8, ncols = ncols1*NCOLS2;
    constexpr Config cfg     = config(ncols);
    constexpr int    nwarps  = cfg.nthreads / WARP_SIZE;
    constexpr int    sK      = swz::tile_stride(cfg.nbatch_K2);
    constexpr int    sV      = swz::tile_stride(cfg.nbatch_V2);
    constexpr size_t kv      = (size_t) cfg.nbatch_fa * (sK + sV) * sizeof(half2);
    constexpr size_t q       = (size_t) ncols * (D/2 + 4) * sizeof(half2);
    constexpr size_t mask    = (size_t) ncols1 * (cfg.nbatch_fa/2 + 4) * sizeof(half2);
    constexpr size_t combine = (size_t) nwarps*16 * (cfg.nbatch_combine + 4) * sizeof(half2);
    const size_t smem = std::max(combine, cfg.Q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
    int occ = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_fa<ncols1, GGML_TYPE_XYZKV2_0, 0>, cfg.nthreads, smem));
    return occ;
}

// every row of a head at one misalignment, each head's 16-byte window inside its token row (the fork's wide_ok)
template <ggml_type type>
bool wide_ok(const void * data, const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    constexpr int rb    = row_bytes<type>();
    constexpr int pitch = stage_pitch<type, true>();
    if ((uintptr_t) data % 16 != 0 || nb1 % 16 != 0 || nb3 % 16 != 0 || nb2 % 4 != 0) {
        return false;
    }
    if (rb % 16 == 0) {
        return nb2 % 16 == 0;
    }
    const int64_t start = (N_HEAD_KV - 1) * nb2;
    return start - start % 16 + pitch <= nb1;
}

// pos: the positional-mask instance (mask = the f32 vector [qpos | kpos], passed as qpos with nb31 = -n_tokens halves).
// The grid takes the occupancy the server's launch would: above REF_TOKENS rows the f16 twin's (the xyzkv2 prompt tile's
// occupancy_ref; the drafter's q4_0 prompt tile, where the server runs the f16 kernel itself), else the instance's (the
// plain one's for pos)
template <int ncols1, ggml_type type, bool pos = false>
void launch(cudaStream_t st, const float * q, const int n_tokens, const int n_head, const void * k, const void * v,
            const int n_kv, const int kv_size, const void * mask, const float scale, float * dst,
            const int * vis_pos = nullptr) {
    constexpr int    ncols  = ncols1*NCOLS2;
    constexpr Config cfg    = config(ncols);
    constexpr int    nwarps = cfg.nthreads / WARP_SIZE;

    const int64_t nb1 = (int64_t) row_bytes<type>() * N_HEAD_KV;   // a token row of the cache: 4 heads
    const int64_t nb2 = row_bytes<type>();                           // one head
    const int64_t nb3 = nb1 * kv_size;
    bool wide = false;
    if constexpr (fuse_enabled<ncols>() && !pos) {
        wide = wide_ok<type>(k, nb1, nb2, nb3) && wide_ok<type>(v, nb1, nb2, nb3);
    }
    const Inst & in = pos ? g_pos : wide ? inst<ncols1, type, true>() : inst<ncols1, type, false>();
    GGML_ASSERT(in.kernel != nullptr && "fa_init() before the first launch");
    const int occ = n_tokens > REF_TOKENS ? g_twin_occ : pos ? inst<ncols1, type, false>().occ : in.occ;

    // the grid: stream-K over (KV tiles x output tiles), tiles counted exactly as the kernel packs them
    const int gqa_ratio    = n_head / N_HEAD_KV;
    const int pd           = pair_d(ncols1, gqa_ratio, n_tokens);
    const int tile_heads   = pd != NCOLS2 ? pd : NCOLS2;
    const int tile_tokens  = ncols / tile_heads;
    const int ntiles_x     = (n_tokens + tile_tokens - 1) / tile_tokens;
    const int ntiles_z_gqa = (gqa_ratio + tile_heads - 1) / tile_heads;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * N_HEAD_KV;
    const int ntiles_KV    = (n_kv + cfg.nbatch_fa - 1) / cfg.nbatch_fa;

    const int max_blocks = occ * g_nsm;
    // on Ada stream-K always; rounded down to a multiple of the output tiles when that loses <= 5% (no fixup then)
    const int nblocks_raw     = std::min(max_blocks, ntiles_KV*ntiles_dst);
    const int nblocks_rounded = (nblocks_raw / ntiles_dst) * ntiles_dst;
    const int loss_percent    = nblocks_rounded > 0 ? 100 * (nblocks_raw - nblocks_rounded) / nblocks_raw : 100;
    const int nblocks         = loss_percent <= 5 ? nblocks_rounded : nblocks_raw;

    float2 * meta = nullptr;
    if (ntiles_dst % nblocks != 0) {
        GGML_ASSERT((size_t) nblocks * ncols * (2 + D/2) <= g_meta_cap);
        meta = g_meta;
    }

    const uint3 ne01 = init_fastdiv_values((uint32_t) n_tokens);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) nblocks, 1, 1), dim3(WARP_SIZE, nwarps, 1), in.smem, st);
    auto kernel = k_fa<ncols1, type, pos ? (int) F_POS_MASK : 0>;
    if constexpr (fuse_enabled<ncols>() && !pos) {
        if (wide) {
            kernel = k_fa<ncols1, type, F_WIDE | F_FAST_SEL>;
        }
    }
    // the mask row stride in bytes; the positional vector: -n_tokens halves (kpos starts n_tokens floats after qpos)
    const int32_t nb31 = pos ? (int32_t) -(n_tokens*(int64_t) sizeof(half)) : (int32_t) (n_kv*sizeof(half));
    ggml_cuda_kernel_launch(kernel, lp, (const char *) q, (const char *) k, (const char *) v, (const char *) mask, dst, meta,
        scale, ne01, (int32_t) n_head, (int32_t) (D*n_head*sizeof(float)), (int32_t) (D*sizeof(float)),
        (int32_t) n_kv, (int32_t) N_HEAD_KV, (int32_t) nb1, (int32_t) nb2, (int32_t) nb1, (int32_t) nb2, nb31, vis_pos);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        // uniform: every output tile has the same number of blocks, one fixup block per tile
        const int bpt = nblocks / ntiles_dst;
        const uint3 fd0 = init_fastdiv_values((uint32_t) (ntiles_x * ntiles_z_gqa * N_HEAD_KV));
        const uint3 fd1 = init_fastdiv_values((uint32_t) (ntiles_x * ntiles_z_gqa));
        const uint3 fd2 = init_fastdiv_values((uint32_t) ntiles_x);
        const ggml_cuda_kernel_launch_params lpf(dim3((unsigned) ntiles_dst, (unsigned) tile_tokens, (unsigned) tile_heads),
                                                 dim3(D, 1, 1), 0, st);
        ggml_cuda_kernel_launch(k_fixup_uniform<ncols1>, lpf, dst, (const float2 *) meta, n_tokens, n_head, nblocks,
            gqa_ratio, bpt, fd0, fd1, fd2, tile_tokens, tile_heads);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;
        const uint3 fd_k_j_z_ne12 = init_fastdiv_values((uint32_t) (ntiles_KV * ntiles_x * ntiles_z_gqa * N_HEAD_KV));
        const uint3 fd_k_j_z      = init_fastdiv_values((uint32_t) (ntiles_KV * ntiles_x * ntiles_z_gqa));
        const uint3 fd_k_j        = init_fastdiv_values((uint32_t) (ntiles_KV * ntiles_x));
        const uint3 fd_k          = init_fastdiv_values((uint32_t) ntiles_KV);
        const ggml_cuda_kernel_launch_params lpf(dim3((unsigned) nblocks, (unsigned) tile_tokens, (unsigned) tile_heads),
                                                 dim3(D, 1, 1), 0, st);
        ggml_cuda_kernel_launch(k_fixup_general<ncols1>, lpf, dst, (const float2 *) meta, n_tokens, n_head, gqa_ratio,
            total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k, tile_tokens, tile_heads);
    }
}

} // namespace

void fa_init() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev));
    init_inst<1, GGML_TYPE_XYZKV2_0, false>();
    init_inst<2, GGML_TYPE_XYZKV2_0, false>();
    init_inst<4, GGML_TYPE_XYZKV2_0, false>();
    init_inst<4, GGML_TYPE_XYZKV2_0, true>();
    init_inst<8, GGML_TYPE_XYZKV2_0, false>();
    init_inst<2, GGML_TYPE_Q4_0, false>();
    init_inst<4, GGML_TYPE_Q4_0, false>();
    init_inst<4, GGML_TYPE_Q4_0, true>();
    init_inst<8, GGML_TYPE_Q4_0, false>();
    {   // the prompt path's positional-mask instance: the plain 64-column tile's shared memory; its grid is sized from the
        // plain instance's occupancy or the f16 twin's (launch), so the fixup buffer covers the larger
        const auto kernel = k_fa<8, GGML_TYPE_XYZKV2_0, F_POS_MASK>;
        g_pos.kernel = (const void *) kernel;
        g_pos.smem   = smem_bytes<8, GGML_TYPE_XYZKV2_0, false>();
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) g_pos.smem));
        g_twin_occ = twin_occupancy();
        GGML_ASSERT(g_twin_occ > 0);
        const int occ = std::max(g_twin_occ, inst<8, GGML_TYPE_XYZKV2_0, false>().occ);
        g_meta_cap = std::max(g_meta_cap, (size_t) occ * g_nsm * (8*NCOLS2) * (2 + D/2));
    }
    CUDA_CHECK(cudaMalloc(&g_meta, g_meta_cap * sizeof(float2)));
}

int fa_twin_occupancy() {
    return g_twin_occ;
}

void flash_attn(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                int n_kv, int kv_size, const half * mask, float scale, float * dst, const int * vis_pos) {
    constexpr ggml_type T2 = GGML_TYPE_XYZKV2_0;
    const int d32 = pair_d(4, n_head / N_HEAD_KV, n_tokens);
    if (n_tokens <= 1) {
        launch<1, T2>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst, vis_pos);
    } else if (n_tokens <= 2) {
        launch<2, T2>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst, vis_pos);
    } else if (n_tokens <= 32/d32) {
        launch<4, T2>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst, vis_pos);
    } else if (n_tokens <= REF_TOKENS) {
        launch<8, T2>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst, vis_pos);
    } else {
        fprintf(stderr, "flash_attn: %d rows with the f16 mask (the prompt path takes the positional one)\n", n_tokens);
        abort();
    }
}

void flash_attn_prefill(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                        int n_kv, int kv_size, const float * mask_pos, float scale, float * dst) {
    if (n_tokens <= 16) {   // llama_kv_cache::kq_mask_pos_ok: the positional mask is the prompt path's only
        fprintf(stderr, "flash_attn_prefill: %d rows take the f16 mask, not the positional one\n", n_tokens);
        abort();
    }
    launch<8, GGML_TYPE_XYZKV2_0, true>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask_pos, scale, dst);
}

void flash_attn_q4_0(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                     int n_kv, int kv_size, const half * mask, float scale, float * dst) {
    constexpr ggml_type Q4 = GGML_TYPE_Q4_0;
    const int d32 = pair_d(4, n_head / N_HEAD_KV, n_tokens);
    if (n_tokens <= 2) {
        launch<2, Q4>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
    } else if (n_tokens <= 32/d32) {
        launch<4, Q4>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
    } else {
        // to 32 rows fattn.cu runs this tile natively; above, it converts the cache to f16 and runs the f16 kernel's
        // 64-column tile -- the same tile and arithmetic on the same values (this decode is convert.cu's
        // dequantize_block_q4_0_f16 bit for bit, fattn-xyzkv-tiles.cuh), its grid from the f16 kernel's occupancy (launch)
        launch<8, Q4>(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
    }
}

void flash_attn_q4_0_prompt(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache,
                            const void * v_cache, int n_kv, int kv_size, const half * mask, float scale, float * dst) {
    flash_attn_q4_0(st, q, n_tokens, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
}

} // namespace eng
