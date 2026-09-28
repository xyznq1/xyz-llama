#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-swizzle.cuh"
#include "fattn-xyzkv-tiles.cuh"

using namespace ggml_cuda_mma;

// Pair packing maps a tile column to (token, head) using the active GQA ratio.
template <ggml_type type_K, ggml_type type_V, int ncols1, int ncols2, int DKQ, int DV>
static constexpr __host__ __device__ bool fattn_pair_capable() {
    return flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>() && ncols1*ncols2 >= 32;
}

// The column group size d of a tile: the real query heads per KV head when fewer than the ncols2 slots of a
// pair-capable tile, else the compile-time ncols2. Larger batches use the f16-compatible mapping.
static constexpr int fattn_xyzkv2_ref_tokens = 32;
template <ggml_type type_K, ggml_type type_V, int ncols1, int ncols2, int DKQ, int DV>
static __host__ __device__ __forceinline__ int fattn_pair_d(const int gqa_ratio, const int n_tokens) {
    if constexpr (fattn_pair_capable<type_K, type_V, ncols1, ncols2, DKQ, DV>()) {
        return n_tokens <= fattn_xyzkv2_ref_tokens && gqa_ratio > 1 && gqa_ratio < ncols2 ? gqa_ratio : ncols2;
    } else {
        GGML_UNUSED(gqa_ratio);
        GGML_UNUSED(n_tokens);
        return ncols2;
    }
}

// Rows of the mask tile = tokens per tile: ncols1, or up to ncols/2 with pair packing (d >= 2).
template <ggml_type type_K, ggml_type type_V, int ncols1, int ncols2, int DKQ, int DV>
static constexpr __host__ __device__ int fattn_mask_rows() {
    return fattn_pair_capable<type_K, type_V, ncols1, ncols2, DKQ, DV>() ? (ncols1*ncols2)/2 : ncols1;
}

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  64,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  64,  96,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2,  32, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 1,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 256, 4,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 256, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 256, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  64, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  64, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_mfma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const bool use_sparse) {
#ifdef CP_ASYNC_AVAILABLE
    const int nstages_target = ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
    // sparse gather is not implemented for multi-stage loading
    return use_sparse && nstages_target > 1 ? 1 : nstages_target;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, use_sparse);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        static_assert(!oob_check || use_sparse, "OOB check not compatible with cp_async");
        constexpr int preload = 64;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                int64_t i_KV;
                if constexpr (use_sparse) {
                    // padded slots gather row 0, the -inf mask removes their contribution
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : 0;
                    i_KV = index >= 0 ? index : 0;
                } else {
                    i_KV = k_VKQ_0 + i;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    if constexpr (swz) {
                        const int smem_offs_b = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk);
                        cp_async_cg_16<preload>(tile_KV_32 + smem_offs_b, KV + i_KV*stride_KV + k*h2_per_chunk);
                    } else {
                        cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i_KV*stride_KV + k*h2_per_chunk);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const half2 * src;
                    if constexpr (use_sparse) {
                        const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                        src = index >= 0 ? KV + int64_t(index)*stride_KV + k*h2_per_chunk : zero;
                    } else {
                        src = !oob_check || i < i_sup ? KV + int64_t(k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero;
                    }
                    if constexpr (swz) {
                        ggml_cuda_memcpy_1<16>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk), src);
                    } else {
                        ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4, src);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    }
}

// The positional-mask variant uses the F32 position vector directly.
// mask_h points at qpos (n_q floats), kpos follows, and stride_mask carries -n_q. The tile gets exactly the values the
// table would have held (+0 / -inf; 0 past i_sup, as the table path writes there), so everything downstream of the tile
// -- the mask add, the softmax -- runs unchanged. Plain shared stores instead of cp.async: every consumer of a mask tile
// sits behind a __syncthreads (the ring waits for its group, then syncs), which publishes them as it publishes the copies.
template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse, bool pos_mask = false>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,
        const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    if constexpr (pos_mask) {
        static_assert(!use_sparse, "the positional mask has no sparse gather");
        const float * const qpos = (const float *) mask_h;
        const float * const kpos = qpos + (-stride_mask);
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }
            const float qp = qpos[fastmodulo(j0 + j_sram, ne01)];
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                if (i0 + warp_size > nbatch_fa && i >= nbatch_fa) {
                    break;
                }
                tile_mask[j_sram*(nbatch_fa + 8) + i] = oob_check && i >= i_sup ? half(0.0f) :
                    (kpos[k_VKQ_0 + i] < qp ? half(0.0f) : half(-INFINITY));
            }
        }
        GGML_UNUSED(indices);
        return;
    }
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*warp_size && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        static_assert(!use_sparse, "sparse gather incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (oob_check || use_sparse) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = index >= 0 ? mask_h[int64_t(j_vram)*stride_mask + index] : half(-INFINITY);
                } else {
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                }
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

// The caller commits one copy group for each half of a staged KV slot.
template <ggml_type type_K, ggml_type type_V, int DKQ, int DV, int nbatch_fa, int nwarps, int mask_rows, bool wide = false>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_ring_issue_K(
        char * const stage, const int kb, const half2 * const __restrict__ K_h2, const int stride_K) {
    constexpr int nthreads = nwarps * ggml_cuda_get_physical_warp_size();
    char * const slot = stage + (kb % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, wide>(nbatch_fa, mask_rows);
    const int stride_K_bytes = stride_K * (int) sizeof(half2);
    const char * src = (const char *) K_h2 + (int64_t) kb*nbatch_fa*stride_K_bytes;
    if constexpr (wide) {
        flash_attn_ext_xyzkv2_stage_issue16<type_K, DKQ, nbatch_fa, nthreads>(src - ((uintptr_t) src & 15), slot, stride_K_bytes);
    } else {
        flash_attn_ext_xyzkv2_stage_issue<type_K, DKQ, nbatch_fa, nthreads, false>(src, slot, stride_K_bytes, nbatch_fa);
    }
}

template <ggml_type type_K, ggml_type type_V, int DKQ, int DV, int nbatch_fa, int nwarps, int mask_rows, bool wide = false,
          bool pos_mask = false>
static __device__ __forceinline__ void flash_attn_ext_xyzkv2_ring_issue_V(
        char * const stage, const int kb, const half2 * const __restrict__ V_h2, const int stride_V,
        const half * const __restrict__ mask_h, const int stride_mask, const int j0, const uint3 ne01) {
    constexpr int nthreads = nwarps * ggml_cuda_get_physical_warp_size();
    constexpr int rowK     = flash_attn_ext_xyzkv2_stage_pitch<type_K, DKQ, wide>();
    constexpr int rowV     = flash_attn_ext_xyzkv2_stage_pitch<type_V, DV, wide>();
    char * const slot = stage + (kb % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, wide>(nbatch_fa, mask_rows);
    const int stride_V_bytes = stride_V * (int) sizeof(half2);
    const char * src = (const char *) V_h2 + (int64_t) kb*nbatch_fa*stride_V_bytes;
    if constexpr (wide) {
        flash_attn_ext_xyzkv2_stage_issue16<type_V, DV, nbatch_fa, nthreads>(src - ((uintptr_t) src & 15), slot + nbatch_fa*rowK, stride_V_bytes);
    } else {
        flash_attn_ext_xyzkv2_stage_issue<type_V, DV, nbatch_fa, nthreads, false>(src, slot + nbatch_fa*rowK, stride_V_bytes, nbatch_fa);
    }
    flash_attn_ext_f16_load_mask<mask_rows, nwarps, nbatch_fa, true, false, false, pos_mask>
        (mask_h, (half *) (slot + nbatch_fa*(rowK + rowV)), stride_mask, kb*nbatch_fa, nbatch_fa, j0, ne01, nullptr);
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16,
    int variant = 0>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int kb0_stop,
        const int k_VKQ_sup,
        const int d,
        const int tpt) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    // Keep this stage count equal to the host shared-memory layout.
    constexpr bool is_xyzkv_kv_    = (type_K != GGML_TYPE_F16) || (type_V != GGML_TYPE_F16);
    constexpr int  nstages         = is_xyzkv_kv_ ? 0 :
                                     ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2, use_sparse);

    // swizzle the tile stride for K and V based on the batch size.
    constexpr int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2);
    constexpr bool swz_K = ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    const int k_VKQ_0 = kb0 * nbatch_fa;
    constexpr int  mask_rows = fattn_mask_rows<type_K, type_V, ncols1, ncols2, DKQ, DV>();
#ifdef CP_ASYNC_AVAILABLE
    constexpr bool ring      = flash_attn_ext_xyzkv2_ring_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>() && !oob_check;
#else
    constexpr bool ring      = false;
#endif // CP_ASYNC_AVAILABLE
    // The fused loop decodes V(t) under KQ(t) and K(t+1) under PV(t).
    constexpr bool fuse      = ring && flash_attn_ext_xyzkv2_fuse_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>();
    static_assert(!fuse || (Q_in_reg && cols_per_warp != 8), "the fused loop threads its decode through the wide Q_in_reg KQ loop");
    constexpr bool wide = fuse && (variant & 16) != 0;
    constexpr int PK = flash_attn_ext_xyzkv2_stage_pitch<type_K, DKQ, wide>();
    constexpr int PV = flash_attn_ext_xyzkv2_stage_pitch<type_V, DV, wide>();
    [[maybe_unused]] const int oK = wide ? (int) ((uintptr_t) K_h2 & 15) : 0;
    [[maybe_unused]] const int oV = wide ? (int) ((uintptr_t) V_h2 & 15) : 0;
    constexpr bool fast_sel = wide && (variant & 32) != 0;
    constexpr bool pos_mask = (variant & 512) != 0;
    [[maybe_unused]] constexpr int nthreads_fuse = nwarps * ggml_cuda_get_physical_warp_size();
    [[maybe_unused]] uint32_t lut_fuse[2];   // the decode unit's centroid LUT, carried between units of one item
    [[maybe_unused]] const char * V_stage_fuse = nullptr;   // tile kb0's raw V rows in its ring slot
    [[maybe_unused]] flash_attn_ext_xyzkv2_decode_regs<DV,  nbatch_fa, nthreads_fuse> V_regs_fuse;
    [[maybe_unused]] flash_attn_ext_xyzkv2_decode_regs<DKQ, nbatch_fa, nthreads_fuse> K_regs_fuse;
    half * tile_mask_it = tile_mask;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_0, k_VKQ_sup, nullptr);
    } else {
        // the sparse mask values are gathered per element, always load them synchronously
        constexpr bool use_cp_async = nstages == 1 && !use_sparse;
        if (!ring && (ncols2 > 1 || mask_h)) {
            flash_attn_ext_f16_load_mask<mask_rows, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse, pos_mask>
                (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*tpt, ne01, indices);
        }
    }

    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        if constexpr (flash_attn_ext_native_kv<type_K>()) {   // xyzkv2 or q4_0 (fattn-xyzkv-tiles.cuh)
            // Read native quantized K bytes directly into the tile.
            //
            // The loader fills a FULL row of width DKQ, so there must be exactly one k0 iteration.
            // With K batching the f16 path would load columns [k0_start, k0_stop) per pass; feeding
            // that a full-row loader would write past the slice and silently corrupt the tile.
            static_assert(nbatch_K2 == DKQ/2, "xyzkv2 K tile loader requires unbatched K (nbatch_K2 == DKQ/2)");
            static_assert(nstages == 0, "xyzkv2 K tile loader cannot be driven by cp_async (it needs ALU dequant)");
            static_assert(!use_sparse, "xyzkv2 K tile loader does not implement the sparse gather");

            // stride_K is nb11 expressed in half2 units, so this recovers the row stride in bytes
            // exactly -- nb11 for a xyzkv2 row is a multiple of 4 (DKQ/128 blocks of 34 bytes).
            const int stride_K_bytes = stride_K * (int) sizeof(half2);
            constexpr int nthreads_tile = nwarps * ggml_cuda_get_physical_warp_size();

            // Staged (fattn-xyzkv-tiles.cuh): this tile's raw K and V rows were issued one iteration ago (or by the
            // process_tile prologue); land them here -- the ONE wait per iteration -- dequantise K from the stage,
            // then put the next tile's raw K in flight while KQ, the softmax and PV run.
            constexpr bool xyzkv_stage = flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>() && !oob_check;
            const char * K_raw   = (const char *) K_h2 + (int64_t) k_VKQ_0 * stride_K_bytes;
            int          K_pitch = stride_K_bytes;
#ifdef CP_ASYNC_AVAILABLE
            constexpr int stage_off = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), mask_rows);
            char * const raw_K = (char *) tile_Q + stage_off;   // mirrors process_tile and the host
            if constexpr (ring) {
                if constexpr (!fuse) {
                    // Land tile kb0 while newer copy groups stay in flight.
                    flash_attn_ext_xyzkv2_cp_async_wait_group<2*FATTN_XYZKV2_RING - 2>();
                    __syncthreads();
                }
                char * const slot = raw_K + (kb0 % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, wide>(nbatch_fa, mask_rows);
                K_raw        = slot + oK;
                K_pitch      = PK;
                V_stage_fuse = slot + nbatch_fa*PK + oV;
                if constexpr (fuse) {
                    flash_attn_ext_xyzkv2_decode_load<type_V, DV, nbatch_fa, nthreads_fuse, PV>(V_stage_fuse, V_regs_fuse);
                }
                tile_mask_it = (half *) (slot + nbatch_fa*(PK + PV));
            } else if constexpr (xyzkv_stage) {
                flash_attn_ext_xyzkv2_cp_async_wait_all();
                __syncthreads();
                K_raw   = raw_K;
                K_pitch = flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>();
            }
#endif // CP_ASYNC_AVAILABLE
            if constexpr (!fuse) {
                flash_attn_ext_xyzkv2_load_tile<type_K, DKQ, stride_tile_K, nbatch_fa, nthreads_tile, oob_check, swz_K, (ncols < 64)>
                    (K_raw, tile_K, K_pitch, k_VKQ_sup);   // valid rows of THIS tile (k_VKQ_sup is tile-relative)
                __syncthreads();
            }
#ifdef CP_ASYNC_AVAILABLE
            if constexpr (ring) {
                if constexpr (!fuse) {
                    // Refill the consumed K half. Every iteration commits the same group count.
                    if (kb0 + FATTN_XYZKV2_RING < kb0_stop) {
                        flash_attn_ext_xyzkv2_ring_issue_K<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows>(raw_K, kb0 + FATTN_XYZKV2_RING, K_h2, stride_K);
                    }
                    flash_attn_ext_xyzkv2_cp_async_commit();
                }
            } else if constexpr (xyzkv_stage && !last_iter) {
                flash_attn_ext_xyzkv2_stage_issue<type_K, DKQ, nbatch_fa, nthreads_tile, oob_check>
                    ((const char *) K_h2 + (int64_t) (k_VKQ_0 + nbatch_fa) * stride_K_bytes, raw_K, stride_K_bytes, nbatch_fa);
            }
#endif // CP_ASYNC_AVAILABLE
        } else if constexpr (nstages <= 1) {
            const int k0_diff = k0_stop - k0_start;
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
#ifdef CP_ASYNC_AVAILABLE
                    if constexpr (fuse) {
                        // Decode one unit of V(kb0) between KQ steps.
                        constexpr int ksteps = (DKQ/2)/T_A_KQ::J;
                        constexpr int steps  = (nbatch_fa/(np*T_A_KQ::I)) * ksteps;
                        constexpr int units  = flash_attn_ext_xyzkv2_decode_units<DV, nbatch_fa, nthreads_fuse>();
                        static_assert(steps % units == 0, "the V decode units must spread evenly over the KQ steps");
                        const int step = (i_KQ_00/(np*T_A_KQ::I))*ksteps + (k_KQ_0 - k0_start)/T_A_KQ::J;
                        if (step % (steps/units) == 0) {
                            flash_attn_ext_xyzkv2_decode_unit_regs<type_V, DV, stride_tile_V, nbatch_fa, nthreads_fuse, swz_V, fast_sel>(
                                V_regs_fuse, tile_V, step/(steps/units), lut_fuse);
                        }
                    }
#endif // CP_ASYNC_AVAILABLE
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (fuse) {
#ifdef CP_ASYNC_AVAILABLE
            // Publish V(kb0) and K(kb0+1) after all warps finish reading tile_K.
            flash_attn_ext_xyzkv2_cp_async_wait_group<(FATTN_XYZKV2_RING >= 2 ? FATTN_XYZKV2_RING - 2 : 0)>();
#endif // CP_ASYNC_AVAILABLE
            __syncthreads();
        } else if constexpr (flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>() == 2) {
            // Decode V while KQ remains in the tensor pipe. The buffers do not alias.
            constexpr int nthreads_tile  = nwarps * ggml_cuda_get_physical_warp_size();
            const int     stride_V_bytes = stride_V * (int) sizeof(half2);
            const char *  V_raw   = (const char *) V_h2 + (int64_t) k_VKQ_0 * stride_V_bytes;
            int           V_pitch = stride_V_bytes;
#ifdef CP_ASYNC_AVAILABLE
            constexpr bool xyzkv_stage = flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>() && !oob_check;
            constexpr int  stage_off   = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), mask_rows);
            if constexpr (ring) {
                V_raw   = (const char *) tile_Q + stage_off + nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>()
                        + (kb0 % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV>(nbatch_fa, mask_rows);
                V_pitch = flash_attn_ext_xyzkv2_row_bytes<type_V, DV>();
            } else if constexpr (xyzkv_stage) {
                V_raw   = (const char *) tile_Q + stage_off + nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>();
                V_pitch = flash_attn_ext_xyzkv2_row_bytes<type_V, DV>();
            }
#endif // CP_ASYNC_AVAILABLE
            flash_attn_ext_xyzkv2_load_tile<type_V, DV, stride_tile_V, nbatch_fa, nthreads_tile, oob_check, swz_V, (ncols < 64)>
                (V_raw, tile_V, V_pitch, k_VKQ_sup);
        } else if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
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

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        const bool mask_add = ncols2 > 1 || mask_h;
        if (mask_add) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / d;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask_it[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / d;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask_it)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
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

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        // Skip the rescale when every scale is 1.0.
        bool rescale = false;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            rescale |= KQ_max_scale[col] != 1.0f;
        }
        if (ncols == 32 || __any_sync(0xFFFFFFFF, rescale))
#endif // defined(TURING_MMA_AVAILABLE)
#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
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
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
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
    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<mask_rows, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse, pos_mask>
                    (mask_h, tile_mask, stride_mask, k_VKQ_0 + nbatch_fa, k_VKQ_sup, jt*tpt, ne01, nullptr);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_0 + nbatch_fa, k_VKQ_sup, nullptr);
        }
    }


    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        if constexpr (flash_attn_ext_native_kv<type_V>()) {
            // Native xyzkv2 V, same reasoning as the K side above. V_is_K_view cannot apply here:
            // that optimisation reuses already-loaded K tile data, which only holds for MLA-style
            // caches where K and V are the same tensor, never for a separately quantized V.
            static_assert(2*nbatch_V2 == DV, "xyzkv2 V tile loader requires unbatched V (2*nbatch_V2 == DV)");
            static_assert(nstages == 0, "xyzkv2 V tile loader cannot be driven by cp_async (it needs ALU dequant)");
            static_assert(!V_is_K_view, "xyzkv2 V tile loader is incompatible with K/V data reuse");
            static_assert(!use_sparse, "xyzkv2 V tile loader does not implement the sparse gather");

            const int stride_V_bytes = stride_V * (int) sizeof(half2);
            constexpr int nthreads_tile = nwarps * ggml_cuda_get_physical_warp_size();

            // Staged: this tile's raw V rows landed with the K rows (the wait in the K phase above); dequantise from
            // the stage, then put the next tile's raw V in flight while PV and the next mask load run.
            constexpr bool xyzkv_stage = flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>() && !oob_check;
            const char * V_raw   = (const char *) V_h2 + (int64_t) k_VKQ_0 * stride_V_bytes;
            int          V_pitch = stride_V_bytes;
#ifdef CP_ASYNC_AVAILABLE
            constexpr int stage_off = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), mask_rows);
            char * const raw_V = (char *) tile_Q + stage_off + nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>();
            if constexpr (ring) {
                V_raw   = (const char *) tile_Q + stage_off + nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>()
                        + (kb0 % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV>(nbatch_fa, mask_rows);
                V_pitch = flash_attn_ext_xyzkv2_row_bytes<type_V, DV>();
            } else if constexpr (xyzkv_stage) {
                V_raw   = raw_V;
                V_pitch = flash_attn_ext_xyzkv2_row_bytes<type_V, DV>();
            }
#endif // CP_ASYNC_AVAILABLE
            if constexpr (flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>() == 1) {
                flash_attn_ext_xyzkv2_load_tile<type_V, DV, stride_tile_V, nbatch_fa, nthreads_tile, oob_check, swz_V, (ncols < 64)>
                    (V_raw, tile_V, V_pitch, k_VKQ_sup);   // valid rows of THIS tile (k_VKQ_sup is tile-relative)
            }
            if constexpr (!fuse) {
                __syncthreads();
            }
#ifdef CP_ASYNC_AVAILABLE
            if constexpr (ring) {
                if constexpr (!fuse) {
                    // Refill the consumed V half and mask.
                    if (kb0 + FATTN_XYZKV2_RING < kb0_stop) {
                        flash_attn_ext_xyzkv2_ring_issue_V<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, false, pos_mask>((char *) tile_Q + stage_off,
                            kb0 + FATTN_XYZKV2_RING, V_h2, stride_V, mask_h, stride_mask, jt*tpt, ne01);
                    }
                    flash_attn_ext_xyzkv2_cp_async_commit();
                }
            } else if constexpr (xyzkv_stage && !last_iter) {
                flash_attn_ext_xyzkv2_stage_issue<type_V, DV, nbatch_fa, nthreads_tile, oob_check>
                    ((const char *) V_h2 + (int64_t) (k_VKQ_0 + nbatch_fa) * stride_V_bytes, raw_V, stride_V_bytes, nbatch_fa);
            }
#endif // CP_ASYNC_AVAILABLE
        } else if constexpr (nstages <= 1) {
            const int i0_diff = i0_stop - i0_start;
            if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                constexpr bool use_cp_async = nstages == 1;
                flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_0, k_VKQ_sup, indices);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }
        }
        const half2 * tile_V_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? tile_V : tile_V + i0_start/2;
#ifdef CP_ASYNC_AVAILABLE
        if constexpr (fuse && !last_iter) {
            constexpr int stage_off_fuse = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), mask_rows);
            flash_attn_ext_xyzkv2_decode_load<type_K, DKQ, nbatch_fa, nthreads_fuse, PK>((const char *) tile_Q + stage_off_fuse
                + ((kb0 + 1) % FATTN_XYZKV2_RING) * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, wide>(nbatch_fa, mask_rows) + oK, K_regs_fuse);
        }
#endif // CP_ASYNC_AVAILABLE

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix_trans<stride_tile_V, swz_V>(A, tile_V, (int)(tile_V_i - tile_V) + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
#ifdef CP_ASYNC_AVAILABLE
                if constexpr (fuse && !last_iter) {
                    // Decode one unit of K(kb0+1) between PV steps.
                    constexpr int ksteps = (nbatch_fa/2)/(np*T_A_VKQ::J);
                    constexpr int steps  = ((2*nbatch_V2)/T_A_VKQ::I) * ksteps;   // xyzkv2: one i0 pass of 2*nbatch_V2 == DV
                    constexpr int units  = flash_attn_ext_xyzkv2_decode_units<DKQ, nbatch_fa, nthreads_fuse>();
                    static_assert(steps % units == 0, "the K decode units must spread evenly over the PV steps");
                    const int step = ((i_VKQ_0 - i0_start)/T_A_VKQ::I)*ksteps + k00/(np*T_A_VKQ::J);
                    if (step % (steps/units) == 0) {
                        flash_attn_ext_xyzkv2_decode_unit_regs<type_K, DKQ, stride_tile_K, nbatch_fa, nthreads_fuse, swz_K, fast_sel>(
                            K_regs_fuse, tile_K, step/(steps/units), lut_fuse);
                    }
                }
#endif // CP_ASYNC_AVAILABLE
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_V, swz_V>(A, tile_V, (int)(tile_V_i - tile_V) + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if constexpr (nstages <= 1 && flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>() == 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
#ifdef CP_ASYNC_AVAILABLE
        if constexpr (fuse) {
            __syncthreads();
            constexpr int stage_off_fuse = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*(stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V), mask_rows);
            if (kb0 + FATTN_XYZKV2_RING < kb0_stop) {
                flash_attn_ext_xyzkv2_ring_issue_K<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, wide>((char *) tile_Q + stage_off_fuse,
                    kb0 + FATTN_XYZKV2_RING, K_h2, stride_K);
                flash_attn_ext_xyzkv2_ring_issue_V<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, wide, pos_mask>((char *) tile_Q + stage_off_fuse,
                    kb0 + FATTN_XYZKV2_RING, V_h2, stride_V, mask_h, stride_mask, jt*tpt, ne01);
            }
            flash_attn_ext_xyzkv2_cp_async_commit();
        }
#endif // CP_ASYNC_AVAILABLE
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int DV> struct mma_tile_sizes<DV, 8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#else
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#else // Volta
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16,
    int variant = 0>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
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
        const int tpt) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  mask_rows       = fattn_mask_rows<type_K, type_V, ncols1, ncols2, DKQ, DV>();

    // Quantized cache rows require the synchronous ALU loader, so their stage count is zero.
    constexpr bool is_xyzkv_kv_    = (type_K != GGML_TYPE_F16) || (type_V != GGML_TYPE_F16);
    constexpr int  nstages         = is_xyzkv_kv_ ? 0 :
                                     ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, use_sparse);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    // swizzle the tile stride for K and V based on the batch size.
    constexpr int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2);
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;
    constexpr bool swz_K = ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    // tile_V gets its own buffer for the multi-stage f16 path and native overlap.
    constexpr bool sep_V = nstages > 1 || flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>() == 2;
    half2 * tile_V    =           sep_V ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (sep_V ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    T_C_VKQ VKQ_C[DV % 32 != 0       ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            // Column jc is a (token, head) pair with group size d; the last
            // ncols - tpt*d columns of a pair-packed tile are dead and zero-filled like padding.
            const int j = jc / d;
            const int c = jc % d;

            if (jc < tpt*d && (ncols1 == 1 || jt*tpt + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*d + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*tpt + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    constexpr bool pos_mask = (variant & 512) != 0;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        // cp_async only accepts the unquantized row layout.
        static_assert(!is_xyzkv_kv_, "quantized KV cannot be preloaded with cp_async (nstages must be 0)");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<mask_rows, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse, pos_mask>
                (mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*tpt, ne01, nullptr);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (K_h2, tile_K, nbatch_K2, stride_K, kb0*nbatch_fa, k_VKQ_sup, nullptr);
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (ncols2 == 1 || use_sparse) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt);
    } else {
        constexpr bool oob_check = false;
#ifdef CP_ASYNC_AVAILABLE
        if constexpr (flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>()) {
            // Staged xyzkv path (see flash_attn_ext_f16_iter): put the first tile's raw K and V rows in flight before
            // the loop; the first iteration lands them. Offsets mirror the iter and the host.
            constexpr int nthreads_tile = nwarps * ggml_cuda_get_physical_warp_size();
            constexpr int stage_off     = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols>()*stride_tile_KV_max, mask_rows);
            char * const raw_K = (char *) tile_Q + stage_off;
            static_assert((nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>()) % 16 == 0, "raw_V must stay 16-byte aligned");
            if constexpr (flash_attn_ext_xyzkv2_fuse_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>()) {
                // Start the ring, land tile kb0, and decode K(kb0+1) under the current tile's PV work.
                constexpr bool wide = (variant & 16) != 0;
                constexpr int slot_bytes = flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, wide>(nbatch_fa, mask_rows);
                const int oK = wide ? (int) ((uintptr_t) K_h2 & 15) : 0;
#pragma unroll
                for (int s = 0; s < FATTN_XYZKV2_RING; ++s) {
                    if (kb0 + s < kb0_stop) {
                        flash_attn_ext_xyzkv2_ring_issue_K<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, wide>(raw_K, kb0 + s, K_h2, stride_K);
                        flash_attn_ext_xyzkv2_ring_issue_V<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, wide, pos_mask>(raw_K, kb0 + s, V_h2, stride_V,
                            mask_h, stride_mask, jt*tpt, ne01);
                    }
                    flash_attn_ext_xyzkv2_cp_async_commit();
                }
                flash_attn_ext_xyzkv2_cp_async_wait_group<FATTN_XYZKV2_RING - 1>();
                __syncthreads();
                flash_attn_ext_xyzkv2_load_tile<type_K, DKQ, stride_tile_K, nbatch_fa, nthreads_tile, false, swz_K, (ncols < 64)>
                    (raw_K + (kb0 % FATTN_XYZKV2_RING)*slot_bytes + oK, tile_K, flash_attn_ext_xyzkv2_stage_pitch<type_K, DKQ, wide>(), nbatch_fa);
                __syncthreads();
            } else if constexpr (flash_attn_ext_xyzkv2_ring_enabled<type_K, type_V, ncols2, DKQ, DV, ncols>()) {
                // Start the ring with a K group and a V plus mask group per tile.
#pragma unroll
                for (int s = 0; s < FATTN_XYZKV2_RING; ++s) {
                    if (kb0 + s < kb0_stop) {
                        flash_attn_ext_xyzkv2_ring_issue_K<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows>(raw_K, kb0 + s, K_h2, stride_K);
                    }
                    flash_attn_ext_xyzkv2_cp_async_commit();
                    if (kb0 + s < kb0_stop) {
                        flash_attn_ext_xyzkv2_ring_issue_V<type_K, type_V, DKQ, DV, nbatch_fa, nwarps, mask_rows, false, pos_mask>(raw_K, kb0 + s, V_h2, stride_V,
                            mask_h, stride_mask, jt*tpt, ne01);
                    }
                    flash_attn_ext_xyzkv2_cp_async_commit();
                }
            } else {
                char * const raw_V = raw_K + nbatch_fa * flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>();
                flash_attn_ext_xyzkv2_stage_issue<type_K, DKQ, nbatch_fa, nthreads_tile, oob_check>
                    ((const char *) K_h2 + (int64_t) (kb0*nbatch_fa) * (stride_K * (int) sizeof(half2)), raw_K,
                     stride_K * (int) sizeof(half2), nbatch_fa);
                flash_attn_ext_xyzkv2_stage_issue<type_V, DV, nbatch_fa, nthreads_tile, oob_check>
                    ((const char *) V_h2 + (int64_t) (kb0*nbatch_fa) * (stride_V * (int) sizeof(half2)), raw_V,
                     stride_V * (int) sizeof(half2), nbatch_fa);
            }
        }
#endif // CP_ASYNC_AVAILABLE
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V, variant>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V, variant>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, kb0_stop, k_VKQ_sup, d, tpt);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % d];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
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
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    constexpr bool combine_needs_sync = swz_K || swz_V;

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
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
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
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
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= warp_size);
        if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / d;
                    const int c_dst = jc_dst % d;

                    if (!is_fixup && (jc_dst >= tpt*d || (ncols1 > 1 && jt*tpt + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*d + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

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
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*tpt + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        // Close the pass when another follows: the next pass overwrites the combine buffer (tile_Q) that other warps may
        // still be reading -- the write-out above takes columns round-robin across warps, not the warp's own. Only
        // multi-pass configs reuse the buffer and must synchronize before the next pass.
        if (np > 1 || k00 + nbatch_combine < DV/2) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(
        const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16,
    int variant = 0>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q              = Q_ptr;
    const char * GGML_CUDA_RESTRICT K              = K_ptr;
    const char * GGML_CUDA_RESTRICT V              = V_ptr;
    const char * GGML_CUDA_RESTRICT mask           = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks          = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max         = use_sparse ? nullptr : KV_max_ptr;
    const int  * GGML_CUDA_RESTRICT sparse_indices = use_sparse ? KV_max_ptr : nullptr;
    float      * GGML_CUDA_RESTRICT dst            = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta       = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
    if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
        NO_DEVICE_CODE;
        return;
    }

    if (!ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2) && use_sparse) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    // d is the column group size and tpt is the token count per tile.
    const int d   = fattn_pair_d<type_K, type_V, ncols1, ncols2, DKQ, DV>(gqa_ratio, (int) ne01.z);
    const int tpt = (ncols1*ncols2) / d;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / (int) sizeof(half);   // signed: the positional mask passes -n_q halves here

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (tpt       - 1)) / tpt;
    const int iter_z_gqa = (gqa_ratio + (d         - 1)) / d;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*d; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
        const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*ne31 + jt*tpt)*ne11 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V, variant>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, d, tpt);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V, variant>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, d, tpt);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*d; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
    const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*ne31 + jt*tpt)*ne11 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V, variant>
        (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, d, tpt);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Wide 16-byte windows need every row of a head at one misalignment (16-byte-aligned base,
// token and sequence strides multiples of 16) and each head's window [row - o, row - o + pitch) inside its token row -- the
// last head's binds. Rows already a 16-byte multiple (q4_0) have no room for an offset: every head must be aligned.
template <ggml_type type, int D>
static bool ggml_cuda_fattn_xyzkv2_wide_ok(const ggml_tensor * t) {
    if constexpr (!flash_attn_ext_native_kv<type>()) {
        GGML_UNUSED(t);
        return false;
    } else {
        constexpr int rb    = flash_attn_ext_xyzkv2_row_bytes<type, D>();
        constexpr int pitch = flash_attn_ext_xyzkv2_stage_pitch<type, D, true>();
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[3] % 16 != 0 || t->nb[2] % 4 != 0) {
            return false;
        }
        if (rb % 16 == 0) {
            return t->nb[2] % 16 == 0;
        }
        const int64_t start = (t->ne[2] - 1) * (int64_t) t->nb[2];
        return start - start % 16 + pitch <= (int64_t) t->nb[1];
    }
}

template <int DKQ, int DV, int ncols1, int ncols2,
          ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    // cp_async copies bytes; it cannot run the ALU dequant the xyzkv tile loaders need, so the
    // native-xyzkv instantiations must stay on the synchronous single-stage path. The kernel also
    // static_asserts this, so a mismatch is a compile error rather than a wrong tile.
    constexpr bool is_xyzkv_kv = (type_K != GGML_TYPE_F16) || (type_V != GGML_TYPE_F16);
    const int  nstages        = is_xyzkv_kv ? 0 :
                                ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    // KV tile strides must match flash_attn_ext_f16_iter / _process_tile.
    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K,  stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (stride_tile_K + stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    constexpr int mask_rows = fattn_mask_rows<type_K, type_V, ncols1, ncols2, DKQ, DV>();
    const size_t nbytes_shared_mask      = mask_rows            * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 && flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols1*ncols2>() == 1 ?
        nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);
    // Staged xyzkv tiles (fattn-xyzkv-tiles.cuh): raw_K and raw_V parked after the larger of the Q tile and the
    // KV+mask tiles. The offset formula is the one the kernel uses; a mismatch would alias the tiles silently.
    if (flash_attn_ext_xyzkv2_stage_enabled<type_K, type_V, ncols2, DKQ, DV>()) {
        const int stage_off = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                                  flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols1*ncols2>()*std::max(stride_tile_K, stride_tile_V), mask_rows);
        const size_t nbytes_shared_stage = flash_attn_ext_xyzkv2_ring_enabled<type_K, type_V, ncols2, DKQ, DV, ncols1*ncols2>() ?
            (size_t) FATTN_XYZKV2_RING * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV>(nbatch_fa, mask_rows) :
            (size_t) nbatch_fa * (flash_attn_ext_xyzkv2_row_bytes<type_K, DKQ>() + flash_attn_ext_xyzkv2_row_bytes<type_V, DV>());
        nbytes_shared_total = std::max(nbytes_shared_total, (size_t) stage_off + nbytes_shared_stage);
    }

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    bool use_sparse = false;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {
            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
                constexpr bool use_sparse_kernel = true;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, type_K, type_V>;
                use_sparse = true;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            } else {
                constexpr bool use_sparse_kernel = false;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, type_K, type_V>;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            }
        } else
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            constexpr bool use_sparse_kernel = false;
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!shared_memory_limit_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                shared_memory_limit_raised[id] = true;
            }
#endif // !defined(GGML_USE_MUSA)
        }
    } else {
        constexpr bool use_logit_softcap = true;
        constexpr bool use_sparse_kernel = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // Use the fused native tile only when its layout and mask requirements hold.
    if constexpr (flash_attn_ext_xyzkv2_fuse_enabled<type_K, type_V, ncols2, DKQ, DV, ncols1*ncols2>()) {
        const ggml_tensor * Kt = dst->src[1];
        const ggml_tensor * Vt = dst->src[2];
        if (logit_softcap == 0.0f && !use_sparse && Kt != nullptr && Vt != nullptr &&
                ggml_cuda_fattn_xyzkv2_wide_ok<type_K, DKQ>(Kt) && ggml_cuda_fattn_xyzkv2_wide_ok<type_V, DV>(Vt)) {
            const int stage_off_w = flash_attn_ext_xyzkv2_stage_off(Q_in_reg, ncols, DKQ, nbatch_fa,
                flash_attn_ext_xyzkv2_kv_bufs<type_K, type_V, ncols2, DKQ, DV, ncols1*ncols2>()*std::max(stride_tile_K, stride_tile_V), mask_rows);
            nbytes_shared_total = std::max(nbytes_shared_total, (size_t) stage_off_w +
                (size_t) FATTN_XYZKV2_RING * flash_attn_ext_xyzkv2_slot_bytes<type_K, DKQ, DV, true>(nbatch_fa, mask_rows));
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, type_K, type_V, 48>;
#if !defined(GGML_USE_MUSA)
            static bool wide_smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!wide_smem_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                wide_smem_raised[id] = true;
            }
#endif // !defined(GGML_USE_MUSA)
        }
    }

    // The 64-column positional-mask instance reads the F32 position vector instead of the f16 table.
    if (dst->src[3] != nullptr && dst->src[3]->type == GGML_TYPE_F32) {
        if constexpr (type_K == GGML_TYPE_XYZKV2_0 && type_V == GGML_TYPE_XYZKV2_0 && ncols1*ncols2 == 64 && DKQ == 256 && DV == 256) {
            GGML_ASSERT(logit_softcap == 0.0f && !use_sparse);
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, type_K, type_V, 512>;
#if !defined(GGML_USE_MUSA)
            static bool pos_smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!pos_smem_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                pos_smem_raised[id] = true;
            }
#endif // !defined(GGML_USE_MUSA)
        } else {
            GGML_ABORT("positional mask reached a %d-column tile (K %s, V %s): only the 64-column xyzkv2 tile reads it",
                       ncols1*ncols2, ggml_type_name(type_K), ggml_type_name(type_V));
        }
    }

    // need_f16_K/V follow what the KERNEL expects, not what the cache happens to be. A default
    // (F16, F16) instantiation wants f16 tiles, so launch_fattn must materialise them -- that is the
    // conversion of the whole KV cache, every pass. A native-xyzkv instantiation reads the quantized
    // bytes itself, so converting would both waste the bandwidth AND hand the kernel a buffer it is
    // about to reinterpret as xyzkv2 blocks.
    //
    // This must agree with ggml_cuda_fattn_mma_use_native_xyzkv2() in fattn.cu, which decides which
    // instantiation is dispatched. Disagreement is silent: the kernel would read f16 bytes as
    // quantized blocks, or quantized blocks as f16.
    constexpr bool need_f16_K = (type_K == GGML_TYPE_F16);
    constexpr bool need_f16_V = (type_V == GGML_TYPE_F16);

    // The grid and stream-K fixup must count tokens per tile exactly as the kernel does.
    const int pair_d = fattn_pair_d<type_K, type_V, ncols1, ncols2, DKQ, DV>((int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]),
                                                                             (int) dst->src[0]->ne[1]);

    // Larger batches use the f16 kernel's occupancy so stream-K partitions match its arithmetic order.
    int occupancy_ref = 0;
    if constexpr (type_K == GGML_TYPE_XYZKV2_0 && type_V == GGML_TYPE_XYZKV2_0) {
        if (dst->src[0]->ne[1] > fattn_xyzkv2_ref_tokens && logit_softcap == 0.0f && !use_sparse) {
            constexpr int ncols_twin = ncols1 * ncols2;
            const int  twin_nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols_twin, cc);
            const int  twin_nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols_twin, cc);
            const int  twin_nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols_twin, cc);
            const int  twin_nbatch_cb = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols_twin, cc);
            const bool twin_Q_in_reg  = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols_twin, cc);
            const int  twin_nstages   = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);
            // the (F16, F16) instance's shared memory: the formula above with no staging, no pair packing, one KV buffer
            const int  twin_stride_K  = ggml_cuda_fattn_smem_swizzle::tile_stride(twin_nbatch_K2, cc);
            const int  twin_stride_V  = V_is_K_view ? twin_stride_K : ggml_cuda_fattn_smem_swizzle::tile_stride(twin_nbatch_V2, cc);
            const size_t twin_KV_1    = twin_nbatch_fa * std::max(twin_stride_K, twin_stride_V) * sizeof(half2);
            const size_t twin_KV_2    = twin_nbatch_fa * (twin_stride_K + twin_stride_V)        * sizeof(half2);
            const size_t twin_Q       = ncols_twin     * (DKQ/2 + 4)                            * sizeof(half2);
            const size_t twin_mask    = ncols1         * (twin_nbatch_fa/2 + 4)                 * sizeof(half2);
            const size_t twin_combine = nwarps*cols_per_warp * (twin_nbatch_cb + 4)             * sizeof(half2);
            const size_t twin_KV      = twin_nstages <= 1 ? twin_KV_1 : twin_KV_2;
            const size_t twin_smem    = std::max(twin_combine, twin_Q_in_reg ?
                std::max(twin_Q, twin_KV + twin_mask) : twin_Q + twin_KV + twin_mask);
            fattn_kernel_t twin = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, GGML_TYPE_F16, GGML_TYPE_F16>;
#if !defined(GGML_USE_MUSA)
            static bool twin_smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!twin_smem_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(twin), cudaFuncAttributeMaxDynamicSharedMemorySize, twin_smem));
                twin_smem_raised[id] = true;
            }
#endif // !defined(GGML_USE_MUSA)
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_ref, twin, nthreads, twin_smem));
        }
    }

    // Use the positional-mask instance's own occupancy for the stream-K grid.
    if constexpr (type_K == GGML_TYPE_XYZKV2_0 && type_V == GGML_TYPE_XYZKV2_0 && ncols1*ncols2 == 64 && DKQ == 256 && DV == 256) {
        if (occupancy_ref == 0 && dst->src[3] != nullptr && dst->src[3]->type == GGML_TYPE_F32) {
            fattn_kernel_t plain = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view, false, type_K, type_V>;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_ref, plain, nthreads, nbytes_shared_total));
        }
    }

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, need_f16_K, need_f16_V, true, use_sparse, warp_size_host,
         pair_d != ncols2 ? pair_d : 0, occupancy_ref);
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// Native cache instantiation reads K and V directly from quantized storage.
#define DECL_FATTN_MMA_XYZKV_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                        \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                                   \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// Native q4_0 (template-instances/fattn-mma-q4_0-instance-dkq256.cu): declared extern so fattn.cu, which dispatches them,
// does not compile them a second time.
extern DECL_FATTN_MMA_XYZKV_CASE(256, 256, 2, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_MMA_XYZKV_CASE(256, 256, 4, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_MMA_XYZKV_CASE(256, 256, 8, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  1,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// Mistral Small 4 (DKQ=320, DV=256), GQA=32-only build:
extern DECL_FATTN_MMA_F16_CASE(320, 256,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(320, 256,  2, 32);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  2, 32);
