// The engine's own MMQ launches (src/mmq/mmq.cuh): ggml_cuda_mul_mat_q's plain path for a 2-D quantized weight
// [K, nrows] (PTQ1_0 ILV16 or Q3_K) and an f32 activation [K, ncols] -- the activation into the D4 q8_1 MMQ layout, the
// n-tile width J chosen as mul_mat_q_switch_J chooses it (the fewest column tiles, the smallest J that reaches it, J <=
// 128), the stream-K grid as launch_mul_mat_q sizes it (every tile if they fill >= 90% of the last wave, else one
// block per SM) and the fixup when the tiles do not divide it. Buffers are the engine's, reserved outside any capture.
#include "mmq/mmq.cuh"

#include "kernels.h"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace eng {

namespace {

int     g_nsm        = 0;
bool    g_pipe       = false;     // the device has cp.async (Ampere+): PTQ1_0 J = 128 runs the pipeline (mq::pipelined)
float * g_fixup      = nullptr;   // one block's I x J partial tile per block of a one-per-SM grid
char *  g_q8         = nullptr;
size_t  g_q8_bytes   = 0;

// the kernel's dynamic shared memory: the pipeline's extra wherever the device code may take it (a device without
// cp.async runs code built without it; an Ampere+ device may run older PTX, where the extra goes unused)
template <ggml_type type, int J>
int smem_bytes() {
    return mq::nbytes_shared<type>(J) + (g_pipe && mq::pipelined(type, J) ? mq::nbytes_shared_pipeline(J) : 0);
}

template <ggml_type type, int J>
void set_smem() {
    CUDA_CHECK(cudaFuncSetAttribute(mq::k_mmq<type, J>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<type, J>()));
}

template <ggml_type type>
void set_smem_all() {
    set_smem<type,   8>(); set_smem<type,  16>(); set_smem<type,  24>(); set_smem<type,  32>();
    set_smem<type,  40>(); set_smem<type,  48>(); set_smem<type,  64>(); set_smem<type,  80>();
    set_smem<type,  96>(); set_smem<type, 112>(); set_smem<type, 128>();
}

size_t q8_bytes(const int64_t ncols, const int64_t K) {
    const int64_t Kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    return (size_t) (ncols*Kp*(int64_t) sizeof(mq::block_q8_1_mmq)/mq::QK8_1_MMQ + 128*(int64_t) sizeof(mq::block_q8_1_mmq));
}

template <ggml_type type, int J>
void launch(cudaStream_t st, const void * w, const int * y, const int64_t K, const int64_t nrows, const int64_t ncols,
            float * dst, const int64_t dst_stride) {
    constexpr int QK = mq::qk<type>();
    const int nty = (int) ((nrows + mq::I - 1) / mq::I);
    const int ntx = (int) ((ncols + J - 1) / J);
    const int ntiles_dst = ntx * nty;
    const int tiles_nwaves = (ntiles_dst + g_nsm - 1) / g_nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (g_nsm*tiles_nwaves);
    const int nblocks = tiles_efficiency_percent >= 90 ? ntiles_dst : g_nsm;
    const bool fixup_needed = ntiles_dst % nblocks != 0;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values((uint64_t) (K / QK));
    const uint3 ntx_fd             = init_fastdiv_values((uint64_t) ntx);
    const dim3 block_dims(WARP_SIZE, mq::NWARPS, 1);
    mq::k_mmq<type, J><<<dim3(nblocks, 1, 1), block_dims, smem_bytes<type, J>(), st>>>(
        (const char *) w, y, dst, g_fixup, blocks_per_ne00_fd, (int) nrows, (int) ncols, (int) (K / QK), (int) ncols,
        (int) dst_stride, ntx_fd);
    if (!fixup_needed) {
        return;
    }
    const dim3 block_nums_fixup(nblocks, mq::I/WARP_SIZE, 1);
    const dim3 block_dims_fixup(WARP_SIZE, mq::NWARPS/2, 1);
    mq::k_mmq_fixup<type, J><<<block_nums_fixup, block_dims_fixup, 0, st>>>(dst, g_fixup, blocks_per_ne00_fd, (int) nrows,
        (int) ncols, (int) dst_stride, ntx_fd);
}

template <ggml_type type>
void switch_J(cudaStream_t st, const void * w, const int * y, const int64_t K, const int64_t nrows, const int64_t ncols,
              float * dst, const int64_t dst_stride) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        if (!mq::j_valid(J)) {
            continue;
        }
        const int ntiles_x = (int) ((ncols + J - 1) / J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    switch (J_best) {
        case   8: launch<type,   8>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  16: launch<type,  16>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  24: launch<type,  24>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  32: launch<type,  32>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  40: launch<type,  40>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  48: launch<type,  48>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  64: launch<type,  64>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  80: launch<type,  80>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case  96: launch<type,  96>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case 112: launch<type, 112>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        case 128: launch<type, 128>(st, w, y, K, nrows, ncols, dst, dst_stride); break;
        default:
            fprintf(stderr, "mmq: no tile width for %lld columns\n", (long long) ncols);
            abort();
    }
}

} // namespace

void mmq_reserve(int64_t max_cols, int64_t max_K) {
    if (g_nsm == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
        int cc_major = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
        g_pipe = cc_major >= 8;
        CUDA_CHECK(cudaMalloc(&g_fixup, (size_t) g_nsm*mq::I*128*sizeof(float)));
        set_smem_all<GGML_TYPE_PTQ1_0>();
        set_smem_all<GGML_TYPE_Q3_K>();
    }
    const size_t need = q8_bytes(max_cols, max_K);
    if (need > g_q8_bytes) {
        if (g_q8 != nullptr) {
            CUDA_CHECK(cudaFree(g_q8));
        }
        CUDA_CHECK(cudaMalloc(&g_q8, need));
        g_q8_bytes = need;
    }
}

void mmq(cudaStream_t st, int type_i, const void * w, int64_t K, int64_t nrows, const float * x, int64_t x_stride,
         int64_t ncols, float * dst, int64_t dst_stride) {
    const ggml_type type = (ggml_type) type_i;
    if ((type != GGML_TYPE_PTQ1_0 && type != GGML_TYPE_Q3_K) || nrows % mq::I != 0 || K % 4 != 0) {
        fprintf(stderr, "mmq: %s %lld x %lld is not an instantiated case (PTQ1_0 / Q3_K, rows a multiple of 128)\n",
                ggml_type_name(type), (long long) nrows, (long long) K);
        abort();
    }
    if (q8_bytes(ncols, K) > g_q8_bytes) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(st, &cs));
        if (cs != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "mmq: %lld columns x K %lld exceed the reserved activation buffer inside a capture\n",
                    (long long) ncols, (long long) K);
            abort();
        }
        mmq_reserve(ncols, K);
    }
    // quantize_mmq_q8_1_cuda (D4): grid (columns, 512-value chunks), 128 threads
    const int64_t Kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t block_num_y = (Kp + 4*128 - 1) / (4*128);
    mq::k_quantize_d4<<<dim3((unsigned) ncols, (unsigned) block_num_y, 1), dim3(128, 1, 1), 0, st>>>(x, g_q8, K, x_stride,
                                                                                                   Kp, (int) ncols);
    if (type == GGML_TYPE_PTQ1_0) {
        switch_J<GGML_TYPE_PTQ1_0>(st, w, (const int *) g_q8, K, nrows, ncols, dst, dst_stride);
    } else {
        switch_J<GGML_TYPE_Q3_K>(st, w, (const int *) g_q8, K, nrows, ncols, dst, dst_stride);
    }
}

void mmq_ptq1(cudaStream_t st, const void * w, int64_t K, int64_t nrows, const float * x, int64_t x_stride, int64_t ncols,
              float * dst, int64_t dst_stride) {
    mmq(st, GGML_TYPE_PTQ1_0, w, K, nrows, x, x_stride, ncols, dst, dst_stride);
}

} // namespace eng
