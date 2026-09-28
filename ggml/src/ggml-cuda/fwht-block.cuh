#pragma once

// Wide-row Hadamard device function and its q8_1 twin store.

#include "common.cuh"
#include "unary.cuh"

// The element at flat index i_cont of an activation, stored in the ptq1_perm q8_1 record layout that the PTQ1_0
// tensor-core kernel v1 reads (the "twin", common.cuh ptq1_q8_twin). Byte-identical to quantize_q8_1<true> (quantize.cu):
// one warp holds one 32-value block with lane == value index, the same warp reductions, the same d and rounding, the same
// record position. All 32 lanes of the warp must call it together.
static __device__ __forceinline__ void ptq1_q8_perm_store(char * __restrict__ vy, const int64_t i_cont, const float xi) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;
    char * rec = vy + (ib / 4) * (4 * (int64_t) sizeof(block_q8_1));
    const int s = ib % 4;
    rec[((iqs % 16)/4*8 + 2*s + iqs/16)*4 + iqs % 4] = q;
    const int sum_q = warp_reduce_sum<QK8_1>((int) q);   // PTQ1_MMA_ISUM: v1's accumulator start (common.cuh)
    if (iqs == 0) {
        ((half2 *) (rec + 4*QK8_1))[s] = ptq1_perm_ds(d, sum, sum_q);
    }
}

// One row of fwht_cuda_block. Row r: with glu, block r % n_blk of token r / n_blk of a [2*glu_nc, tokens] SWIGLU input
// (silu(first half) * second half, computed in the load); else row r of src. tid: this thread's index in the NT-thread
// group (all NT threads call it together; it uses __syncthreads). s: N floats of shared memory. cg_src: read src through
// L2 only (the chain reads rows other CTAs of the same launch wrote).
template <int N, int NT, bool has_signs, bool glu, bool cg_src = false>
static __device__ __forceinline__ void fwht_block_row(const float * src, float * dst, const int64_t r, const float scale,
        const float * signs, const int n_blk, char * __restrict__ q8_out, const int64_t glu_nc, const int tid, float * s) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");
    static_assert(!glu || has_signs, "the swiglu load is only wired for the signed transform");

    if constexpr (glu) {
        src += (r / n_blk) * 2 * glu_nc + (r % n_blk) * N;
    } else {
        src += r * N;
    }
    dst += r * N;

    const int lane = tid % warp_size;

    auto ld = [](const float * p) -> float {
        if constexpr (cg_src) {
            return __ldcg(p);
        } else {
            return *p;
        }
    };

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        float v;
        if constexpr (glu) {
            v = ggml_cuda_op_silu_single(ld(src + i * NT + tid)) * ld(src + glu_nc + i * NT + tid);
        } else {
            v = ld(src + i * NT + tid);
        }
        reg[i] = v * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
        if (q8_out != nullptr) {   // the q8_1 twin (ptq1_q8_perm_store): uniform per launch, so whole warps call it
            ptq1_q8_perm_store(q8_out, r * N + i * NT + tid, reg[i]);
        }
    }
}
