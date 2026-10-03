#pragma once
// The engine's rotation building blocks: the butterfly stages of a Walsh-Hadamard transform (inside a warp by shuffle,
// across warps through shared memory, across registers of one thread), the q8_1 twin store the PTQ1 kernels read, the
// two activations the chains fold in, and xyzkv's sign vectors. A stage h pairs elements e and e ^ h: the lower one
// becomes a + b, the upper b_lower - a_upper. Only the ORDER of the stages fixes the bits -- whichever lane, warp or
// register carries a pair -- and every kernel here runs them h = 1, 2, ..., N/2, the order the fork's kernels use, so the
// engine's rotations are the fork's bits (tools/had_test.cu).
#include "common.cuh"
#include "ptq1_rec.cuh"

namespace eng {

__device__ __forceinline__ float act_silu(const float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float act_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// stage h < 32: the partner is lane ^ h
template <int NE>
__device__ __forceinline__ void had_stage_lane(float (&reg)[NE], const int h, const int lane) {
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const float a = reg[j];
        const float b = __shfl_xor_sync(0xFFFFFFFF, a, h, WARP_SIZE);
        reg[j] = (lane & h) == 0 ? a + b : b - a;
    }
}

// stage 32 <= h < NT: the partner is thread tid ^ h of the NT-thread group, through s (NE*NT floats)
template <int NE, int NT>
__device__ __forceinline__ void had_stage_smem(float (&reg)[NE], float * s, const int h, const int tid) {
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        s[j*NT + tid] = reg[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        const float a = reg[j];
        const float b = s[j*NT + (tid ^ h)];
        reg[j] = (tid & h) == 0 ? a + b : b - a;
    }
    __syncthreads();
}

// the stages above the group width: register j pairs with register j + step of the same thread
template <int NE>
__device__ __forceinline__ void had_stages_reg(float (&reg)[NE]) {
#pragma unroll
    for (int step = 1; step < NE; step *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k];
                const float b = reg[j + k + step];
                reg[j + k]        = a + b;
                reg[j + k + step] = a - b;
            }
        }
    }
}

// every stage of an N-point transform whose element i*NT + tid sits in register i of thread tid (NT threads call it)
template <int N, int NT>
__device__ __forceinline__ void had_all(float (&reg)[N/NT], float * s, const int tid) {
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
        had_stage_lane(reg, h, tid % WARP_SIZE);
    }
#pragma unroll
    for (int h = WARP_SIZE; h < NT; h *= 2) {
        had_stage_smem<N/NT, NT>(reg, s, h, tid);
    }
    had_stages_reg(reg);
}

// The element at flat index i of an activation into its q8_1 twin: the ptq1_perm record layout the PTQ1 tensor-core
// kernels read, the bytes quantize_q8_1<true> writes (one 32-value block per warp, lane = value index, the same
// reductions, d, rounding and record position, the -sum(q) start in the pair's second half). All 32 lanes call it.
__device__ __forceinline__ void q8_twin_store(char * __restrict__ vy, const int64_t i, const float xi) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int64_t ib  = i / QK8_1;
    const int64_t iqs = i % QK8_1;
    char * rec = vy + (ib / 4) * (4 * (int64_t) sizeof(block_q8_1));
    const int s = ib % 4;
    rec[ptq1_rec_pos(s, (int) iqs)] = q;   // ptq1_rec.cuh
    const int sum_q = warp_reduce_sum<QK8_1>((int) q);
    if (iqs == 0) {
        ((half2 *) (rec + 4*QK8_1))[s] = ptq1_perm_ds(d, sum, sum_q);
    }
}

// One 1024-block row on an NT-thread group (tid < NT; all call it, it syncs): row r of src, or with glu block r % n_blk of
// token r / n_blk of a [2*glu_nc, tokens] SwiGLU input (silu(gate half) * up half in the load); * scale, * the signs of
// block r % n_blk when signed_rows; every stage; dst row r and, with q8, its twin. s: 1024 floats of shared memory. The
// network does not depend on NT (the same pairs in the same stage order), so every NT gives the same bits.
template <int NT, bool signed_rows, bool glu>
__device__ __forceinline__ void had_block_row(const float * src, float * dst, const int64_t r, const float scale,
                                              const float * signs, const int n_blk, char * __restrict__ q8,
                                              const int64_t glu_nc, const int tid, float * s) {
    constexpr int N  = 1024;
    constexpr int NE = N / NT;
    static_assert(!glu || signed_rows, "the SwiGLU load is only wired for the signed transform");
    if constexpr (glu) {
        src += (r / n_blk) * 2 * glu_nc + (r % n_blk) * N;
    } else {
        src += r * N;
    }
    dst += r * N;
    const float * sg = signed_rows ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        float v;
        if constexpr (glu) {
            v = act_silu(src[i * NT + tid]) * src[glu_nc + i * NT + tid];
        } else {
            v = src[i * NT + tid];
        }
        reg[i] = v * scale;
        if (signed_rows) {
            reg[i] *= sg[i * NT + tid];
        }
    }
    had_all<N, NT>(reg, s, tid);
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
        if (q8 != nullptr) {   // uniform per launch: whole warps store the twin
            q8_twin_store(q8, r * N + i * NT + tid, reg[i]);
        }
    }
}

// xyzkv's 128-point rotation signs (seed 42): forward x*S1 -> WHT -> *S2, inverse x*S2 -> WHT -> *S1
static __constant__ float xyzkv_s1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float xyzkv_s2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

} // namespace eng
