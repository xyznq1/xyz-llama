#pragma once

#include "common.cuh"
#include "fwht-block.cuh"
#include "unary.cuh"

template <bool permuted>
static __device__ __forceinline__ void ggml_cuda_gdn_out_chain_task(
        const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ gate,
        const float * __restrict__ signs, float * __restrict__ dst, const float eps, const float fwht_scale,
        char * __restrict__ q8_out, const int r, const int tid, float * s_values, float * s_sum,
        const float preloaded_w, const float * preloaded_signs) {
    constexpr int HD   = 128;
    constexpr int NK   = 16;
    constexpr int REP  = 3;
    constexpr int N    = 1024;
    constexpr int NT   = 256;
    constexpr int NE   = N / NT;
    constexpr int NBLK = HD*NK*REP/N;
    constexpr int NH   = N/HD;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int token = r / NBLK;
    const int blk   = r % NBLK;
    const int lane  = tid % warp_size;

#pragma unroll
    for (int local_head = 0; local_head < NH; ++local_head) {
        const int grouped_head = blk*(N/HD) + local_head;
        const int nk           = grouped_head / REP;
        const int rep          = grouped_head % REP;
        const int source_head  = permuted ? nk + NK*rep : grouped_head;
        const int64_t off      = ((int64_t) token*(NK*REP) + source_head)*HD;

        const float xi  = tid < HD ? x[off + tid] : 0.0f;
        float       sum = tid < HD ? xi*xi : 0.0f;
        sum = block_reduce<block_reduce_method::SUM, NT>(sum, s_sum);
        const float scale = rsqrtf(sum/HD + eps);
        if (tid < HD) {
            const float wi = preloaded_signs != nullptr ? preloaded_w : w[tid];
            s_values[local_head*HD + tid] = scale*xi*wi*ggml_cuda_op_silu_single(gate[off + tid]);
        }
        __syncthreads();
    }

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int col = i*NT + tid;
        reg[i] = s_values[col]*fwht_scale;
        reg[i] *= preloaded_signs != nullptr ? preloaded_signs[i] : signs[blk*N + col];
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            s_values[j*NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = s_values[j*NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
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

    const int64_t base = (int64_t) r*N;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[base + i*NT + tid] = reg[i];
        if (q8_out != nullptr) {
            ptq1_q8_perm_store(q8_out, base + i*NT + tid, reg[i]);
        }
    }
}
