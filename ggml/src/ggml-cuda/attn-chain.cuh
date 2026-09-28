#pragma once

#include "common.cuh"
#include "ggml.h"

// Device code shared by the attention layer's chain fusions:
// rope.cu's YaRN helpers, moved here verbatim so both files compile the same code, and the Q/K head prologue.

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// The M-RoPE parameters of a ROPE node, as rope_multi_cuda computes them on the host.
struct attn_mrope_params {
    int            n_dims;
    int            n_offs;
    float          freq_scale;
    float          ext_factor;
    float          attn_factor;
    rope_corr_dims corr_dims;
    float          theta_scale;
    mrope_sections sections;
    bool           is_imrope;
};

// false: not an M-RoPE (vision excluded) the 256-wide head prologue handles
static bool attn_mrope_params_get(const ggml_tensor * rope, attn_mrope_params & p) {
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
    p.n_dims    = ((const int32_t *) rope->op_params)[1];
    p.n_offs    = ((const int32_t *) rope->op_params)[15];
    p.is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    if (!is_mrope || is_vision || p.n_dims % 2 != 0 || p.n_offs % 2 != 0 || p.n_offs + p.n_dims > 256 ||
            rope->src[1] == nullptr || rope->src[1]->type != GGML_TYPE_I32 || rope->src[2] != nullptr) {
        return false;
    }
    float freq_base, beta_fast, beta_slow;
    memcpy(&freq_base,     (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&p.freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&p.ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,     (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,     (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&p.sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);
    ggml_rope_yarn_corr_dims(p.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, p.corr_dims.v);
    p.theta_scale = powf(freq_base, -2.0f / p.n_dims);   // rope_multi_cuda's, on the host
    return true;
}

// One (head, token) row of 256 through RMS_NORM + MUL(w), M-RoPE and the KV rotation's 256-point Hadamard, into s[256]
// (shared; complete and synced on return). Called by all 256 threads with x at the row. Bit-identical, stage by stage:
//   norm  rms_norm_f32<256, true>: one element per thread, the same block_reduce, mean = tmp/256, (scale*x)*w
//   rope  rope_multi<true, false>: the same sector/theta_base expressions, the host's theta_scale, the same rope_yarn,
//         x0*cos - x1*sin / x0*sin + x1*cos -- each pair read and written by one thread (in place is exact)
//   fwht  fwht_cuda<256, false>: x*scale first, then stages h = 1, 2, ..., 128 (lower = a + b, upper = a - b); a stage
//         computes the same pairs whichever thread carries them
static __device__ __forceinline__ void attn_head_prologue_256(
        float * s, float * s_sum, const float * x, const float * w, const float eps,
        const int32_t * pos, const int tok, const int ne02, const attn_mrope_params rp, const float fwht_scale) {
    constexpr int N = 256;
    const int tid = threadIdx.x;

    // 1. RMS_NORM + MUL
    const float xi = x[tid];
    float tmp = 0.0f;
    tmp += xi * xi;
    tmp = block_reduce<block_reduce_method::SUM, N>(tmp, s_sum);
    const float mean  = tmp / N;
    const float scale = rsqrtf(mean + eps);
    s[tid] = scale * xi * w[tid];
    __syncthreads();

    // 2. ROPE (M-RoPE): thread p owns pair p; pairs outside [n_offs, n_offs + n_dims) pass through
    if (tid < N/2) {
        const int i0 = 2*tid;
        if (!(i0 < rp.n_offs || i0 >= rp.n_offs + rp.n_dims)) {
            const int iw = i0 - rp.n_offs;

            const mrope_sections sections = rp.sections;
            const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
            const int sec_w = sections.v[1] + sections.v[0];
            const int sector = (iw / 2) % sect_dims;

            float theta_base = 0.0;
            if (rp.is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
                    theta_base = pos[tok + ne02 * 1] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
                    theta_base = pos[tok + ne02 * 2] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
                    theta_base = pos[tok] * powf(rp.theta_scale, iw / 2.0f);
                } else {
                    theta_base = pos[tok + ne02 * 3] * powf(rp.theta_scale, iw / 2.0f);
                }
            } else {
                if (sector < sections.v[0]) {
                    theta_base = pos[tok] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= sections.v[0] && sector < sec_w) {
                    theta_base = pos[tok + ne02 * 1] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                    theta_base = pos[tok + ne02 * 2] * powf(rp.theta_scale, iw / 2.0f);
                } else if (sector >= sec_w + sections.v[2]) {
                    theta_base = pos[tok + ne02 * 3] * powf(rp.theta_scale, iw / 2.0f);
                }
            }

            const float freq_factor = 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn<true>(theta_base/freq_factor, rp.freq_scale, rp.corr_dims, iw, rp.ext_factor, rp.attn_factor,
                            cos_theta, sin_theta);

            const int a = i0/2 + rp.n_offs/2;
            const int b = a + rp.n_dims/2;
            const float x0 = s[a];
            const float x1 = s[b];
            s[a] = x0*cos_theta - x1*sin_theta;
            s[b] = x0*sin_theta + x1*cos_theta;
        }
    }
    __syncthreads();

    // 3. the KV rotation: 256-point Hadamard
    s[tid] = s[tid] * fwht_scale;
    __syncthreads();
#pragma unroll
    for (int h = 1; h < N; h *= 2) {
        if (tid < N/2) {
            const int e = (tid / h) * 2*h + (tid % h);
            const float a = s[e];
            const float b = s[e + h];
            s[e]     = a + b;
            s[e + h] = a - b;
        }
        __syncthreads();
    }
}
