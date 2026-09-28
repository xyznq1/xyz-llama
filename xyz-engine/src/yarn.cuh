#pragma once
// YaRN rotary angles (forward): the corrected angle's cos / sin times the magnitude scale -- rope.cu's rope_yarn<true>
// (ggml's YaRN, after LlamaYaRNScaledRotaryEmbedding.py, MIT, Jeffrey Quesnelle and Bowen Peng).
#include "common.cuh"

namespace eng {

__device__ __forceinline__ float yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

__device__ __forceinline__ void yarn(const float theta_extrap, const float freq_scale, const float corr0, const float corr1,
                                     const int64_t i0, const float ext_factor, float mscale, float & cos_theta,
                                     float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

} // namespace eng
