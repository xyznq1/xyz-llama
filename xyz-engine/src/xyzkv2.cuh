#pragma once
// The attention cache format, xyzkv2_0 (ggml-common.h block_xyzkv2_0: a half corrected L2 norm + 128 two-bit indices, one
// block per 128-group): the Lloyd-Max centroids for N(0, 1/128) and the decision midpoints between them.
#include "common.cuh"

namespace eng {

static __constant__ float xyzkv2_centroid[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };
static __constant__ float xyzkv2_mid[3]      = { -0.086728f, 0.0f, 0.086728f };

__device__ __forceinline__ uint8_t xyzkv2_nearest(const float v) {
    if      (v < xyzkv2_mid[0]) return 0;
    else if (v < xyzkv2_mid[1]) return 1;
    else if (v < xyzkv2_mid[2]) return 2;
    else                        return 3;
}

} // namespace eng
