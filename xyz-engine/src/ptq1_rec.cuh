#pragma once
// The ptq1_perm activation record the PTQ1 verify kernels read (one 128-value group = 4 q8_1 blocks: 128 B of q in the
// tensor-core fragment order, then 4 x half2 (d, -sum q)): where value iqs (0..31) of slice s (the group's q8_1 block s)
// goes.
//
// A slice is one 32-wide MMA step with one activation scale, and its int32 sum is exact in any order -- so a value may
// sit at any fragment position kk inside its slice, as long as the weight decode puts the matching weight there.
//
// Slices 0, 1 and the first half of slice 2 keep kk = iqs (each lane decodes its own qs word, the PTQ1_0 element
// order). The rest -- elements 80..127, the tail bytes qs[16..23] and the qh pair -- is laid out so lane c decodes ITS
// OWN two tail bytes 16+2c, 17+2c (all five trits) and two trits of qh[c & 1], with no per-lane source selects
// (k_ptq1_own.cu ptq1_decode_row_ut):
//   slice 2, kk 16+4c..19+4c: trit 0 of bytes 16+2c, 17+2c, trit 1 of the same      (elements 80+2c, 81+2c, 88+2c, 89+2c)
//   slice 3, kk  4c.. 3+4c  : trits 2 and 3 of bytes 16+2c, 17+2c                   (96+2c, 97+2c, 104+2c, 105+2c)
//   slice 3, kk 16+4c..19+4c: trit 4 of bytes 16+2c, 17+2c, trits t, t+1 of qh[c&1] (112+2c, 113+2c, 120+2t+h, 122+2t+h),
//                             t = 2(c >> 1), h = c & 1
// ggml's quantize_q8_1<true> writes kk = iqs everywhere: the same bytes, at the fork kernels' positions.
#include "common.cuh"

namespace eng {

// the fragment position of slice s's value iqs
static __host__ __device__ inline int ptq1_rec_kk(const int s, const int iqs) {
    if (s < 2 || (s == 2 && iqs < 16)) {
        return iqs;
    }
    if (s == 2) {   // tail trits 0, 1: element 80 + 8t + j, byte j = u % 8, trit t = u / 8
        const int u = iqs - 16;
        return 16 + 4*((u % 8)/2) + 2*(u/8) + u % 2;
    }
    if (iqs < 16) {   // tail trits 2, 3
        return 4*((iqs % 8)/2) + 2*(iqs/8) + iqs % 2;
    }
    if (iqs < 24) {   // tail trit 4
        const int j = iqs - 16;
        return 16 + 4*(j/2) + j % 2;
    }
    const int v = iqs - 24;   // the qh pair: element 120 + 2t + h, t = v / 2, h = v % 2 -> lane h + 2(t/2), stage t % 2
    return 16 + 4*(v % 2 + 2*(v/4)) + 2 + (v/2) % 2;
}

// the byte offset of slice s's value iqs in the record's q area: lane c = (kk%16)/4 holds 8 words, word 2s + kk/16 of
// them is its B register for slice s, byte kk % 4
static __host__ __device__ inline int ptq1_rec_pos(const int s, const int iqs) {
    const int kk = ptq1_rec_kk(s, iqs);
    return ((kk % 16)/4*8 + 2*s + kk/16)*4 + kk % 4;
}

} // namespace eng
