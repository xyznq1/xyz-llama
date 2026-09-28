#pragma once
// PTQ1_0, the target's weight format (ggml-common.h block_ptq1_0: 24 bytes of 5 trits + 2 bytes of 4 trits + a half scale
// LAST, 128 weights per block), and its ILV16 device layout: in each whole 16-row tile, the 16 rows of one k-block sit
// together ([k-block][16 rows]); the rows from (nrows & ~15) on stay row-major.
#include "common.cuh"

namespace eng {

// block index of (row, kb) in a PTQ1_0 matrix of nrows rows, stride_row blocks per row
__host__ __device__ __forceinline__ int64_t ptq1_block(const int64_t row, const int64_t kb, const int64_t stride_row,
                                                       const int64_t nrows) {
    return row < (nrows & ~(int64_t) 15) ? (row & ~(int64_t) 15)*stride_row + kb*16 + (row & 15) : row*stride_row + kb;
}

// element e of a block in {-1, 0, 1}: e < 80 is trit e/16 of qs[e%16], e < 120 trit (e-80)/8 of qs[16 + (e-80)%8], else
// trit (e-120)/2 of qh[(e-120)%2]; trit t of byte b is ((b * 3^t mod 256) * 3) >> 8
__device__ __forceinline__ float ptq1_elem(const block_ptq1_0 * x, const int e) {
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    uint8_t b;
    int t;
    if (e < 80) {
        t = e >> 4;
        b = x->qs[e & 15];
    } else if (e < 120) {
        const int e2 = e - 80;
        t = e2 >> 3;
        b = x->qs[16 + (e2 & 7)];
    } else {
        const int e2 = e - 120;
        t = e2 >> 1;
        b = x->qh[e2 & 1];
    }
    const uint8_t q = b * pow3[t];
    const int16_t xi = ((uint16_t) q * 3) >> 8;
    return (float) (xi - 1);
}

} // namespace eng
