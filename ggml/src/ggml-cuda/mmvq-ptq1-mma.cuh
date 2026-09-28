#pragma once
// PTQ1_0 x Q8_1 on tensor cores for 1..32 columns. Weights decode directly into MMA fragments.
//
// Fragment layout (PTX m16n8k32 .s8, what mma(tile<16,8,int>, tile<16,8,int>, tile<8,8,int>) passes through):
//   A.x[0] rows g,   k 4c..4c+3      A.x[1] rows g+8, k 4c..4c+3        (g = lane/4, c = lane%4)
//   A.x[2] rows g,   k 16+4c..+3     A.x[3] rows g+8, k 16+4c..+3
//   B.x[0] col g, k 4c..4c+3         B.x[1] col g, k 16+4c..+3
//   D.x[0] (g, 2c)  D.x[1] (g, 2c+1)  D.x[2] (g+8, 2c)  D.x[3] (g+8, 2c+1)
// PTQ1_0's own element order lines up with it (element t*16+m = byte m of qs[0..15] at trit t): lane c's A registers
// for slices 0-2 are the successive trits of ONE word, qs[4c..4c+3] -- the llamAmpere lane word. Slices 2-3 take word
// (c&1) of qs[16..23] at a lane-dependent trit, and lanes 2/3 one qh group, chosen with selects (no divergence).

#include "mma.cuh"

static constexpr int ptq1_mma_max_cols = 32;

// 4 packed trit digits (0..2 per byte) -> 4 signed weights (-1..1 per byte): +0x7F per byte cannot carry (<= 0x81),
// then the xor re-centres. Two instructions for 4 weights.
static __device__ __forceinline__ int ptq1_signed4(const uint32_t q) {
    return (int) ((q + 0x7F7F7F7Fu) ^ 0x80808080u);
}

// The permuted path feeds raw digits d = w + 1 and initializes each accumulator with -sum(q).
// Thus sum(d*q) - sum(q) equals sum(w*q) in the same int32 range.

template <bool digits>
static __device__ __forceinline__ int ptq1_a4(const uint32_t q) {
    if constexpr (digits) {
        return (int) q;
    } else {
        return ptq1_signed4(q);
    }
}

// Read the -sum(q) stored in the second half of a permuted activation record.
static __device__ __forceinline__ int ptq1_neg_isum(const half2 ds) {
    return (int) __half_as_short(__high2half(ds));
}

// Successive trits of the 4 bytes of one word: bytes 0,1 in the 16-bit halves of lo, 2,3 in hi (b*3 <= 765 never
// crosses a half). ptq1_walk_next returns trit t of each byte, packed in byte order, and advances t. (Free functions,
// not a struct: MSVC's host pass parses constructor bodies and has no __byte_perm.)
static __device__ __forceinline__ void ptq1_walk_init(const uint32_t packed, uint32_t & lo, uint32_t & hi) {
    lo = __byte_perm(packed, 0, 0x4140);
    hi = __byte_perm(packed, 0, 0x4342);
}

static __device__ __forceinline__ uint32_t ptq1_walk_next(uint32_t & lo, uint32_t & hi) {
    const uint32_t wl = lo * 3;
    const uint32_t wh = hi * 3;
    lo = wl & 0x00FF00FFu;
    hi = wh & 0x00FF00FFu;
    return __byte_perm(wl, wh, 0x7531);
}

// The 16 A registers (4 slices x {x0..x3}) one lane needs from its two rows' blocks, as signed int8x4.
struct ptq1_frag {
    int s[4][4];
};

template <bool digits = false>   // Raw 0..2 digits or signed -1..1 weights.
static __device__ __forceinline__ void ptq1_decode_rows(const block_ptq1_0 * __restrict__ ba, const block_ptq1_0 * __restrict__ bb,
                                                        const int c, ptq1_frag & f) {
    const block_ptq1_0 * rows[2] = { ba, bb };
#pragma unroll
    for (int r = 0; r < 2; ++r) {                         // r = 0: row g (x0/x2), r = 1: row g+8 (x1/x3)
        const block_ptq1_0 * b = rows[r];
        uint32_t ql, qhi, tl, thi;
        ptq1_walk_init(get_int_b4(b->qs, c), ql, qhi);          // qs[4c..4c+3]: elements t*16 + 4c + j
        ptq1_walk_init(get_int_b4(b->qs + 16, c & 1), tl, thi); // qs[16+4(c&1)..]: elements 80 + t*8 + 4(c&1) + j
        const uint32_t qhd = (uint32_t) get_int_b4(b, 6);       // qh[0] | qh[1] << 8 (then d, unused here)
        const uint32_t q0 = ptq1_walk_next(ql, qhi);
        const uint32_t q1 = ptq1_walk_next(ql, qhi);
        const uint32_t q2 = ptq1_walk_next(ql, qhi);
        const uint32_t q3 = ptq1_walk_next(ql, qhi);
        const uint32_t q4 = ptq1_walk_next(ql, qhi);
        const uint32_t u0 = ptq1_walk_next(tl, thi);
        const uint32_t u1 = ptq1_walk_next(tl, thi);
        const uint32_t u2 = ptq1_walk_next(tl, thi);
        const uint32_t u3 = ptq1_walk_next(tl, thi);
        const uint32_t u4 = ptq1_walk_next(tl, thi);
        // qh: element 120 + 2t + h = trit t of qh[h]; lane 2 takes t 0-1 (120..123), lane 3 t 2-3 (124..127)
        uint32_t v = (qhd & 0xFFu) | ((qhd & 0xFF00u) << 8);
        const uint32_t h0 = v * 3; v = h0 & 0x00FF00FFu;
        const uint32_t h1 = v * 3; v = h1 & 0x00FF00FFu;
        const uint32_t h2 = v * 3; v = h2 & 0x00FF00FFu;
        const uint32_t h3 = v * 3;
        const uint32_t qh01 = __byte_perm(h0, h1, 0x7531);
        const uint32_t qh23 = __byte_perm(h2, h3, 0x7531);
        const bool upper = c >= 2;
        f.s[0][0 + r] = ptq1_a4<digits>(q0);                                   // k  0..15: trit 0
        f.s[0][2 + r] = ptq1_a4<digits>(q1);                                   // k 16..31: trit 1
        f.s[1][0 + r] = ptq1_a4<digits>(q2);
        f.s[1][2 + r] = ptq1_a4<digits>(q3);
        f.s[2][0 + r] = ptq1_a4<digits>(q4);                                   // k 64..79
        f.s[2][2 + r] = ptq1_a4<digits>(upper ? u1 : u0);                      // k 80..95: tail trit c>>1
        f.s[3][0 + r] = ptq1_a4<digits>(upper ? u3 : u2);                      // k 96..111: tail trit 2 + (c>>1)
        f.s[3][2 + r] = ptq1_a4<digits>(!upper ? u4 : (c == 2 ? qh01 : qh23)); // k 112..127
    }
}

struct ptq1_words {
    uint32_t q, t, hd;
};

static __device__ __forceinline__ ptq1_words ptq1_load_words(const block_ptq1_0 * __restrict__ b, const int c) {
    ptq1_words r;
    r.q  = (uint32_t) get_int_b4(b->qs, c);
    r.t  = (uint32_t) get_int_b4(b->qs + 16, c & 1);
    r.hd = (uint32_t) get_int_b4(b, 6);
    return r;
}

static __device__ __forceinline__ float ptq1_word_scale(const uint32_t hd) {
    return __half2float(__ushort_as_half((unsigned short) (hd >> 16)));
}

// same fragment mapping as ptq1_decode_rows, from registers
template <bool digits = false>
static __device__ __forceinline__ void ptq1_decode_words(const ptq1_words & wa, const ptq1_words & wb, const int c,
                                                         ptq1_frag & f) {
    const ptq1_words rows[2] = { wa, wb };
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        uint32_t ql, qhi, tl, thi;
        ptq1_walk_init(rows[r].q, ql, qhi);
        ptq1_walk_init(rows[r].t, tl, thi);
        const uint32_t q0 = ptq1_walk_next(ql, qhi);
        const uint32_t q1 = ptq1_walk_next(ql, qhi);
        const uint32_t q2 = ptq1_walk_next(ql, qhi);
        const uint32_t q3 = ptq1_walk_next(ql, qhi);
        const uint32_t q4 = ptq1_walk_next(ql, qhi);
        const uint32_t u0 = ptq1_walk_next(tl, thi);
        const uint32_t u1 = ptq1_walk_next(tl, thi);
        const uint32_t u2 = ptq1_walk_next(tl, thi);
        const uint32_t u3 = ptq1_walk_next(tl, thi);
        const uint32_t u4 = ptq1_walk_next(tl, thi);
        uint32_t v = (rows[r].hd & 0xFFu) | ((rows[r].hd & 0xFF00u) << 8);
        const uint32_t h0 = v * 3; v = h0 & 0x00FF00FFu;
        const uint32_t h1 = v * 3; v = h1 & 0x00FF00FFu;
        const uint32_t h2 = v * 3; v = h2 & 0x00FF00FFu;
        const uint32_t h3 = v * 3;
        const uint32_t qh01 = __byte_perm(h0, h1, 0x7531);
        const uint32_t qh23 = __byte_perm(h2, h3, 0x7531);
        const bool upper = c >= 2;
        f.s[0][0 + r] = ptq1_a4<digits>(q0);
        f.s[0][2 + r] = ptq1_a4<digits>(q1);
        f.s[1][0 + r] = ptq1_a4<digits>(q2);
        f.s[1][2 + r] = ptq1_a4<digits>(q3);
        f.s[2][0 + r] = ptq1_a4<digits>(q4);
        f.s[2][2 + r] = ptq1_a4<digits>(upper ? u1 : u0);
        f.s[3][0 + r] = ptq1_a4<digits>(upper ? u3 : u2);
        f.s[3][2 + r] = ptq1_a4<digits>(!upper ? u4 : (c == 2 ? qh01 : qh23));
    }
}

// Choose the kernel and activation layout together.
static int ptq1_mma_version(const int ncols_dst, const int nrows_x) {
    const bool v2_wins = (ncols_dst >= 32 && nrows_x >= 16384) || (ncols_dst > 16 && nrows_x >= 65536);
    return v2_wins ? 2 : 1;
}

// dst[col][row] for 16 rows per warp, nwarps warps splitting K, ntiles x 8 columns. ilv: the matrix is ILV16.
template <int ntiles, int nwarps, bool has_fusion, bool ilv>
static __device__ __forceinline__ void mul_mat_ptq1_mma_tile(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int row0, const uint32_t channel_dst, const uint32_t sample_dst) {
#if defined(TURING_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int c    = lane & 3;

    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const int nblocks = ncols_x / QK_PTQ1_0;
    // rows past the end read the last row (valid memory) and are never written
    const int ra = min(row0 + g,     nrows_x - 1);
    const int rb = min(row0 + g + 8, nrows_x - 1);
    // ILV16: in a whole tile row g of block kb is at tile + kb*16 + g (ptq1_ilv_block); a partial tile is row-major
    const bool tile_ilv = ilv && row0 + 16 <= (nrows_x & ~15);
    const int  kstep    = tile_ilv ? 16 : 1;
    const int  oa       = tile_ilv ? row0*stride_row_x + g     : ra*stride_row_x;
    const int  ob       = tile_ilv ? row0*stride_row_x + g + 8 : rb*stride_row_x;
    const int x_off = sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_ptq1_0 * xa = (const block_ptq1_0 *) vx + x_off + oa;
    const block_ptq1_0 * xb = (const block_ptq1_0 *) vx + x_off + ob;
    [[maybe_unused]] const block_ptq1_0 * ga = nullptr;
    [[maybe_unused]] const block_ptq1_0 * gb = nullptr;
    [[maybe_unused]] bool use_gate = false;
    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        if (use_gate) {
            ga = (const block_ptq1_0 *) fusion.gate + x_off + oa;
            gb = (const block_ptq1_0 *) fusion.gate + x_off + ob;
        }
    }
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    float acc[ntiles][4]  = {{0.0f}};
    [[maybe_unused]] float accg[ntiles][4] = {{0.0f}};

    for (int kb = w; kb < nblocks; kb += nwarps) {
        const block_ptq1_0 * pa = xa + kb*kstep;
        const block_ptq1_0 * pb = xb + kb*kstep;
        ptq1_frag fx;
        ptq1_decode_rows<true>(pa, pb, c, fx);
        const float da = ptq1_word_scale((uint32_t) get_int_b4(pa, 6));
        const float db = ptq1_word_scale((uint32_t) get_int_b4(pb, 6));
        [[maybe_unused]] ptq1_frag fg;
        [[maybe_unused]] float dga = 0.0f, dgb = 0.0f;
        if constexpr (has_fusion) {
            if (use_gate) {
                const block_ptq1_0 * pga = ga + kb*kstep;
                const block_ptq1_0 * pgb = gb + kb*kstep;
                ptq1_decode_rows<true>(pga, pgb, c, fg);
                dga = __half2float(pga->d);
                dgb = __half2float(pgb->d);
            }
        }
        float blk[ntiles][4]  = {{0.0f}};
        [[maybe_unused]] float blkg[ntiles][4] = {{0.0f}};
        // lane (g, c): its B column's 8 fragment words (2 x 16 B) and its two C columns' 4 (d, sum) pairs (16 B each), all
        // from the permuted record of k-block kb (quantize.cu: word c*8 + 2s + h = bytes 16h+4c.. of block s; ds at +128)
        int   bw[ntiles][8];
        half2 dsa[ntiles][4];
        half2 dsb[ntiles][4];
#pragma unroll
        for (int nt = 0; nt < ntiles; ++nt) {
            const int colB = nt*8 + g;
            const int colC = nt*8 + 2*c;
            int4 lo = make_int4(0, 0, 0, 0), hi = make_int4(0, 0, 0, 0), za = make_int4(0, 0, 0, 0), zb = make_int4(0, 0, 0, 0);
            if (colB < ncols_dst) {
                const char * rb = (const char *) (y + colB*stride_col_y + kb*(QK_PTQ1_0/QK8_1));
                lo = *(const int4 *) (rb + 32*c);
                hi = *(const int4 *) (rb + 32*c + 16);
            }
            if (colC < ncols_dst) {
                za = *(const int4 *) ((const char *) (y + colC*stride_col_y + kb*(QK_PTQ1_0/QK8_1)) + 4*QK8_1);
            }
            if (colC + 1 < ncols_dst) {
                zb = *(const int4 *) ((const char *) (y + (colC + 1)*stride_col_y + kb*(QK_PTQ1_0/QK8_1)) + 4*QK8_1);
            }
            bw[nt][0] = lo.x; bw[nt][1] = lo.y; bw[nt][2] = lo.z; bw[nt][3] = lo.w;
            bw[nt][4] = hi.x; bw[nt][5] = hi.y; bw[nt][6] = hi.z; bw[nt][7] = hi.w;
            memcpy(dsa[nt], &za, sizeof(za));
            memcpy(dsb[nt], &zb, sizeof(zb));
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            tile_A A;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                A.x[l] = fx.s[s][l];
            }
            [[maybe_unused]] tile_A Ag;
            if constexpr (has_fusion) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    Ag.x[l] = fg.s[s][l];
                }
            }
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
                tile_B B;
                B.x[0] = bw[nt][2*s];
                B.x[1] = bw[nt][2*s + 1];
                const float d8_0 = __low2float(dsa[nt][s]);
                const float d8_1 = __low2float(dsb[nt][s]);
                tile_C C;
                const int s8_0 = ptq1_neg_isum(dsa[nt][s]);
                const int s8_1 = ptq1_neg_isum(dsb[nt][s]);
                C.x[0] = s8_0;
                C.x[1] = s8_1;
                C.x[2] = s8_0;
                C.x[3] = s8_1;
                mma(C, A, B);
                blk[nt][0] += d8_0 * (float) C.x[0];
                blk[nt][1] += d8_1 * (float) C.x[1];
                blk[nt][2] += d8_0 * (float) C.x[2];
                blk[nt][3] += d8_1 * (float) C.x[3];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tile_C Cg;
                        Cg.x[0] = s8_0;
                        Cg.x[1] = s8_1;
                        Cg.x[2] = s8_0;
                        Cg.x[3] = s8_1;
                        mma(Cg, Ag, B);
                        blkg[nt][0] += d8_0 * (float) Cg.x[0];
                        blkg[nt][1] += d8_1 * (float) Cg.x[1];
                        blkg[nt][2] += d8_0 * (float) Cg.x[2];
                        blkg[nt][3] += d8_1 * (float) Cg.x[3];
                    }
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < ntiles; ++nt) {
            acc[nt][0] += da * blk[nt][0];
            acc[nt][1] += da * blk[nt][1];
            acc[nt][2] += db * blk[nt][2];
            acc[nt][3] += db * blk[nt][3];
            if constexpr (has_fusion) {
                accg[nt][0] += dga * blkg[nt][0];
                accg[nt][1] += dga * blkg[nt][1];
                accg[nt][2] += dgb * blkg[nt][2];
                accg[nt][3] += dgb * blkg[nt][3];
            }
        }
    }

    // K-split reduction: warps 1..nwarps-1 hand their partials to warp 0 through shared memory
    constexpr int nacc = ntiles*4*(has_fusion ? 2 : 1);
    __shared__ float red[nwarps > 1 ? nwarps - 1 : 1][nacc][WARP_SIZE];
    if constexpr (nwarps > 1) {
        if (w > 0) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    red[w - 1][nt*4 + l][lane] = acc[nt][l];
                    if constexpr (has_fusion) {
                        red[w - 1][ntiles*4 + nt*4 + l][lane] = accg[nt][l];
                    }
                }
            }
        }
        __syncthreads();
        if (w > 0) {
            return;
        }
#pragma unroll
        for (int v = 0; v < nwarps - 1; ++v) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[nt][l] += red[v][nt*4 + l][lane];
                    if constexpr (has_fusion) {
                        accg[nt][l] += red[v][ntiles*4 + nt*4 + l][lane];
                    }
                }
            }
        }
    }

    float * d        = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    int     d_stride = stride_col_dst;
    int     d_row0   = 0;
    if (fusion.dst_hi2 != nullptr && row0 >= fusion.dst_hi2_row0) {
        d        = fusion.dst_hi2;
        d_stride = fusion.dst_hi2_stride_col;
        d_row0   = fusion.dst_hi2_row0;
    } else if (fusion.dst_hi != nullptr && row0 >= fusion.dst_hi_row0) {   // sibling merge: this 16-row tile is B's (2D only)
        d        = fusion.dst_hi;
        d_stride = fusion.dst_hi_stride_col;
        d_row0   = fusion.dst_hi_row0;
    }
    [[maybe_unused]] const float * x_bias    = nullptr;
    [[maybe_unused]] const float * gate_bias = nullptr;
    if constexpr (has_fusion) {
        if (fusion.x_bias) {
            x_bias = (const float *) fusion.x_bias + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
        }
    }
    if constexpr (has_fusion) {
        if (use_gate && fusion.gate_bias) {
            gate_bias = (const float *) fusion.gate_bias + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
        }
    }
#pragma unroll
    for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = row0 + g + (l >> 1)*8;
            const int col = nt*8 + 2*c + (l & 1);
            if (row >= nrows_x || col >= ncols_dst) {
                continue;
            }
            float result = acc[nt][l];
            if constexpr (has_fusion) {
                if (x_bias) {
                    result += x_bias[col*stride_col_dst + row];
                }
            }
            if constexpr (has_fusion) {
                if (use_gate) {
                    float gate_value = accg[nt][l];
                    if (gate_bias) {
                        gate_value += gate_bias[col*stride_col_dst + row];
                    }
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            d[col*d_stride + row - d_row0] = result;
        }
    }
#else
    GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, row0, channel_dst, sample_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

template <int ntiles, int nwarps, bool has_fusion, bool ilv>
__launch_bounds__(nwarps*WARP_SIZE, (ntiles == 1 && nwarps == 4) ? (has_fusion ? 5 : 6) : 2)
static __global__ void mul_mat_ptq1_mma(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    ggml_cuda_pdl_sync();
    mul_mat_ptq1_mma_tile<ntiles, nwarps, has_fusion, ilv>(vx, vy, fusion, dst,
        ncols_x, nrows_x, ncols_dst,
        stride_row_x, stride_col_y, stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, (int) blockIdx.x * 16, blockIdx.y, blockIdx.z);
}

// The staged kernel shares each activation block across row warps while K groups split work inside the CTA.

template <int ntiles, int nwr, int nwk, bool has_fusion>
__launch_bounds__(nwr*nwk*WARP_SIZE, 1)
static __global__ void mul_mat_ptq1_mma2(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
#if defined(TURING_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    constexpr int ncp  = ntiles*8;                               // padded columns
    constexpr int nacc = ntiles*4*(has_fusion ? 2 : 1);
    // staged activations of one k-block per k-group: [nwk][4 slices][ncp] q8_1 blocks. 36-byte stride = 9 words (odd),
    // so the 32 lanes' B reads (column g, int c) land on 32 distinct banks. After the K loop the same bytes hold the
    // k-group reduction.
    constexpr int ys_bytes  = nwk*4*ncp*(int) sizeof(block_q8_1);
    constexpr int red_bytes = nwk > 1 ? (nwk - 1)*nwr*nacc*WARP_SIZE*(int) sizeof(float) : 0;
    __shared__ __align__(16) char smem[ys_bytes > red_bytes ? ys_bytes : red_bytes];
    block_q8_1 (*ys)[4][ncp] = (block_q8_1 (*)[4][ncp]) smem;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int wr   = w % nwr;                                    // this warp's row tile
    const int wk   = w / nwr;                                    // this warp's k-group
    const int g    = lane >> 2;
    const int c    = lane & 3;
    const int row0 = (blockIdx.x*nwr + wr)*16;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    ggml_cuda_pdl_sync();

    const int nblocks = ncols_x / QK_PTQ1_0;
    const int ra = min(row0 + g,     nrows_x - 1);
    const int rb = min(row0 + g + 8, nrows_x - 1);
    // ILV16 (common.cuh): a whole 16-row tile keeps row g of block kb at tile + kb*16 + g; a partial tile is row-major
    const bool tile_ilv = row0 + 16 <= (nrows_x & ~15);
    const int  kstep    = tile_ilv ? 16 : 1;
    const int  oa       = tile_ilv ? row0*stride_row_x + g     : ra*stride_row_x;
    const int  ob       = tile_ilv ? row0*stride_row_x + g + 8 : rb*stride_row_x;
    const int x_off = sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_ptq1_0 * xa = (const block_ptq1_0 *) vx + x_off + oa;
    const block_ptq1_0 * xb = (const block_ptq1_0 *) vx + x_off + ob;
    [[maybe_unused]] const block_ptq1_0 * ga = nullptr;
    [[maybe_unused]] const block_ptq1_0 * gb = nullptr;
    [[maybe_unused]] bool use_gate = false;
    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        if (use_gate) {
            ga = (const block_ptq1_0 *) fusion.gate + x_off + oa;
            gb = (const block_ptq1_0 *) fusion.gate + x_off + ob;
        }
    }
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    float acc[ntiles][4] = {{0.0f}};
    [[maybe_unused]] float accg[ntiles][4] = {{0.0f}};

    const int niter = (nblocks + nwk - 1) / nwk;                  // every k-group runs the same trip count
    // weights of this warp's first block, then always one block ahead (the loads overlap the staging + barrier)
    ptq1_words nxa = {0, 0, 0}, nxb = {0, 0, 0};
    [[maybe_unused]] ptq1_words nga = {0, 0, 0}, ngb = {0, 0, 0};
    if (wk < nblocks) {
        nxa = ptq1_load_words(xa + wk*kstep, c);
        nxb = ptq1_load_words(xb + wk*kstep, c);
        if constexpr (has_fusion) {
            if (use_gate) {
                nga = ptq1_load_words(ga + wk*kstep, c);
                ngb = ptq1_load_words(gb + wk*kstep, c);
            }
        }
    }
    for (int it = 0; it < niter; ++it) {
        const int  kb     = it*nwk + wk;
        const bool active = kb < nblocks;
        const ptq1_words cxa = nxa, cxb = nxb;
        [[maybe_unused]] const ptq1_words cga = nga, cgb = ngb;
        const int kn = kb + nwk;
        if (kn < nblocks) {
            nxa = ptq1_load_words(xa + kn*kstep, c);
            nxb = ptq1_load_words(xb + kn*kstep, c);
            if constexpr (has_fusion) {
                if (use_gate) {
                    nga = ptq1_load_words(ga + kn*kstep, c);
                    ngb = ptq1_load_words(gb + kn*kstep, c);
                }
            }
        }
        // stage this k-group's k-block by the nwr warps of the group: per column its 4 q8_1 blocks are 36 contiguous
        // words in global memory (read in order, coalesced), scattered into the conflict-free [slice][column] layout
        {
            constexpr int nwords = ncp*36;
            for (int i = wr*WARP_SIZE + lane; i < nwords; i += nwr*WARP_SIZE) {
                const int col  = i / 36;
                const int r    = i - col*36;
                const int s    = r / 9;
                const int word = r - s*9;
                int v = 0;
                if (active && col < ncols_dst) {
                    v = ((const int *) (y + col*stride_col_y + kb*(QK_PTQ1_0/QK8_1)))[r];
                }
                ((int *) &ys[wk][s][col])[word] = v;
            }
        }
        __syncthreads();
        if (active) {
            ptq1_frag fx;
            ptq1_decode_words(cxa, cxb, c, fx);
            const float da = ptq1_word_scale(cxa.hd);
            const float db = ptq1_word_scale(cxb.hd);
            [[maybe_unused]] ptq1_frag fg;
            [[maybe_unused]] float dga = 0.0f, dgb = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    ptq1_decode_words(cga, cgb, c, fg);
                    dga = ptq1_word_scale(cga.hd);
                    dgb = ptq1_word_scale(cgb.hd);
                }
            }
            float blk[ntiles][4] = {{0.0f}};
            [[maybe_unused]] float blkg[ntiles][4] = {{0.0f}};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                tile_A A;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    A.x[l] = fx.s[s][l];
                }
                [[maybe_unused]] tile_A Ag;
                if constexpr (has_fusion) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        Ag.x[l] = fg.s[s][l];
                    }
                }
#pragma unroll
                for (int nt = 0; nt < ntiles; ++nt) {
                    const block_q8_1 & yb = ys[wk][s][nt*8 + g];     // column g of this n-tile (zeros past ncols)
                    tile_B B;
                    B.x[0] = get_int_b4(yb.qs, c);
                    B.x[1] = get_int_b4(yb.qs, 4 + c);
                    const float d8_0 = __low2float(ys[wk][s][nt*8 + 2*c    ].ds);
                    const float d8_1 = __low2float(ys[wk][s][nt*8 + 2*c + 1].ds);
                    tile_C C;
                    mma(C, A, B);
                    blk[nt][0] += d8_0 * (float) C.x[0];
                    blk[nt][1] += d8_1 * (float) C.x[1];
                    blk[nt][2] += d8_0 * (float) C.x[2];
                    blk[nt][3] += d8_1 * (float) C.x[3];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tile_C Cg;
                            mma(Cg, Ag, B);
                            blkg[nt][0] += d8_0 * (float) Cg.x[0];
                            blkg[nt][1] += d8_1 * (float) Cg.x[1];
                            blkg[nt][2] += d8_0 * (float) Cg.x[2];
                            blkg[nt][3] += d8_1 * (float) Cg.x[3];
                        }
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
                acc[nt][0] += da * blk[nt][0];
                acc[nt][1] += da * blk[nt][1];
                acc[nt][2] += db * blk[nt][2];
                acc[nt][3] += db * blk[nt][3];
                if constexpr (has_fusion) {
                    accg[nt][0] += dga * blkg[nt][0];
                    accg[nt][1] += dga * blkg[nt][1];
                    accg[nt][2] += dgb * blkg[nt][2];
                    accg[nt][3] += dgb * blkg[nt][3];
                }
            }
        }
        __syncthreads();                                         // the next stage overwrites ys
    }

    // k-group reduction: groups 1..nwk-1 hand their partials to group 0 (same row tile) through the reused smem
    if constexpr (nwk > 1) {
        float (*red)[nwr][nacc][WARP_SIZE] = (float (*)[nwr][nacc][WARP_SIZE]) smem;
        if (wk > 0) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    red[wk - 1][wr][nt*4 + l][lane] = acc[nt][l];
                    if constexpr (has_fusion) {
                        red[wk - 1][wr][ntiles*4 + nt*4 + l][lane] = accg[nt][l];
                    }
                }
            }
        }
        __syncthreads();
        if (wk > 0) {
            return;
        }
#pragma unroll
        for (int v = 0; v < nwk - 1; ++v) {
#pragma unroll
            for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[nt][l] += red[v][wr][nt*4 + l][lane];
                    if constexpr (has_fusion) {
                        accg[nt][l] += red[v][wr][ntiles*4 + nt*4 + l][lane];
                    }
                }
            }
        }
    }

    float * d        = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    int     d_stride = stride_col_dst;
    int     d_row0   = 0;
    if (fusion.dst_hi != nullptr && row0 >= fusion.dst_hi_row0) {   // sibling merge: this 16-row tile is B's (2D only)
        d        = fusion.dst_hi;
        d_stride = fusion.dst_hi_stride_col;
        d_row0   = fusion.dst_hi_row0;
    }
    [[maybe_unused]] const float * x_bias    = nullptr;
    [[maybe_unused]] const float * gate_bias = nullptr;
    if constexpr (has_fusion) {
        if (fusion.x_bias) {
            x_bias = (const float *) fusion.x_bias + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
        }
        if (use_gate && fusion.gate_bias) {
            gate_bias = (const float *) fusion.gate_bias + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
        }
    }
#pragma unroll
    for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = row0 + g + (l >> 1)*8;
            const int col = nt*8 + 2*c + (l & 1);
            if (row >= nrows_x || col >= ncols_dst) {
                continue;
            }
            float result = acc[nt][l];
            if constexpr (has_fusion) {
                if (x_bias) {
                    result += x_bias[col*stride_col_dst + row];
                }
                if (use_gate) {
                    float gate_value = accg[nt][l];
                    if (gate_bias) {
                        gate_value += gate_bias[col*stride_col_dst + row];
                    }
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            d[col*d_stride + row - d_row0] = result;
        }
    }
#else
    GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

template <int ntiles, int nwr, bool has_fusion>
static void launch_mul_mat_ptq1_mma2_t(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    constexpr int nwk = 8 / nwr;                                 // 8 warps per CTA
    const dim3 block_nums((nrows_x + 16*nwr - 1) / (16*nwr), nchannels_dst, nsamples_dst);
    const dim3 block_dims(WARP_SIZE, nwr*nwk, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_ptq1_mma2<ntiles, nwr, nwk, has_fusion>, launch_params,
        vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
}

template <int ntiles, bool has_fusion>
static void launch_mul_mat_ptq1_mma2_rows(const int nwr, const void * vx, const void * vy,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
#define PTQ1_MMA2_ROWS(NWR) launch_mul_mat_ptq1_mma2_t<ntiles, NWR, has_fusion>(vx, vy, fusion, dst, ncols_x, nrows_x, \
        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, \
        stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, \
        nsamples_dst, stream)
    switch (nwr) {
        case 8:  PTQ1_MMA2_ROWS(8); break;
        case 4:  PTQ1_MMA2_ROWS(4); break;
        case 2:  PTQ1_MMA2_ROWS(2); break;
        default: PTQ1_MMA2_ROWS(1); break;
    }
#undef PTQ1_MMA2_ROWS
}

// Use as many row tiles per CTA as preserve at least two CTAs per SM.
static int ptq1_mma2_nwr(const int nrows_x, const int nsm) {
    constexpr int min_ctas_per_sm = 2;
    const int tiles = (nrows_x + 15) / 16;
    for (const int nwr : {8, 4, 2}) {
        if (tiles / nwr >= min_ctas_per_sm*nsm) {
            return nwr;
        }
    }
    return 1;
}

template <int ntiles, bool has_fusion>
static void launch_mul_mat_ptq1_mma_t(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    constexpr int nwarps = 4;
    const dim3 block_nums((nrows_x + 15) / 16, nchannels_dst, nsamples_dst);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_ptq1_mma<ntiles, nwarps, has_fusion, true>, launch_params,
        vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static void launch_mul_mat_ptq1_mma(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio_fd, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio_fd, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    GGML_ASSERT(ncols_dst >= 1 && ncols_dst <= ptq1_mma_max_cols);
    GGML_ASSERT(ncols_x % QK_PTQ1_0 == 0);
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    GGML_ASSERT(fusion.x_scale == nullptr && fusion.gate_scale == nullptr);
    const int version = ptq1_mma_version(ncols_dst, nrows_x);
    if (version != 1) {
        const int nwr = ptq1_mma2_nwr(nrows_x, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm);
#define PTQ1_MMA2_LAUNCH(NT) \
        if (has_fusion) { launch_mul_mat_ptq1_mma2_rows<NT, true >(nwr, vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, \
            stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, stream); } \
        else            { launch_mul_mat_ptq1_mma2_rows<NT, false>(nwr, vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, \
            stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, stream); }
        switch ((ncols_dst + 7) / 8) {
            case 1:  PTQ1_MMA2_LAUNCH(1); break;
            case 2:  PTQ1_MMA2_LAUNCH(2); break;
            default: PTQ1_MMA2_LAUNCH(4); break;
        }
#undef PTQ1_MMA2_LAUNCH
        return;
    }
#define PTQ1_MMA_LAUNCH(NT) \
    if (has_fusion)      { launch_mul_mat_ptq1_mma_t<NT, true >(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, \
        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, \
        nchannels_dst, nsamples_dst, stream); } \
    else                 { launch_mul_mat_ptq1_mma_t<NT, false>(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, \
        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, \
        nchannels_dst, nsamples_dst, stream); }
    switch ((ncols_dst + 7) / 8) {
        case 1:  PTQ1_MMA_LAUNCH(1); break;
        case 2:  PTQ1_MMA_LAUNCH(2); break;
        default: PTQ1_MMA_LAUNCH(4); break;
    }
#undef PTQ1_MMA_LAUNCH
}
