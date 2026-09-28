// The engine's drafter draw: one row of the compact head's logits (<= 32768 columns) -> the exact top_k under the CPU's
// order (logit desc, id asc), top-p, and the coupled Gumbel draw keyed by key2 -- the host sampler's chain step for step
// (common_topk_scan::run_mapped, llama_sampler_top_p_apply, llama_sampler_dist_apply with llama_coupled_gumbel), so the
// drafts are the ones the CPU would draw. One CTA of 1024 threads: the row in registers (32 columns per thread), a
// bisection over the bits of the top_k-th largest thread maximum (every column of the exact top-k lies at or above it),
// the candidates at or above it ranked by counting under the CPU's key, the tail on warp 0 with the CPU's in-order sums.
// The fork's k_draft_sample_v3, statement for statement (bit-identical records, tools/draw_test.cu).
#include "common.cuh"

#include "kernels.h"

namespace eng {

namespace {

constexpr int DR_THREADS = 1024;
constexpr int DR_PER     = 32;            // columns per thread
constexpr int DR_CMAX    = DR_THREADS;    // top_k distinct thread maxima admit at most 32*top_k <= 1024 candidates
constexpr int DR_KMAX    = GGML_DRAFT_SAMPLE_MAX_K;

// -0 keys as +0 (the CPU compares floats: the two tie and the smaller id wins); then the bits in float order
__device__ __forceinline__ uint32_t dr_orderable(float v) {
    const uint32_t u = __float_as_uint(v) == 0x80000000u ? 0u : __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float dr_from_orderable(uint32_t o) {
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
}

__device__ __forceinline__ uint64_t dr_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// llama_coupled_gumbel: the same bits, in double
__device__ float dr_gumbel(uint64_t key, int32_t id) {
    const uint64_t h = dr_splitmix64(key ^ dr_splitmix64((uint64_t) (uint32_t) id));
    const double   u = ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return (float) -log(-log(u));
}

// correctly rounded in all but double-rounding corners
__device__ __forceinline__ float dr_expf(float x) {
    return (float) exp((double) x);
}

// the chain as the CPU runs it, on one full warp: lane i < top_k holds the i-th strongest candidate's key (0 = none);
// the sums the CPU does in order are gathered and done in order. Writes the record's candidates and out[0..2]
// (out[3] = 0); returns the selected candidate's index (warp-uniform), -1 when fewer than top_k finite logits exist.
__device__ int dr_tail(const uint64_t mine, const int lane, const int top_k, const float top_p,
                       const uint32_t * __restrict__ key2, int32_t * __restrict__ out, int32_t & sel_tok) {
    const bool    have = lane < top_k && mine != 0;
    const int32_t id   = have ? (int32_t) (0xFFFFFFFFu - (uint32_t) (mine & 0xFFFFFFFFull)) : -1;
    const float   lg   = have ? dr_from_orderable((uint32_t) (mine >> 32)) : -INFINITY;
    const float   lg0  = __shfl_sync(0xffffffffu, lg, 0);
    const bool    full = __all_sync(0xffffffffu, lane >= top_k || have);

    const float e = have ? dr_expf(lg - lg0) : 0.0f;   // softmax numerators: top-p's and the dist's are equal
    int kept = top_k;
    float cum_sum = 0.0f;
    if (top_p < 1.0f) {
        for (int i = 0; i < top_k; ++i) {
            cum_sum += __shfl_sync(0xffffffffu, e, i);
        }
        const float pn = have ? e / cum_sum : 0.0f;
        float cum = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cum += __shfl_sync(0xffffffffu, pn, i);
            if (cum >= top_p) {
                kept = i + 1;
                break;   // every lane walks the same values: the loop exits uniformly
            }
        }
    }

    int   sel = 0;
    float pr  = 0.0f;
    if (kept == 1) {
        pr = lane == 0 ? 1.0f : 0.0f;
    } else {
        double sum_cum = 0.0;
        for (int i = 0; i < kept; ++i) {
            sum_cum += (double) __shfl_sync(0xffffffffu, e, i);
        }
        const uint64_t key = (uint64_t) key2[0] | ((uint64_t) key2[1] << 32);
        const float    s   = lane < kept ? lg + dr_gumbel(key, id) : -INFINITY;
        float bestv = -INFINITY;
        for (int i = 0; i < kept; ++i) {   // strict >: the first maximum wins, as on the CPU
            const float si = __shfl_sync(0xffffffffu, s, i);
            if (si > bestv) {
                bestv = si;
                sel   = i;
            }
        }
        pr = lane < kept ? (float) ((double) e / sum_cum) : 0.0f;
    }

    if (lane < top_k) {
        out[4 + lane]              = full ? id : 0;
        out[4 + DR_KMAX + lane]    = full ? __float_as_int(lg) : 0;
        out[4 + 2*DR_KMAX + lane]  = full ? __float_as_int(pr) : 0;
    }
    sel_tok = __shfl_sync(0xffffffffu, id, sel);
    const int32_t id1 = __shfl_sync(0xffffffffu, id, 1);
    if (lane == 0) {
        out[0] = full ? sel_tok : -1;
        out[1] = full ? kept    : -1;
        out[2] = full && kept > 1 ? id1 : -1;
        out[3] = 0;
    }
    return full ? sel : -1;
}

__global__ void __launch_bounds__(DR_THREADS, 1)
k_draw(const float * __restrict__ x, const int32_t * __restrict__ col_ids, const uint32_t * __restrict__ key2,
       int32_t * __restrict__ out, const int32_t * __restrict__ step, int32_t * __restrict__ rec, const int64_t rec_nb1,
       int32_t * __restrict__ col_out, const int n, const int top_k, const float top_p) {
    __shared__ uint64_t cand_key[DR_CMAX];
    __shared__ int32_t  cand_col[DR_CMAX];
    __shared__ uint64_t top_key[DR_KMAX];
    __shared__ int32_t  top_col[DR_KMAX];
    __shared__ int      n_cand;

    const int t    = threadIdx.x;
    const int lane = t & 31;

    // 1: column t + j*DR_THREADS as an orderable key in v[j]; NaN and -inf become 0 and are never kept
    uint32_t v[DR_PER];
    uint32_t vmax = 0;
#pragma unroll
    for (int j = 0; j < DR_PER; ++j) {
        const int   c = t + j*DR_THREADS;
        const float f = c < n ? x[c] : -INFINITY;
        v[j] = f > -INFINITY ? dr_orderable(f) : 0u;
        vmax = max(vmax, v[j]);
    }
    if (t == 0) {
        n_cand = 0;
    }
    if (t < DR_KMAX) {
        top_key[t] = 0;
        top_col[t] = 0;
    }

    // 2: the bar, bit by bit: keep a bit while at least top_k thread maxima reach it (the count is block-uniform)
    uint32_t bar = 0;
    for (int b = 31; b >= 0; --b) {
        const uint32_t probe = bar | (1u << b);
        const int      cnt   = __syncthreads_count(vmax >= probe);
        if (cnt >= top_k) {
            bar = probe;
            if (cnt == top_k) {
                break;
            }
        }
    }

    // 3: the candidates: every finite column at or above the bar
    const uint32_t lo = bar > 0 ? bar : 1u;
#pragma unroll
    for (int j = 0; j < DR_PER; ++j) {
        if (v[j] >= lo) {
            const int slot = atomicAdd(&n_cand, 1);
            if (slot < DR_CMAX) {
                const int c = t + j*DR_THREADS;
                cand_key[slot] = ((uint64_t) v[j] << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) col_ids[c]);
                cand_col[slot] = c;
            }
        }
    }
    __syncthreads();

    // 4: rank by counting (keys are unique, as the ids are): the top_k strongest land in order. Past DR_CMAX nothing is
    //    ranked and the record reads "not full": the host stops the draft.
    const int m = n_cand;
    if (m <= DR_CMAX && t < m) {
        const uint64_t mine = cand_key[t];
        int rank = 0;
        for (int i = 0; i < m; ++i) {
            rank += cand_key[i] > mine;
        }
        if (rank < top_k) {
            top_key[rank] = mine;
            top_col[rank] = cand_col[t];
        }
    }
    __syncthreads();

    // 5: the tail on warp 0; the selected candidate's column is at hand
    if (t < 32) {
        int32_t sel_tok;
        const int sel = dr_tail(lane < top_k ? top_key[lane] : 0, lane, top_k, top_p, key2, out, sel_tok);
        const int32_t c = sel >= 0 ? top_col[sel] : 0;
        if (lane == 0) {
            out[3] = c;
            if (col_out != nullptr) {
                col_out[0] = c;
            }
        }
        __syncwarp();   // the record row copies what every lane wrote
        if (rec != nullptr) {
            int32_t * row = (int32_t *) ((char *) rec + (int64_t) step[0]*rec_nb1);
            for (int i = lane; i < GGML_DRAFT_SAMPLE_OUT; i += 32) {
                row[i] = out[i];
            }
        }
    }
}

} // namespace

void draft_draw(cudaStream_t st, const float * logits, const int32_t * col_ids, const uint32_t * key2, int32_t * out,
                const int32_t * step, int32_t * rec, int64_t rec_nb1, int32_t * col_out, int n, int top_k, float top_p) {
    if (n > DR_THREADS*DR_PER || top_k < 1 || top_k > DR_KMAX) {
        fprintf(stderr, "draft_draw: %d columns, top_k %d (the draw handles <= %d columns, top_k <= %d)\n", n, top_k,
                DR_THREADS*DR_PER, DR_KMAX);
        abort();
    }
    k_draw<<<1, DR_THREADS, 0, st>>>(logits, col_ids, key2, out, step, rec, rec_nb1, col_out, n, top_k, top_p);
}

} // namespace eng
