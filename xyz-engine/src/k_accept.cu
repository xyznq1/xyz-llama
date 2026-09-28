// xyz-engine: the accept step on the device, bit-exact with the server's host path (see accept.h). NO fast math, NO
// FMA contraction in this translation unit (CMake target xe_exact): every float/double operation below is one IEEE
// operation in the host's order.
#include "accept.h"

#include <cstdio>
#include <cstdlib>

#define CK_ACC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

namespace xe {

// ---- std::mt19937 --------------------------------------------------------------------------------------------------
static __device__ uint32_t mt_next(Mt19937 & s) {
    if (s.idx >= 624) {
        for (int i = 0; i < 624; ++i) {
            const uint32_t y = (s.mt[i] & 0x80000000u) | (s.mt[(i + 1) % 624] & 0x7fffffffu);
            s.mt[i] = s.mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        s.idx = 0;
    }
    uint32_t y = s.mt[s.idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// std::uniform_real_distribution<double>(0, 1) of MSVC 14.44: generate_canonical<double, 53> for a 32-bit engine --
// Sx = (g0 >> 11) + (g1 << 21), times the float 2^-53 -- then * (max - min) + min
static __device__ double uni01(Mt19937 & s) {
    const uint32_t g0 = mt_next(s);
    const uint32_t g1 = mt_next(s);
    const uint64_t sx = (uint64_t) (g0 >> 11) + ((uint64_t) g1 << 21);
    const double   c  = (double) sx * (double) (1.0f / (float) (1ull << 53));
    return c * (1.0 - 0.0) + 0.0;
}

// ---- MSVC's expf ---------------------------------------------------------------------------------------------------
// (float) exp((double) x) is MSVC's expf(x) except at the table's inputs (sorted pairs: x bits, MSVC result bits)
static __device__ float msvc_expf(const float x, const uint32_t * __restrict__ exc, const int n_exc) {
    const uint32_t u = __float_as_uint(x);
    if (u >= 0x80000000u && u <= 0xC2D00000u) {
        int lo = 0, hi = n_exc - 1;
        while (lo <= hi) {
            const int      mid = (lo + hi) >> 1;
            const uint32_t k   = exc[2*mid];
            if (k == u) {
                return __uint_as_float(exc[2*mid + 1]);
            }
            if (k < u) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    return (float) exp((double) x);
}

// ---- the samplers (llama-sampler.cpp), candidates sorted strongest first ----------------------------------------
// top_p: llama_sampler_softmax_impl (float exps, float sum, float division) then the float running sum; returns the
// kept count
static __device__ int top_p_cut(const float * ex, const int n, const float p_thr, const int min_keep, float * pb) {
    if (p_thr >= 1.0f) {
        return n;
    }
    float cum_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float p = ex[i];
        pb[i] = p;
        cum_sum += p;
    }
    for (int i = 0; i < n; ++i) {
        pb[i] /= cum_sum;
    }
    float cs = 0.0f;
    for (int i = 0; i < n; ++i) {
        cs += pb[i];
        if (cs >= p_thr && i + 1 >= min_keep) {
            return i + 1;
        }
    }
    return n;
}

// llama_coupled_gumbel (llama-sampler.cpp): splitmix of the key and the id, u in (0, 1) exclusive, -log(-log(u)) in
// double, then float -- the device draw's ds_gumbel (ggml-cuda/top-k.cu), the same bits
static __device__ __forceinline__ uint64_t cp_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static __device__ float cp_gumbel(const uint64_t key, const int32_t id) {
    const uint64_t h = cp_splitmix64(key ^ cp_splitmix64((uint64_t) (uint32_t) id));
    const double   u = ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return (float) -log(-log(u));
}

static __device__ double prob_of(const int32_t * ids, const float * p, const int n, const int32_t t) {
    for (int i = 0; i < n; ++i) {
        if (ids[i] == t) {
            return p[i];
        }
    }
    return 0.0;
}

// common_sampler_sample_and_accept_n_block, relax_tau 0. Order-bound sums stay on one thread; independent exponentials
// and Gumbel keys are computed in parallel.
static __global__ void k_accept(const AcceptParams prm, const int32_t * __restrict__ t_ids, const float * __restrict__ t_logit,
                                const int32_t * __restrict__ d_ids, const float * __restrict__ d_logit,
                                const int32_t * __restrict__ draft, Mt19937 * __restrict__ rng,
                                const uint32_t * __restrict__ exc, const int n_exc, AcceptOut * __restrict__ out,
                                const uint32_t * __restrict__ keys2) {
    const int G = prm.G, tk = prm.tk;
    const bool plain = prm.mode == ACC_PLAIN;
    // 1. every exp in parallel: expf(l_i - l_0) per candidate of each target row and each draft (each value is one
    //    function of one input, so the order it is computed in cannot matter)
    __shared__ float ex_t[ACC_MAX_G + 1][ACC_MAX_TK], ex_d[ACC_MAX_G][ACC_MAX_TK];
    for (int t = threadIdx.x; t < (2*G + 1)*tk; t += blockDim.x) {
        const int j = t / tk, i = t % tk;
        if (j <= G) {
            ex_t[j][i] = msvc_expf(t_logit[j*tk + i] - t_logit[j*tk], exc, n_exc);
        } else {
            const int d = j - G - 1;
            ex_d[d][i] = msvc_expf(d_logit[d*tk + i] - d_logit[d*tk], exc, n_exc);
        }
    }
    // 1b. The Gumbel keys: the chain re-check's per draft j < G over d_ids, the plain draw's per row j <= G over
    //     t_ids (= p_id), all tk candidates (the kept counts come later; the extras go unused)
    __shared__ float gum_d[ACC_MAX_G][ACC_MAX_TK], gum_t[ACC_MAX_G + 1][ACC_MAX_TK];
    if (keys2 != nullptr) {
        const int n_d = G*tk, n_t = plain ? (G + 1)*tk : 0;
        for (int t = threadIdx.x; t < n_d + n_t; t += blockDim.x) {
            const bool dr = t < n_d;
            const int  u  = dr ? t : t - n_d;
            const int  j  = u / tk, i = u % tk;
            const uint64_t key = (uint64_t) keys2[2*j] | ((uint64_t) keys2[2*j + 1] << 32);
            if (dr) {
                gum_d[j][i] = cp_gumbel(key, d_ids[j*tk + i]);
            } else {
                gum_t[j][i] = cp_gumbel(key, t_ids[j*tk + i]);
            }
        }
    }
    __syncthreads();
    // 2a. The rows, one thread each (they are independent; inside a row every float/double sum stays in the host's
    //     order): each target row's and each draft's top_p cut and dist sum (dist_probs: sum_cum += e in double)
    __shared__ int    n_t[ACC_MAX_G + 1], n_d[ACC_MAX_G];
    __shared__ double sum_t[ACC_MAX_G + 1], sum_d[ACC_MAX_G];
    if (threadIdx.x < 2*G + 1) {
            const int  r  = threadIdx.x;
            const bool tr = r <= G;
            const int  j  = tr ? r : r - G - 1;
            const float * ex = tr ? ex_t[j] : ex_d[j];
            float pb[ACC_MAX_TK];
            const int n = top_p_cut(ex, tk, tr ? prm.top_p_t : prm.top_p_d, tr ? prm.min_keep_t : prm.min_keep_d, pb);
            double s = 0.0;
            if (n != 1) {
                for (int i = 0; i < n; ++i) {
                    s += ex[i];
                }
            }
            if (tr) { n_t[j] = n; sum_t[j] = s; } else { n_d[j] = n; sum_d[j] = s; }
    }
    __syncthreads();
    // 2b. Every kept candidate's probability in parallel -- (float) ((double) e / sum), a lone candidate 1.0f -- with
    //     its id and the counts; the drafts' only in block mode (the plain path never wrote them)
    const int n_rows = plain ? G + 1 : 2*G + 1;
    for (int t = threadIdx.x; t < n_rows*tk; t += blockDim.x) {
            const int  r  = t / tk, i = t % tk;
            const bool tr = r <= G;
            const int  j  = tr ? r : r - G - 1;
            const int  n  = tr ? n_t[j] : n_d[j];
            if (i >= n) {
                continue;
            }
            const float e = tr ? ex_t[j][i] : ex_d[j][i];
            const float p = n == 1 ? 1.0f : (float) ((double) e / (tr ? sum_t[j] : sum_d[j]));
            if (tr) {
                out->p_p[j][i]  = p;
                out->p_id[j][i] = t_ids[j*tk + i];
            } else {
                out->q_p[j][i]  = p;
                out->q_id[j][i] = d_ids[j*tk + i];
            }
    }
    if (threadIdx.x <= (unsigned) G) {
        out->p_n[threadIdx.x] = n_t[threadIdx.x];
    }
    if (!plain && threadIdx.x < (unsigned) G) {
        out->q_n[threadIdx.x] = n_d[threadIdx.x];
    }
    __syncthreads();
    // 2. the order-bound part on one thread: every sum in the host's order
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    out->one_mask = 0;
    // the server's host re-check of the device draft chain (common/speculative.cpp draft_device_chain): each draw's chain
    // re-run on its record with the host's math -- the draft ends at the first draw it does not reproduce
    out->cut = -1;
    if (keys2 != nullptr) {
        for (int j = 0; j < G && out->cut < 0; ++j) {
            const int nq = n_d[j];
            int sel = 0;
            if (nq > 1) {
                float best = -INFINITY;
                for (int i = 0; i < nq; ++i) {
                    const float s = d_logit[j*tk + i] + gum_d[j][i];
                    if (s > best) {
                        best = s;
                        sel  = i;
                    }
                }
            }
            if (d_ids[j*tk + sel] != draft[j]) {
                out->cut = j;
            }
        }
    }
    if (plain) {
        // common_sampler_sample_and_accept_n: row j (armed at coupled_pos0 + j) draws the argmax of logit + gumbel over
        // its kept candidates (strict >, the first maximum wins; one candidate: it, with a draw from the dist's own
        // rng on the host); a draw that differs from draft j ends the round, as does the bonus row G
        int n_out = 0;
        for (int j = 0; j <= G; ++j) {
            const int n = out->p_n[j];
            int sel = 0;
            if (n == 1) {
                out->one_mask |= 1 << j;
            } else {
                float best = -INFINITY;
                for (int i = 0; i < n; ++i) {
                    const float s = t_logit[j*tk + i] + gum_t[j][i];
                    if (s > best) {
                        best = s;
                        sel  = i;
                    }
                }
            }
            const int32_t t = out->p_id[j][sel];
            out->tokens[j] = t;
            n_out = j + 1;
            if (j == G || t != draft[j]) {
                break;
            }
        }
        out->n = n_out;
        return;
    }
    const auto P = [&](int j, int32_t t) { return prob_of(out->p_id[j], out->p_p[j], out->p_n[j], t); };
    const auto Q = [&](int j, int32_t t) { return prob_of(out->q_id[j], out->q_p[j], out->q_n[j], t); };

    double w[ACC_MAX_G + 1], h[ACC_MAX_G + 1];
    w[0] = 1.0;
    for (int j = 0; j < G; ++j) {
        const double q = Q(j, draft[j]);
        w[j + 1] = q > 0.0 ? fmin(w[j] * P(j, draft[j]) / q, 1.0) : 0.0;
    }
    for (int j = 0; j < G; ++j) {
        double S = 0.0;
        for (int i = 0; i < out->p_n[j]; ++i) {
            const double r = w[j] * (double) out->p_p[j][i] - Q(j, out->p_id[j][i]);
            if (r > 0.0) {
                S += r;
            }
        }
        const double den = S + 1.0 - w[j];
        h[j] = den > 1e-15 ? S / den : 1.0;
    }
    h[G] = w[G];

    int tau = 0;
    for (int j = 1; j <= G; ++j) {
        const double u = uni01(*rng);
        if (u <= h[j]) {
            tau = j;
        }
    }

    // the token after the accepted prefix: the residual w p - q of row tau (row G: p itself), else p
    int32_t res_id[ACC_MAX_TK];
    double  res_r[ACC_MAX_TK];
    int     n_res = 0;
    double  z = 0.0;
    for (int i = 0; i < out->p_n[tau]; ++i) {
        const double r = tau == G ? (double) out->p_p[tau][i] : w[tau] * (double) out->p_p[tau][i] - Q(tau, out->p_id[tau][i]);
        if (r > 0.0) {
            res_id[n_res] = out->p_id[tau][i];
            res_r[n_res]  = r;
            ++n_res;
            z += r;
        }
    }
    if (z <= 0.0) {
        n_res = 0;
        for (int i = 0; i < out->p_n[tau]; ++i) {
            res_id[n_res] = out->p_id[tau][i];
            res_r[n_res]  = (double) out->p_p[tau][i];
            ++n_res;
            z += out->p_p[tau][i];
        }
    }
    const double u0 = uni01(*rng);
    double u = u0 * z;
    int32_t last = n_res == 0 ? draft[0] : res_id[n_res - 1];
    for (int i = 0; i < n_res; ++i) {
        if (u < res_r[i]) {
            last = res_id[i];
            break;
        }
        u -= res_r[i];
    }
    for (int j = 0; j < tau; ++j) {
        out->tokens[j] = draft[j];
    }
    out->tokens[tau] = last;
    out->n = tau + 1;
}

void accept(cudaStream_t st, const AcceptParams & prm, const int32_t * t_ids, const float * t_logit, const int32_t * d_ids,
            const float * d_logit, const int32_t * draft, Mt19937 * rng, const uint32_t * exc, int n_exc, AcceptOut * out,
            const uint32_t * keys2) {
    if (prm.G < 0 || prm.G > ACC_MAX_G || prm.tk < 1 || prm.tk > ACC_MAX_TK || (prm.mode == ACC_PLAIN && keys2 == nullptr)) {
        fprintf(stderr, "accept: G %d tk %d mode %d out of range\n", prm.G, prm.tk, prm.mode);
        abort();
    }
    k_accept<<<1, 256, 0, st>>>(prm, t_ids, t_logit, d_ids, d_logit, draft, rng, exc, n_exc, out, keys2);
}

// ---- exact top-k rows (the host scan's order) ----------------------------------------------------------------------
static __device__ __forceinline__ uint64_t tk_key(const float v, const int32_t id, const int32_t * skip, const int n_skip) {
    const uint32_t b = __float_as_uint(v);
    if (v != v || b == 0xff800000u) {   // NaN or -inf: the scan's `x > thr` (thr starts at -inf) never admits them
        return 0;
    }
    for (int lo = 0, hi = n_skip - 1; lo <= hi;) {
        const int mid = (lo + hi) >> 1;
        if (skip[mid] == id) {
            return 0;
        }
        if (skip[mid] < id) lo = mid + 1; else hi = mid - 1;
    }
    const uint32_t ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t) ord << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) id);   // higher logit, then lower id: stronger
}

// descending bitonic sort of N keys in shared memory, blockDim.x = 1024
template <int N>
static __device__ void bitonic_desc(uint64_t * s) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = (i & k) == 0;
                    const uint64_t a = s[i], b = s[ixj];
                    if (desc ? a < b : a > b) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// one block per row: radix-select the tk-th largest orderable logit T (four 8-bit passes, per-warp histograms), gather every
// key with ord >= T (the ones above T are < tk; ties at T come with their ids), sort the few gathered keys, keep tk
static constexpr int TK_THREADS = 1024;
static constexpr int TK_GATHER  = 1024;

static __device__ __forceinline__ uint32_t tk_ord(const float v, const int32_t id, const int32_t * skip, const int n_skip) {
    const uint32_t b = __float_as_uint(v);
    if (v != v || b == 0xff800000u) {
        return 0;
    }
    for (int lo = 0, hi = n_skip - 1; lo <= hi;) {
        const int mid = (lo + hi) >> 1;
        if (skip[mid] == id) {
            return 0;
        }
        if (skip[mid] < id) lo = mid + 1; else hi = mid - 1;
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);   // > 0 for every admitted value (-FLT_MAX maps to 0x00800000)
}

// Wide exact top-k: (1) 64 blocks per row histogram the top 12 bits of every admitted key (block histogram in shared
// memory, non-zero bins added to the row's global histogram); (2) one thread per row finds the bin b* holding the tk-th
// largest key; (3) 64 blocks per row gather every key whose top 12 bits are >= b* (the top tk are among them); (4) one
// block per row sorts the gathered keys (bitonic over the next power of two) and keeps tk. Keys: ord(logit) << 32 |
// ~id -- the scan's order (logit desc, then id asc); NaN, -inf and skipped ids never enter.
static constexpr int TKW_BLOCKS = 64;     // blocks per row in the two scans
static constexpr int TKW_BINS   = 4096;   // top 12 bits
static constexpr int TKW_CAP    = 4096;   // gathered keys per row (more: flagged, n_ok = -1)

struct TkScratch {
    uint32_t hist[8][TKW_BINS];
    uint32_t bstar[8];
    uint32_t n_cand[8];
    uint64_t cand[8][TKW_CAP];
};

static __global__ void k_tkw_hist(const float * __restrict__ logits, const int n_vocab, const int32_t * __restrict__ skip,
                                  const int n_skip, TkScratch * __restrict__ sc) {
    __shared__ uint32_t h[TKW_BINS];
    const int row = blockIdx.y;
    for (int i = threadIdx.x; i < TKW_BINS; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const float * x = logits + (size_t) row*n_vocab;
    const int per = (n_vocab + gridDim.x - 1) / gridDim.x;
    const int i0 = blockIdx.x*per, i1 = min(n_vocab, i0 + per);
    for (int id = i0 + threadIdx.x; id < i1; id += blockDim.x) {
        const uint32_t o = tk_ord(x[id], id, skip, n_skip);
        if (o != 0) atomicAdd(&h[o >> 20], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TKW_BINS; i += blockDim.x) {
        if (h[i] != 0) atomicAdd(&sc->hist[row][i], h[i]);
    }
}

// the same bin -- the highest b with (sum of the bins >= b) >= tk, else 0 -- from suffix sums: thread t holds bins
// 4t..4t+3 (one 16-byte load), exactly one thread's span holds the crossing and walks it down as the serial loop does.
// (The serial walk crossed ~1000 empty bins above the logits' range one dependent load at a time: ~50 us per call.)
static __global__ void __launch_bounds__(TKW_BINS/4) k_tkw_bstar(const int tk, TkScratch * __restrict__ sc) {
    static_assert(TKW_BINS == 4*1024, "one block of 1024 threads, four bins each");
    __shared__ uint32_t s_warp[32];
    __shared__ int      s_bin;
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const uint4 h4 = ((const uint4 *) sc->hist[row])[t];
    const uint32_t local = h4.x + h4.y + h4.z + h4.w;
    uint32_t v = local;   // -> the sum over this warp's lanes >= lane
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t n = __shfl_down_sync(0xffffffffu, v, o);
        if (lane + o < 32) v += n;
    }
    if (lane == 0) s_warp[wid] = v;
    if (t == 0) s_bin = -1;
    __syncthreads();
    uint32_t above_w = 0;   // the warps above this one
    for (int w2 = wid + 1; w2 < 32; ++w2) above_w += s_warp[w2];
    const uint32_t incl = v + above_w;   // bins >= 4t
    const uint32_t excl = incl - local;  // bins >= 4t + 4
    if (excl < (uint32_t) tk && incl >= (uint32_t) tk) {
        const uint32_t hb[4] = { h4.x, h4.y, h4.z, h4.w };
        uint32_t above = excl;
        int bin = 4*t;
        for (int k = 3; k >= 0; --k) {
            above += hb[k];
            if (above >= (uint32_t) tk) { bin = 4*t + k; break; }
        }
        s_bin = bin;
    }
    __syncthreads();
    if (t == 0) {
        sc->bstar[row]  = s_bin < 0 ? 0 : (uint32_t) s_bin;
        sc->n_cand[row] = 0;
    }
}

static __global__ void k_tkw_gather(const float * __restrict__ logits, const int n_vocab, const int32_t * __restrict__ skip,
                                    const int n_skip, TkScratch * __restrict__ sc) {
    const int row = blockIdx.y;
    const uint32_t bs = sc->bstar[row];
    const float * x = logits + (size_t) row*n_vocab;
    const int per = (n_vocab + gridDim.x - 1) / gridDim.x;
    const int i0 = blockIdx.x*per, i1 = min(n_vocab, i0 + per);
    for (int id = i0 + threadIdx.x; id < i1; id += blockDim.x) {
        const uint32_t o = tk_ord(x[id], id, skip, n_skip);
        if (o != 0 && (o >> 20) >= bs) {
            const uint32_t k = atomicAdd(&sc->n_cand[row], 1u);
            if (k < TKW_CAP) sc->cand[row][k] = ((uint64_t) o << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) id);
        }
    }
}

template <int N>
static __device__ void bitonic_desc_n(uint64_t * s, const int n) {   // n = a power of two <= N
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = (i & k) == 0;
                    const uint64_t a = s[i], b = s[ixj];
                    if (desc ? a < b : a > b) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __global__ void __launch_bounds__(1024) k_tkw_select(const float * __restrict__ logits, const int n_vocab, const int tk,
        TkScratch * __restrict__ sc, int32_t * __restrict__ ids, float * __restrict__ vals, int32_t * __restrict__ n_ok) {
    __shared__ uint64_t g[TKW_CAP];
    const int row = blockIdx.x;
    const uint32_t nc = sc->n_cand[row];
    const int n = (int) min(nc, (uint32_t) TKW_CAP);
    int np2 = 1;
    while (np2 < n || np2 < 2) np2 <<= 1;
    for (int i = threadIdx.x; i < np2; i += blockDim.x) g[i] = i < n ? sc->cand[row][i] : 0;
    __syncthreads();
    bitonic_desc_n<TKW_CAP>(g, np2);
    const float * x = logits + (size_t) row*n_vocab;
    if (threadIdx.x == 0) {
        int ok = nc <= (uint32_t) TKW_CAP ? 0 : -1;
        bool tie = false;   // two kept candidates with the same logit, or the last kept tied with the first left out
        for (int i = 0; i < tk; ++i) {
            const uint64_t k = i < np2 ? g[i] : 0;
            const int32_t id = (int32_t) (0xFFFFFFFFu - (uint32_t) (k & 0xFFFFFFFFu));
            if (ok >= 0) ok += k != 0;
            ids[row*tk + i]  = k != 0 ? id : -1;
            vals[row*tk + i] = k != 0 ? x[id] : -INFINITY;
            if (i > 0 && k != 0 && (k >> 32) == (g[i - 1] >> 32)) tie = true;
        }
        if (n > tk && g[tk] != 0 && (g[tk] >> 32) == (g[tk - 1] >> 32)) tie = true;
        // TK_TIE: the host's full-candidate path (a grammar turns the scan off) orders ties by its partial sort, not by id
        n_ok[row] = ok >= 0 && tie ? (ok | TK_TIE) : ok;
    }
    // the histogram back to zero for the next call
    for (int i = threadIdx.x; i < TKW_BINS; i += blockDim.x) sc->hist[row][i] = 0;
}

size_t topk_rows_scratch(int n_vocab, int rows) {
    (void) n_vocab; (void) rows;
    return sizeof(TkScratch);
}

void topk_rows(cudaStream_t st, const float * logits, int n_vocab, int rows, int tk, const int32_t * skip, int n_skip,
               int32_t * ids, float * vals, int32_t * n_ok, uint64_t * scratch) {
    if (tk > ACC_MAX_TK || rows > 8) {
        fprintf(stderr, "topk_rows: tk %d rows %d out of range\n", tk, rows);
        abort();
    }
    TkScratch * sc = (TkScratch *) scratch;
    static bool zeroed = false;   // the select pass re-zeroes the histograms; the first call needs them zero
    if (!zeroed) {
        CK_ACC(cudaMemsetAsync(sc, 0, sizeof(TkScratch), st));
        zeroed = true;
    }
    k_tkw_hist<<<dim3(TKW_BLOCKS, rows), 256, 0, st>>>(logits, n_vocab, skip, n_skip, sc);
    k_tkw_bstar<<<rows, TKW_BINS/4, 0, st>>>(tk, sc);
    k_tkw_gather<<<dim3(TKW_BLOCKS, rows), 256, 0, st>>>(logits, n_vocab, skip, n_skip, sc);
    k_tkw_select<<<rows, 1024, 0, st>>>(logits, n_vocab, tk, sc, ids, vals, n_ok);
}

} // namespace xe
