#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"
#include "../src/llama-ext.h"   // llama_sampler_grammar_awaiting

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#define COMMON_TOPK_SSE2 1
#endif

// The k best candidates of a whole logits row in the chain's total order (logit descending, then id ascending --
// llama-sampler.cpp llama_token_data_logit_desc), found by a threshold scan over the raw floats instead of building
// n_vocab candidates and partially sorting them: 18 us against 230 us for 248,320 logits and k 20 (x86-64 SSE2),
// same rows in the same order. Ids arrive ascending, so a value
// equal to a kept one is always the weaker of the two and a strict `>` against the k-th value is exact.
// ponytail: a row whose logits RISE with the id inserts at every element (O(n_vocab k), ~5 ms at k 20); model rows
// are not ordered by id, so the scan stays near its 18 us.
struct common_topk_scan {
    std::vector<llama_token_data> top;   // ascending in the total order: top[0] is the weakest kept
    int32_t k = 0;
    int32_t n = 0;

    float thr() const {
        return n < k ? -INFINITY : top[0].logit;
    }

    void push(llama_token id, float v) {
        if (n < k) {
            int32_t p = n++;
            while (p > 0 && top[p - 1].logit >= v) {   // an equal, earlier id is stronger
                top[p] = top[p - 1];
                --p;
            }
            top[p] = { id, v, 0.0f };
            return;
        }
        int32_t p = 1;
        while (p < k && top[p].logit < v) {
            top[p - 1] = top[p];
            ++p;
        }
        top[p - 1] = { id, v, 0.0f };
    }

    // false when the row has fewer than k finite (non-NaN, > -inf) values outside `skip` -- the caller then builds
    // the whole candidate set, where such entries take part in the order
    bool run(const float * x, int32_t n_vocab, int32_t k_, const std::vector<llama_token> & skip) {
        k = k_;
        n = 0;
        top.resize((size_t) k);
        const auto skipped = [&](int32_t id) {
            return !skip.empty() && std::binary_search(skip.begin(), skip.end(), id);
        };
        const auto consider = [&](int32_t id) {
            if (x[id] > thr() && !skipped(id)) {
                push(id, x[id]);
            }
        };
        int32_t i = 0;
#ifdef COMMON_TOPK_SSE2
        __m128 t = _mm_set1_ps(thr());
        for (; i + 4 <= n_vocab; i += 4) {
            const int m = _mm_movemask_ps(_mm_cmpgt_ps(_mm_loadu_ps(x + i), t));
            if (m) {
                for (int b = 0; b < 4; ++b) {
                    if (m >> b & 1) {
                        consider(i + b);
                    }
                }
                t = _mm_set1_ps(thr());
            }
        }
#endif
        for (; i < n_vocab; ++i) {
            consider(i);
        }
        if (n < k) {
            return false;
        }
        std::reverse(top.begin(), top.end());   // strongest first: the order the chain's top-k leaves
        return true;
    }

    // A COMPACT row (XYZ2_COMPACT_LOGITS): x[c] is the logit of id col_ids[c], columns in no id order. run() breaks a
    // tie by scan order, which there IS id order (an equal, smaller id is stronger); here the tie is broken by id
    // explicitly, so both keep the same k candidates in the same order -- as the chain's top-k over id-sorted input.
    bool run_mapped(const float * x, int32_t n_col, const llama_token * col_ids, int32_t k_,
                    const std::vector<llama_token> & skip) {
        k = k_;
        n = 0;
        top.resize((size_t) k);
        const auto consider = [&](int32_t c) {
            const float       v  = x[c];
            const llama_token id = col_ids[c];
            if (!(v > -INFINITY)) {
                return;   // NaN and -inf are never kept (run(): x > thr() fails for them)
            }
            if (n == k && !(v > top[0].logit || (v == top[0].logit && id < top[0].id))) {
                return;
            }
            if (!skip.empty() && std::binary_search(skip.begin(), skip.end(), id)) {
                return;
            }
            if (n < k) {
                int32_t p = n++;
                while (p > 0 && (top[p - 1].logit > v || (top[p - 1].logit == v && top[p - 1].id < id))) {
                    top[p] = top[p - 1];
                    --p;
                }
                top[p] = { id, v, 0.0f };
                return;
            }
            int32_t p = 1;
            while (p < k && (top[p].logit < v || (top[p].logit == v && top[p].id > id))) {
                top[p - 1] = top[p];
                ++p;
            }
            top[p - 1] = { id, v, 0.0f };
        };
        int32_t c = 0;
#ifdef COMMON_TOPK_SSE2
        __m128 t = _mm_set1_ps(thr());
        for (; c + 4 <= n_col; c += 4) {
            const int m = _mm_movemask_ps(_mm_cmpge_ps(_mm_loadu_ps(x + c), t));   // >=: a tie may still win on id
            if (m) {
                for (int b = 0; b < 4; ++b) {
                    if (m >> b & 1) {
                        consider(c + b);
                    }
                }
                t = _mm_set1_ps(thr());
            }
        }
#endif
        for (; c < n_col; ++c) {
            consider(c);
        }
        if (n < k) {
            return false;
        }
        std::reverse(top.begin(), top.end());
        return true;
    }
};

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // coupled speculative selection. Trailing members with default initialisers so the two
    // aggregate constructions of common_sampler keep working untouched.
    bool     coupled_enabled = false;
    uint32_t coupled_seed    = 0;
    int32_t  coupled_seq_id  = 0;
    int32_t  coupled_pos0    = 0;   // absolute position that draft[0] predicts

    // xyz-engine: set_logits reads this row (n_vocab floats: an external engine's verify row) instead of the context's
    // outputs -- the host path over the full row, exactly what the context's row would give
    const float * row_override = nullptr;

    std::vector<llama_token> vocab_subset;   // common_sampler_set_vocab_subset; empty = the whole vocabulary
    std::vector<int32_t>     vocab_subset_idx;   // non-empty: COMPACT logits, vocab_subset[i] is column vocab_subset_idx[i]
    std::vector<llama_token> vocab_subset_col_ids;   // the inverse: column c holds the logit of id vocab_subset_col_ids[c]

    // > 0: the chain provably reads only its top-k before it truncates (common_sampler_topk_scan_k), so a whole-row
    // call builds just those k candidates (common_topk_scan); topk_skip are ids the chain biases to -inf
    int32_t                  topk_scan_k = 0;
    std::vector<llama_token> topk_skip;
    // GRAMMAR FIRST-PASS SCAN. A grammar turns the scan off (topk_scan_k 0): its constrained resample and a
    // grammar_first call filter the WHOLE candidate set. But common_sampler_sample's first draw is unconstrained -- the
    // chain alone, the grammar only checks the token afterwards -- so that draw reads nothing but the chain's top-k,
    // exactly as without a grammar. Building all 248k candidates and partial-sorting them for it cost the tool-carrying
    // requests (every chat request) ~0.5 ms per verify row on the host while the GPU waited. The scan keeps k+1 and the
    // draw uses it only when those k+1 logits are all distinct: then the full path's top-k is the same k candidates in
    // the same order, so the draw is bit for bit the full path's; with a tie it builds the full set as before.
    // COMMON_SAMPLER_GRAMMAR_SCAN=0 turns it off.
    int32_t                  topk_scan_k_gr = 0;
    std::vector<llama_token> topk_skip_gr;
    bool                     scan_first = false;   // set around common_sampler_sample's unconstrained first draw
    common_topk_scan         topk_scan;

    // arm every dist sampler in the chain for one position; the setter ignores non-dist samplers
    void arm_coupled(int32_t i) const {
        if (!coupled_enabled) {
            return;
        }

        const uint64_t key = llama_sampler_coupled_key(coupled_seed, coupled_seq_id, coupled_pos0 + i);

        const int n = llama_sampler_chain_n(chain);
        for (int j = 0; j < n; ++j) {
            llama_sampler_dist_set_coupled(llama_sampler_chain_get(chain, j), true, key);
        }
    }

    void disarm_coupled() const {
        const int n = llama_sampler_chain_n(chain);
        for (int j = 0; j < n; ++j) {
            llama_sampler_dist_set_coupled(llama_sampler_chain_get(chain, j), false, 0);
        }
    }

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = row_override ? nullptr : llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = row_override ? nullptr : llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = row_override ? nullptr : llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            // the scan is exact only while the reasoning budget passes logits through (forcing reads every id)
            const bool scan_ok = topk_scan_k > 0 && vocab_subset.empty()
                    && (rbudget == nullptr || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING);
            // THE LOGITS PREFILTER: the decode copied only each row's top-tk (id, logit) pairs, the
            // full row stays on the device. The same scan over those pairs keeps the same candidates in the same order
            // whenever its weakest kept logit is STRICTLY above the pairs' smallest one: every id left out has a logit
            // <= that smallest one, so none of them could have displaced a kept candidate. Otherwise (a tie at the
            // edge, skipped ids eating the margin) the full row is fetched and scanned as before.
            if (scan_ok && row_override == nullptr) {
                const int32_t * tk_ids  = nullptr;
                const float   * tk_vals = nullptr;
                const int32_t   tk      = llama_get_logits_topk_ith(ctx, idx, &tk_ids, &tk_vals);
                if (tk > topk_scan_k && topk_scan.run_mapped(tk_vals, tk, tk_ids, topk_scan_k, topk_skip)) {
                    float v_min = INFINITY;
                    for (int32_t t = 0; t < tk; ++t) {
                        v_min = std::min(v_min, tk_vals[t]);   // a NaN compares false and is passed over
                    }
                    if (topk_scan.top[topk_scan_k - 1].logit > v_min) {   // top[] is strongest first after run_mapped
                        cur.assign(topk_scan.top.begin(), topk_scan.top.end());
                        cur_p = { cur.data(), cur.size(), -1, true };
                        return;
                    }
                }
            }
            // a grammar's unconstrained first draw (topk_scan_k_gr): k+1 kept, used when all k+1 logits differ -- from the
            // decode's top-tk pairs when they settle it (the prefilter's own rule above), else from the full row
            const bool scan_gr = scan_first && topk_scan_k == 0 && topk_scan_k_gr > 0 && vocab_subset.empty()
                    && (rbudget == nullptr || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING);
            const auto use_gr = [&]() {
                for (int32_t i = 0; i < topk_scan_k_gr; ++i) {
                    if (topk_scan.top[i].logit == topk_scan.top[i + 1].logit) {
                        return false;
                    }
                }
                cur.assign(topk_scan.top.begin(), topk_scan.top.begin() + topk_scan_k_gr);
                cur_p = { cur.data(), cur.size(), -1, true };
                return true;
            };
            if (scan_gr && row_override == nullptr) {
                const int32_t * tk_ids  = nullptr;
                const float   * tk_vals = nullptr;
                const int32_t   tk      = llama_get_logits_topk_ith(ctx, idx, &tk_ids, &tk_vals);
                if (tk > topk_scan_k_gr + 1 && topk_scan.run_mapped(tk_vals, tk, tk_ids, topk_scan_k_gr + 1, topk_skip_gr)) {
                    float v_min = INFINITY;
                    for (int32_t t = 0; t < tk; ++t) {
                        v_min = std::min(v_min, tk_vals[t]);
                    }
                    if (topk_scan.top[topk_scan_k_gr].logit > v_min && use_gr()) {
                        return;
                    }
                }
            }
            const auto * logits = row_override ? row_override : llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            if (scan_gr && topk_scan.run(logits, n_vocab, topk_scan_k_gr + 1, topk_skip_gr) && use_gr()) {
                return;
            }
            if (scan_ok && topk_scan.run(logits, n_vocab, topk_scan_k, topk_skip)) {
                cur.assign(topk_scan.top.begin(), topk_scan.top.end());
                cur_p = { cur.data(), cur.size(), -1, true };
                return;
            }
            if (!vocab_subset.empty() && !vocab_subset_idx.empty()) {
                // COMPACT row: the draft head's own columns; same ids, same order, same values as the scattered row.
                // The chain reads only its top-k (topk_scan_k): scan the row once for it instead of building all
                // 32,768 candidates and partial-sorting them (65 us per draft step)
                if (topk_scan_k > 0
                        && (rbudget == nullptr || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING)
                        && topk_scan.run_mapped(logits, (int32_t) vocab_subset_col_ids.size(), vocab_subset_col_ids.data(),
                                                topk_scan_k, topk_skip)) {
                    cur.assign(topk_scan.top.begin(), topk_scan.top.end());
                    cur_p = { cur.data(), cur.size(), -1, true };
                    return;
                }
                cur.resize(vocab_subset.size());
                for (size_t i = 0; i < vocab_subset.size(); i++) {
                    cur[i] = llama_token_data{vocab_subset[i], logits[vocab_subset_idx[i]], 0.0f};
                }
            } else if (!vocab_subset.empty()) {
                cur.resize(vocab_subset.size());
                for (size_t i = 0; i < vocab_subset.size(); i++) {
                    const llama_token token_id = vocab_subset[i];
                    cur[i] = llama_token_data{token_id, logits[token_id], 0.0f};
                }
            } else {
                cur.resize(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
                }
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// The k for common_topk_scan, or 0: the chain must read nothing but its top-k before truncating to it -- every sampler
// ahead of TOP_K provably a no-op for these params (the early returns of their apply functions in
// src/llama-sampler.cpp), no grammar, no mirostat, logit biases only -inf (their ids are skipped by the scan).
static int32_t common_sampler_topk_scan_k(const common_params_sampling & params, const llama_sampler * grmr,
                                          const llama_vocab * vocab, std::vector<llama_token> & skip) {
    skip.clear();
    if (params.mirostat != 0 || grmr != nullptr
            || params.top_k <= 0 || params.top_k > 128) {
        return 0;
    }
    for (const auto & lb : params.logit_bias) {
        if (lb.bias != -INFINITY) {
            return 0;
        }
        skip.push_back(lb.token);
    }
    int32_t n_suppress = 0;
    const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    if (n_suppress > 0) {
        GGML_ASSERT(suppress != nullptr);
        skip.insert(skip.end(), suppress, suppress + n_suppress);
    }
    std::sort(skip.begin(), skip.end());
    skip.erase(std::unique(skip.begin(), skip.end()), skip.end());

    for (const auto & cnstr : params.samplers) {
        switch (cnstr) {
            case COMMON_SAMPLER_TYPE_TOP_K:
                return params.top_k;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (!(params.penalty_last_n == 0 ||
                      (params.penalty_repeat == 1.0f && params.penalty_freq == 0.0f && params.penalty_present == 0.0f))) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (!(params.dry_multiplier == 0.0f || params.dry_base < 1.0f || params.dry_penalty_last_n == 0)) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (params.top_n_sigma > 0.0f) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (params.typ_p < 1.0f) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (params.top_p < 1.0f) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (params.min_p > 0.0f) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (!(params.xtc_probability <= 0.0f || params.xtc_threshold > 0.5f)) {
                    return 0;
                }
                break;
            case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                break;   // not placed here: it replaces the final dist
            default:
                return 0;   // temperature, infill, ...: not provably order-neutral ahead of the top-k
        }
    }
    return 0;   // no top-k in the chain: every candidate matters
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };
    result->topk_scan_k = common_sampler_topk_scan_k(params, grmr, vocab, result->topk_skip);
    if (grmr != nullptr) {
        // the chain's own top-k rule without the grammar: its unconstrained first draws may scan
        result->topk_scan_k_gr = common_sampler_topk_scan_k(params, nullptr, vocab, result->topk_skip_gr);
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * res = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };
    res->vocab_subset = gsmpl->vocab_subset;
    res->vocab_subset_idx = gsmpl->vocab_subset_idx;
    res->vocab_subset_col_ids = gsmpl->vocab_subset_col_ids;
    res->topk_scan_k  = gsmpl->topk_scan_k;
    res->topk_skip    = gsmpl->topk_skip;
    res->topk_scan_k_gr = gsmpl->topk_scan_k_gr;
    res->topk_skip_gr   = gsmpl->topk_skip_gr;
    return res;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->vocab_subset = src->vocab_subset;
    dst->vocab_subset_idx = src->vocab_subset_idx;
    dst->vocab_subset_col_ids = src->vocab_subset_col_ids;
    dst->topk_scan_k  = src->topk_scan_k;
    dst->topk_skip    = src->topk_skip;
    dst->topk_scan_k_gr = src->topk_scan_k_gr;
    dst->topk_skip_gr   = src->topk_skip_gr;
    dst->t_total_us = src->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // the first draw is unconstrained unless the grammar filters first (topk_scan_k_gr)
    gsmpl->scan_first = !(grammar_first && grammar_should_apply(gsmpl));
    gsmpl->set_logits(ctx, idx);
    gsmpl->scan_first = false;

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // Verify position i predicts absolute position coupled_pos0 + i -- the same position the drafter armed when it
        // proposed draft[i]. If the two ever disagree the noise streams diverge and acceptance falls BELOW the uncoupled
        // baseline: a loud, measurable failure rather than a silent one.
        gsmpl->arm_coupled((int32_t) i);

        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        gsmpl->arm_coupled((int32_t) i);

        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    // the bonus token past the draft is a normal draw again
    gsmpl->disarm_coupled();

    return result;
}

// A clone of the sampler's STATEFUL parts only (chain, reasoning budget, grammar, token history): the candidate buffers
// are scratch that every sample refills, and copying `cur` (up to one entry per vocabulary token) per round would cost
// more than the whole verify decision.
static common_sampler * common_sampler_clone_state(common_sampler * gsmpl) {
    auto * res = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };
    res->vocab_subset         = gsmpl->vocab_subset;
    res->vocab_subset_idx     = gsmpl->vocab_subset_idx;
    res->vocab_subset_col_ids = gsmpl->vocab_subset_col_ids;
    res->topk_scan_k          = gsmpl->topk_scan_k;
    res->topk_skip            = gsmpl->topk_skip;
    res->topk_scan_k_gr       = gsmpl->topk_scan_k_gr;
    res->topk_skip_gr         = gsmpl->topk_skip_gr;
    return res;
}

// Block verification (Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv 2403.10444, Alg. 2):
// lossless -- every emitted token is an exact sample of the target -- and it keeps the LONGEST draft prefix whose block
// weight clears its uniform instead of stopping at the first rejection:
//   w_0 = 1,  w_{j+1} = min(w_j * p_j(d_j) / q_j(d_j), 1),
//   h_j = S_j / (S_j + 1 - w_j),  S_j = sum_x max(w_j * p_j(x) - q_j(x), 0)   (j < G),   h_G = w_G,
//   tau = max{ j : eta_j <= h_j },  then the bonus from p_G (tau == G) or one draw from max(w_tau p_tau - q_tau, 0).
// Measured +2.28% tokens per round over token-wise rejection (2.823 -> 2.887 on 46,662 held-out anchors).
// The target's distributions along the WHOLE draft are needed before tau is known, so they come from a state clone that
// accepts the draft tokens as it goes (reasoning budget, grammar, history advance exactly as they would); the real
// sampler then accepts only the tokens this round emits.
std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & dists,
        std::mt19937 & rng, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(dists.size() == draft.size() && "one proposal distribution per draft token");

    // the randomness of an accepted token came from the drafter's draw: the target's chain must not add a keyed draw
    gsmpl->disarm_coupled();
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const size_t G = draft.size();

    // 1. the target's distribution at every verify position, conditioned on the draft prefix
    std::vector<std::vector<llama_token_data>> P(G + 1);
    {
        common_sampler * cl = common_sampler_clone_state(gsmpl);
        for (size_t j = 0; j <= G; ++j) {
            common_sampler_sample(cl, ctx, idxs[j], grammar_first);
            P[j].assign(cl->cur_p.data, cl->cur_p.data + cl->cur_p.size);
            if (j < G) {
                common_sampler_accept(cl, draft[j], true);
            }
        }
        common_sampler_free(cl);
    }
    const auto prob = [](const std::vector<llama_token_data> & v, llama_token t) -> double {
        for (const auto & e : v) {
            if (e.id == t) {
                return e.p;
            }
        }
        return 0.0;
    };

    // 2. block weights and stop weights
    std::vector<double> w(G + 1, 1.0);
    for (size_t j = 0; j < G; ++j) {
        const double q = prob(dists[j], draft[j]);
        w[j + 1] = q > 0.0 ? std::min(w[j] * prob(P[j], draft[j]) / q, 1.0) : 0.0;   // a drawn token has q > 0
    }
    std::vector<double> h(G + 1, 1.0);
    for (size_t j = 0; j < G; ++j) {
        double S = 0.0;
        for (const auto & e : P[j]) {
            const double r = w[j] * (double) e.p - prob(dists[j], e.id);
            if (r > 0.0) {
                S += r;
            }
        }
        const double den = S + 1.0 - w[j];
        h[j] = den > 1e-15 ? S / den : 1.0;
    }
    h[G] = w[G];

    // 3. the longest prefix whose stop weight clears its own uniform (h_0 = 1: the empty prefix always qualifies)
    size_t tau = 0;
    for (size_t j = 1; j <= G; ++j) {
        if (uni(rng) <= h[j]) {
            tau = j;
        }
    }

    // 4. the final token: the bonus from p_G, or one draw from the block residual at tau (p_tau itself if it is empty)
    llama_token last = -1;
    {
        std::vector<std::pair<llama_token, double>> res;
        double z = 0.0;
        for (const auto & e : P[tau]) {
            const double r = tau == G ? (double) e.p : w[tau] * (double) e.p - prob(dists[tau], e.id);
            if (r > 0.0) {
                res.emplace_back(e.id, r);
                z += r;
            }
        }
        if (z <= 0.0) {
            res.clear();
            for (const auto & e : P[tau]) {
                res.emplace_back(e.id, (double) e.p);
                z += e.p;
            }
        }
        double u = uni(rng) * z;
        last = res.empty() ? draft[0] : res.back().first;
        for (const auto & kv : res) {
            if (u < kv.second) {
                last = kv.first;
                break;
            }
            u -= kv.second;
        }
    }

    // 5. the real sampler accepts exactly what this round emits
    std::vector<llama_token> result;
    result.reserve(tau + 1);
    for (size_t j = 0; j < tau; ++j) {
        common_sampler_accept(gsmpl, draft[j], true);
        result.push_back(draft[j]);
    }
    common_sampler_accept(gsmpl, last, true);
    result.push_back(last);
    return result;
}
void common_sampler_set_coupled(struct common_sampler * gsmpl, bool enabled, uint32_t seed, int32_t seq_id, int32_t pos0) {
    if (gsmpl == nullptr) {
        return;
    }

    gsmpl->coupled_enabled = enabled;
    gsmpl->coupled_seed    = seed;
    gsmpl->coupled_seq_id  = seq_id;
    gsmpl->coupled_pos0    = pos0;

    if (!enabled) {
        gsmpl->disarm_coupled();
    }
}

void common_sampler_arm_coupled(struct common_sampler * gsmpl, int32_t i) {
    if (gsmpl == nullptr) {
        return;
    }

    gsmpl->arm_coupled(i);
}

uint64_t common_sampler_coupled_key(const struct common_sampler * gsmpl, int32_t i) {
    if (gsmpl == nullptr || !gsmpl->coupled_enabled) {
        return 0;
    }
    return llama_sampler_coupled_key(gsmpl->coupled_seed, gsmpl->coupled_seq_id, gsmpl->coupled_pos0 + i);
}

bool common_sampler_chain_only(const struct common_sampler * gsmpl) {
    return gsmpl != nullptr && gsmpl->grmr == nullptr && gsmpl->rbudget == nullptr;
}

// THE DEVICE DRAFT CHAIN's host re-check: the chain on candidates that ARE the row's exact top-k,
// strongest first -- what set_logits' scan would have left in cur -- so the draw the GPU made is reproduced by the
// very samplers the host path runs. Only for a sampler with no grammar and no reasoning budget (chain_only).
llama_token common_sampler_sample_topk(struct common_sampler * gsmpl, const llama_token_data * cands, size_t n) {
    GGML_ASSERT(common_sampler_chain_only(gsmpl) && n > 0);
    gsmpl->cur.assign(cands, cands + n);
    gsmpl->cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), -1, true };
    llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);
    GGML_ASSERT(gsmpl->cur_p.selected >= 0 && "no selected token during sampling - check your sampling configuration");
    return gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
}

void common_sampler_set_vocab_subset(struct common_sampler * gsmpl, std::vector<llama_token> ids, std::vector<int32_t> idx) {
    if (gsmpl == nullptr) {
        return;
    }
    GGML_ASSERT(idx.empty() || idx.size() == ids.size());

    gsmpl->vocab_subset_col_ids.assign(idx.size(), LLAMA_TOKEN_NULL);
    for (size_t i = 0; i < idx.size(); ++i) {
        GGML_ASSERT(idx[i] >= 0 && (size_t) idx[i] < idx.size());
        gsmpl->vocab_subset_col_ids[(size_t) idx[i]] = ids[i];
    }
    gsmpl->vocab_subset     = std::move(ids);
    gsmpl->vocab_subset_idx = std::move(idx);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

// ---- xyz-engine ----------------------------------------------------------------------------------------
// The server hands a response's speculative rounds to an external engine (xyz_engine.dll) and walks every round's tokens
// through this sampler afterwards; see sampling.h.

bool common_sampler_engine_ok(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, std::vector<llama_token> & skip,
                              float & min_p, std::string & why) {
    const auto & p = gsmpl->params;
    const auto no = [&](const char * r) {
        why = r;
        return false;
    };
    min_p = 0.0f;
    if (p.mirostat != 0) return no("mirostat");
    if (p.top_k != 20) return no("top_k != 20");
    if (p.top_p != 0.95f) return no("top_p != 0.95");
    if (p.min_keep != 0) return no("min_keep");
    if (p.temp != 1.0f) return no("temperature != 1");
    if (p.dynatemp_range > 0.0f) return no("dynatemp");
    if (p.typ_p < 1.0f) return no("typical_p");
    if (p.top_n_sigma > 0.0f) return no("top_n_sigma");
    if (!(p.xtc_probability <= 0.0f || p.xtc_threshold > 0.5f)) return no("xtc");
    if (!(p.penalty_last_n == 0 || (p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f))) {
        return no("penalties");
    }
    if (!(p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n == 0)) return no("dry");
    // the engine keeps the top 20, then the nucleus of those, then (min_p > 0) the min-p prefix of those: top_k ahead of
    // top_p ahead of min_p, and the chain ends in dist
    bool seen_k = false, seen_p = false, seen_m = false;
    for (const auto & s : p.samplers) {
        if (s == COMMON_SAMPLER_TYPE_TOP_K) seen_k = true;
        if (s == COMMON_SAMPLER_TYPE_TOP_P) {
            if (!seen_k) return no("top_p ahead of top_k");
            if (seen_m && p.min_p > 0.0f) return no("min_p ahead of top_p");
            seen_p = true;
        }
        if (s == COMMON_SAMPLER_TYPE_MIN_P && p.min_p > 0.0f) {
            if (seen_m) return no("min_p twice in the chain");
            if (!seen_k) return no("min_p ahead of top_k");
            seen_m = true;
        }
        if (s == COMMON_SAMPLER_TYPE_ADAPTIVE_P) return no("adaptive_p");
        if (s == COMMON_SAMPLER_TYPE_INFILL) return no("infill");
    }
    if (!seen_k || !seen_p) return no("no top_k / top_p in the chain");
    if (seen_m) {
        min_p = p.min_p;
    }
    // logit biases: only -inf (the top-k never admits those ids), plus the vocabulary's suppress tokens
    skip.clear();
    for (const auto & lb : p.logit_bias) {
        if (lb.bias != -INFINITY) return no("a finite logit bias");
        skip.push_back(lb.token);
    }

    int32_t n_suppress = 0;
    const llama_token * suppress = vocab != nullptr ? llama_vocab_get_suppress_tokens(vocab, &n_suppress) : nullptr;
    for (int32_t i = 0; i < n_suppress; ++i) skip.push_back(suppress[i]);
    std::sort(skip.begin(), skip.end());
    skip.erase(std::unique(skip.begin(), skip.end()), skip.end());
    return true;
}

int common_sampler_engine_budget(const struct common_sampler * gsmpl, int32_t * remaining, int32_t * budget) {
    if (gsmpl->rbudget == nullptr) {
        *remaining = 0;
        *budget    = 0;
        return -1;
    }
    *remaining = common_reasoning_budget_get_remaining(gsmpl->rbudget, budget);
    return (int) common_reasoning_budget_get_state(gsmpl->rbudget);
}

bool common_sampler_engine_grammar_active(const struct common_sampler * gsmpl) {
    if (gsmpl == nullptr || gsmpl->grmr == nullptr) {
        return false;
    }
    return !gsmpl->params.grammar_lazy || !llama_sampler_grammar_awaiting(gsmpl->grmr);
}

// While a lazy grammar waits, llama_grammar_apply_impl returns at once and llama_grammar_accept_impl only looks for a
// trigger (src/llama-grammar.cpp): with token triggers alone it switches on exactly when the sampler accepts one.
bool common_sampler_lazy_idle(const struct common_sampler * gsmpl, std::vector<llama_token> & trig) {
    trig.clear();
    if (gsmpl == nullptr || gsmpl->grmr == nullptr || !gsmpl->params.grammar_lazy ||
            !llama_sampler_grammar_awaiting(gsmpl->grmr)) {
        return false;
    }
    for (const auto & t : gsmpl->params.grammar_triggers) {
        if (t.type != COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            trig.clear();
            return false;
        }
        trig.push_back(t.token);
    }
    const std::vector<llama_token> base = trig;
    const auto has_trig = [&](const llama_tokens & seq) {
        for (const llama_token id : seq) {
            if (std::find(base.begin(), base.end(), id) != base.end()) {
                return true;
            }
        }
        return false;
    };
    // the forced sequence runs only when a budget expires: decline if it could trigger
    if (trig.empty() || has_trig(gsmpl->params.reasoning_budget_forced)) {
        trig.clear();
        return false;
    }
    // a reasoning end sequence is replayed into the grammar once it completes (common_sampler_accept): one that holds
    // a trigger id switches the grammar on when its LAST token is accepted, so that token cuts too (a thinking chat
    // template lists "<tool_call>" itself as an end tag: the same id, nothing added)
    for (const auto & seq : gsmpl->params.reasoning_budget_end) {
        if (!seq.empty() && has_trig(seq) && std::find(trig.begin(), trig.end(), seq.back()) == trig.end()) {
            trig.push_back(seq.back());
        }
    }
    return true;
}

size_t common_draft_cut_at_trigger(const llama_tokens & draft, const std::vector<llama_token> & trig) {
    for (size_t j = 0; j < draft.size(); ++j) {
        if (std::find(trig.begin(), trig.end(), draft[j]) != trig.end()) {
            return j;
        }
    }
    return draft.size();
}

bool common_sampler_engine_grammar_ok(struct common_sampler * gsmpl, llama_token id) {
    if (!grammar_should_apply(gsmpl)) {
        return true;
    }
    // common_sampler_sample's check of an unconstrained draw
    llama_token_data       d = { id, 1.0f, 0.0f };
    llama_token_data_array a = { &d, 1, -1, false };
    llama_sampler_apply(gsmpl->grmr, &a);
    return a.data[0].logit != -INFINITY;
}

void common_sampler_set_row_override(struct common_sampler * gsmpl, const float * row) {
    if (gsmpl != nullptr) {
        gsmpl->row_override = row;
    }
}

void common_sampler_engine_dist_skip(struct common_sampler * gsmpl, int32_t n) {
    const int m = llama_sampler_chain_n(gsmpl->chain);
    for (int j = 0; j < m; ++j) {
        llama_sampler_dist_discard(llama_sampler_chain_get(gsmpl->chain, j), n);
    }
}

void common_sampler_engine_accept_from(struct common_sampler * gsmpl, struct llama_context * ctx, const float * const * rows,
                                       int32_t k, const llama_tokens & draft, llama_tokens & out) {
    // common_sampler_sample_and_accept_n from draft position k, on the engine's verify rows
    const auto draw = [&](size_t i) {
        gsmpl->arm_coupled((int32_t) i);
        gsmpl->row_override = rows[i];
        const llama_token id = common_sampler_sample(gsmpl, ctx, 0, false);
        gsmpl->row_override = nullptr;
        common_sampler_accept(gsmpl, id, true);
        out.push_back(id);
        return id;
    };
    size_t i = (size_t) k;
    for (; i < draft.size(); i++) {
        if (draw(i) != draft[i]) {
            break;
        }
    }
    if (i == draft.size()) {
        draw(i);
    }
    gsmpl->disarm_coupled();
}
