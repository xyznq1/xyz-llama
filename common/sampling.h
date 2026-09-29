#pragma once

#include "llama.h"

#include "common.h"

#include <random>
#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
// Arm coupled (shared-noise) selection for speculative decoding. `pos0` is the absolute position
// that draft[0] predicts; position i of the draft uses key(seed, seq_id, pos0 + i). The drafter must
// arm the identical (seed, seq_id, pos0) or the two noise streams diverge.
void common_sampler_set_coupled(struct common_sampler * gsmpl, bool enabled, uint32_t seed, int32_t seq_id, int32_t pos0);

// Arm the chain for draft position `i` (relative to pos0). Callers that drive their own draft loop
// (the drafters in common/speculative.cpp) use this; common_sampler_sample_and_accept_n does it
// internally for the verify side.
void common_sampler_arm_coupled(struct common_sampler * gsmpl, int32_t i);

// the device draft chain: the coupled key arm_coupled(i) would use (0 when not coupled); whether
// the sampler is the chain alone (no grammar, no reasoning budget); the chain run on the row's exact top-k candidates
uint64_t    common_sampler_coupled_key(const struct common_sampler * gsmpl, int32_t i);
bool        common_sampler_chain_only (const struct common_sampler * gsmpl);
llama_token common_sampler_sample_topk(struct common_sampler * gsmpl, const llama_token_data * cands, size_t n);

// Candidates come only from these token ids (ascending) instead of the whole vocabulary: for a reduced-vocabulary
// draft head, whose graph writes -inf to every other logit, so the chain sees the same finite candidates without
// building and sorting ~215k dead entries per call. Empty = the whole vocabulary. Backend-sampled rows are unaffected.
// idx non-empty: the logits row is COMPACT (the draft head's own columns, XYZ2_COMPACT_LOGITS); ids[i] reads column idx[i].
void common_sampler_set_vocab_subset(struct common_sampler * gsmpl, std::vector<llama_token> ids, std::vector<int32_t> idx = {});

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);


// Speculative verify by block verification (Sun et al. 2403.10444, Alg. 2): draft[i] was DRAWN from dists[i] (ids
// with probabilities). Lossless -- every returned token is an exact sample of the target's distribution after its chain
// at idxs[i] -- and the kept prefix is the LONGEST whose block weight clears its uniform, not the first rejection's
// (token-wise rejection reaches 1 - TV(p, q) per position; this keeps more). Returns a draft prefix plus one token.
// Needs a chain that ends in a dist draw (temperature > 0, no mirostat).
std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & dists,
        std::mt19937 & rng, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;

// ---- xyz-engine: the server hands a response's speculative rounds to an external engine
// (xyz_engine.dll) and walks each round's tokens through the slot's sampler afterwards
// whether the target's chain is the one the engine reproduces: top_k 20 ahead of top_p 0.95, then dist; every other
// sampler neutral (temperature 1, min_p 0, no penalties / dry / xtc / typical / top-n-sigma / mirostat / adaptive-p),
// logit biases only -inf -- their ids (and the vocabulary's suppress tokens) in skip; why: the first reason it is not
bool common_sampler_engine_ok(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, std::vector<llama_token> & skip,
                              std::string & why);
// the reasoning budget: its state (common_reasoning_budget_state, -1 without one), the tokens left and the block's budget
int  common_sampler_engine_budget(const struct common_sampler * gsmpl, int32_t * remaining, int32_t * budget);
// whether common_sampler_sample keeps `id` drawn unconstrained: false only when the grammar applies and rejects it
bool common_sampler_engine_grammar_ok(struct common_sampler * gsmpl, llama_token id);
// a grammar constrains the next token now (a non-lazy grammar, or a lazy one that has triggered)
bool common_sampler_engine_grammar_active(const struct common_sampler * gsmpl);
// A lazy grammar still waiting for its trigger, whose triggers are all single tokens (a tool-call grammar on
// "<tool_call>"). Until the sampler accepts one of those ids the grammar filters nothing, so rejection and block
// verification are exact on a draft cut before its first cut id. The cut ids go to trig: the triggers, plus the last
// token of any reasoning end sequence holding one. False when there is no grammar, it is not lazy, it has triggered, a
// trigger is a word or a pattern, or the reasoning budget's forced sequence holds a trigger id.
bool common_sampler_lazy_idle(const struct common_sampler * gsmpl, std::vector<llama_token> & trig);
// the draft length verification may use: the index of the first trigger id in draft, or its size
size_t common_draft_cut_at_trigger(const llama_tokens & draft, const std::vector<llama_token> & trig);
// n draws of the chain's dist rng (verify rows the engine decided with one candidate)
void common_sampler_engine_dist_skip(struct common_sampler * gsmpl, int32_t n);
// the rest of a plain coupled verify from draft position k (common_sampler_sample_and_accept_n from i = k) on the
// engine's verify rows (rows[i]: row i, n_vocab floats; set_coupled first); the tokens it keeps are appended to out
// the next draws read this logits row (n_vocab floats) instead of the context's outputs -- the prompt's output row the
// engine computed (xe_prefill); nullptr clears it
void common_sampler_set_row_override(struct common_sampler * gsmpl, const float * row);
void common_sampler_engine_accept_from(struct common_sampler * gsmpl, struct llama_context * ctx, const float * const * rows,
                                       int32_t k, const llama_tokens & draft, llama_tokens & out);
