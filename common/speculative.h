#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

// xyz-engine: an xyz drafter's next seed decode for a sequence -- the catch-up rows draft() emits ahead
// of the seed (ids[k] at pos[k] with g row k) and the deferred boundary row (pos_last, g_last: the seed row, whose token
// is the server's id_last) -- plus the last verify's g rows (process()'s verify_g: first position, row count)
struct common_speculative_engine_rows {
    std::vector<llama_token> ids;
    std::vector<llama_pos>   pos;
    std::vector<float>       g;          // [ids.size() * n_embd]
    llama_pos                pos_last = -1;
    std::vector<float>       g_last;     // [n_embd]
    llama_pos                verify_pos_first = -1;
    int32_t                  verify_rows = 0;
    std::vector<float>       verify_g;   // [verify_rows * n_embd]
    int32_t                  n_embd = 0;
    int32_t                  merge_max = 0;   // engine_state: the catch-up merge's stash capacity (0: merging off)
};
// the state a PROMPT batch starts from (the stash as process() trims it, the deferred boundary or pos_last -1, the merge
// capacity); false: not xyz / block drafting
bool common_speculative_engine_state(common_speculative * spec, llama_seq_id seq_id, common_speculative_engine_rows & out);
// false: no implementation has one (not xyz, block drafting, no deferred boundary)
bool common_speculative_engine_seed(common_speculative * spec, llama_seq_id seq_id, common_speculative_engine_rows & out);
// the state an external engine's rounds leave: what the next draft() starts from, as if the server had run them
bool common_speculative_engine_set (common_speculative * spec, llama_seq_id seq_id, const common_speculative_engine_rows & in);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // Absolute position that result[0] predicts, written by the drafter: the verify side arms coupled selection on the
    // same positions (-1 = not set). Kept LAST, with default initialisers, so the aggregate construction of this struct
    // in tools/server/server-context.cpp keeps binding its fields positionally.
    llama_pos pos0 = -1;

    // Per-REQUEST part of the coupled-selection key, set by the server for each task and mixed into the server-wide
    // coupled_seed on both sides (drafter and verifier): without it identical prompts would draw identical samples for
    // the whole server lifetime -- every draw still an exact sample, but no longer independent across requests.
    uint32_t coupled_nonce = 0;

    // The early verify. early_rs_prev: set by the server when the drafter may issue this round's target verify itself,
    // right behind the device draft chain (the draft rows read on the device from the chain's records,
    // llama_verify_chain_arm); the value is the recurrent prefix the server would arm for that decode
    // (llama_rs_set_prefix), -2 = not allowed. verify_preissued: set by the drafter -- the verify of [id_last, result...]
    // is already issued on ctx_tgt, so the server skips its own llama_rs_set_prefix and llama_decode and only synchronizes.
    int32_t early_rs_prev    = -2;
    bool    verify_preissued = false;

    // dist[i]: the distribution result[i] was DRAWN from (candidate ids with probabilities summing to 1), aligned with
    // result, or empty when any token was not a plain draw from its recorded distribution. The rejection verify
    // (--spec-rejection) accepts result[i] with probability min(1, p/q), else draws from the residual.
    std::vector<std::vector<llama_token_data>> dist;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// notify implementations after replacing token history
void common_speculative_history_replaced(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
