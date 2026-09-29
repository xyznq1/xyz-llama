#pragma once
// xyz-engine C API: Qwen3.8-27B PTQ1_0 and the xyz2 drafter bound to the host server's weights and memory. Prompt batches
// and speculative rounds hand state back exactly as the server's own path leaves it. The server retains prompt caching,
// protocol handling, and bookkeeping.
#include <stdint.h>

#ifdef XE_BUILD
#define XE_API __declspec(dllexport)
#else
#define XE_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xe_ctx xe_ctx;

// The server's device pointer for a GGUF tensor of model 0 (target) or 1 (drafter), NULL if it lives on the host or is absent;
// model 2: the HOST pointer of a target tensor the server keeps on the host (the input embedding, read in place), else NULL.
typedef const void * (*xe_tensor_fn)(void * user, int model, const char * name);

struct xe_bind {
    const char * target_gguf;
    const char * drafter_gguf;
    const char * expf_table;          // ref/expf_exc.bin
    xe_tensor_fn tensor;
    void *       user;
    // The target's memory, per model layer (NULL where a layer has none); recurrent pointers are sequence 0's row.
    int32_t kv_size;
    void *  k[128];
    void *  v[128];
    float * conv[128];
    float * ssm[128];
    float * pk[128];
    float * px[128];
    // The drafter's SWA cache: cells, K/V (q4_0), and window.
    int32_t d_kv_size;
    void *  dk;
    void *  dv;
    int32_t d_n_swa;
};

// Loads metadata and binds the server's memory; returns NULL on failure after writing the reason to stderr.
XE_API xe_ctx * xe_create(const struct xe_bind * b);
XE_API void     xe_free(xe_ctx * c);
// Rebind recurrent rows before takeover when a checkpoint restore moved them between requests.
XE_API void     xe_rebind_rs(xe_ctx * c, float * const * conv, float * const * ssm, float * const * pk, float * const * px);

// The first seed decode carries at most the merged prompt stash (<= 64), boundary, and seed rows.
#define XE_SEED_MAX 72

struct xe_start {
    int32_t  p0;            // the first verify's position (id_last sits there)
    int32_t  id_last;
    int32_t  pending;       // the recurrent-state prefix the first verify replays
    int32_t  m;             // catch-up rows plus the seed row (<= XE_SEED_MAX)
    int32_t  seed_tok[XE_SEED_MAX];
    int32_t  seed_pos0;
    const float * seed_g;   // [5120, m] host rows
    uint32_t cseed;         // coupled_seed ^ nonce
    uint32_t rng[624];      // the slot's rejection RNG textual state (last 624 outputs, oldest first)
    // The drafter's cell table: positions of cells [0, d_n), followed by empty cells, and the cache head.
    const int32_t * d_pos;
    int32_t  d_n;
    int32_t  d_head;
    int32_t  n_draws;       // 4
    // 0: block verification without grammar; 1: plain coupled verification with grammar.
    int32_t  accept_mode;
    // Token ids the target sampler biases to -inf.
    const int32_t * skip;
    int32_t  n_skip;
    // 1: each round drafts min(n_draws, max_tokens - handed - 1); at G = 0 control returns to the server for its plain
    // one-token decode. 0: every round drafts n_draws.
    int32_t  budget_tail;
    // Drafter chain length; 0 means n_draws. The pruned final-step FFN runs only at the last chain step.
    int32_t  chain_steps;
    // A lazy grammar still waiting for a token trigger (common_sampler_lazy_idle): with accept_mode 0, block verification
    // reads each draft only up to its first id in cut[0, n_cut), as the server's verification does, so a trigger id
    // can only be a round's last token. n_cut 0: no cut.
    const int32_t * cut;
    int32_t  n_cut;
};

// One round: toks[0..*n) contains accepted drafts followed by the new token; one_mask bit j marks a plain-verification
// row that kept one candidate; drafts[0..n_drafts) contains the draft. A callback may rewrite toks and *n (<= 5) after
// sampling the verify logits through xe_logits_row. Return 0 to stop after the round.
#define XE_RF_TIE   1u
#define XE_RF_SHORT 2u
#define XE_RF_CUT   4u
typedef int (*xe_round_fn)(void * user, int32_t * toks, int32_t * n, uint32_t one_mask, const int32_t * drafts,
                           int32_t n_drafts, uint32_t flags);

struct xe_end {
    int32_t  rounds;
    int32_t  n_tokens;
    int32_t  p0, id_last, pending;    // the next round's start
    int32_t  m;                       // the next seed decode: catch-up rows plus seed
    int32_t  seed_tok[8];
    int32_t  seed_pos0;
    uint64_t rng_draws;               // outputs consumed by acceptance; discard this many from the server RNG
    int32_t  short_rows;              // verify rows with fewer than top_k finite candidates
    int32_t  tie_rows;                // verify rows whose top-k held tied logits
    int32_t  cut_rounds;              // rounds the server's host re-check would have cut short
    int32_t  captures;                // CUDA graphs captured during this call
};

// Runs until max_tokens are handed over or cb returns 0, then synchronizes the device. xe_seed_rows copies the final
// verify fold rows and xe_flush_pack restores the last GDN pack row to server memory when necessary.
XE_API int  xe_generate(xe_ctx * c, const struct xe_start * s, int32_t max_tokens, xe_round_fn cb, void * user,
                        struct xe_end * out);
XE_API int  xe_seed_rows(xe_ctx * c, float * dst, int32_t m);
// Copies row r of the last verify logits (n_vocab floats); valid inside the round callback.
XE_API void xe_logits_row(xe_ctx * c, int32_t r, float * dst);
XE_API void xe_flush_pack(xe_ctx * c);
// Returns the used drafter cache span after copying positions [0, n) and the head.
XE_API int  xe_ring(xe_ctx * c, int32_t * pos, int32_t n, int32_t * head);

// One prompt batch. The target processes n_ubatch chunks into the server's cells and recurrent row. The drafter consumes
// the target fold rows, a carried stash and boundary row, and the shifted batch tokens; a small merge may defer those rows.
struct xe_prefill_in {
    const int32_t * tokens;       // positions p0 .. p0 + n - 1 of sequence 0
    int32_t n, p0;
    int32_t n_ubatch;
    int32_t n_ubatch_dft;         // must equal n_ubatch
    int32_t want_logits;          // copy the final row's logits to out->logits
    const int32_t * d_pos;
    int32_t d_n, d_head;
    const int32_t * stash_ids;
    const int32_t * stash_pos;
    const float *   stash_g;      // [5120, n_stash]
    int32_t n_stash;
    int32_t pending_pos;          // deferred boundary position, or -1
    const float * pending_g;      // [5120]
    int32_t merge_max;            // stash capacity; 0 disables merging
    int32_t pending;              // packed recurrent rows replayed by the first ubatch
};
struct xe_prefill_out {
    float *   logits;             // [n_vocab] when want_logits
    float *   g_last;             // [5120], the new deferred boundary row
    float *   verify_g;           // [5120, n]
    int32_t * stash_ids;          // [merge_max]
    int32_t * stash_pos;
    float *   stash_g;            // [5120, merge_max]
    int32_t   n_stash;
    int32_t   decoded_rows;
};
// 0: done; > 0: unsupported batch with nothing written; < 0: failure after writing, leaving state untrustworthy.
XE_API int  xe_prefill(xe_ctx * c, const struct xe_prefill_in * in, struct xe_prefill_out * out);

#ifdef __cplusplus
}
#endif
