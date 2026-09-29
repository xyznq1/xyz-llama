#pragma once
// xyz-engine session: one response's seed decode, device draft chain, verify pass with FC fold, exact top-20, and accept.
#include <cstdint>
#include <functional>
#include <vector>

#include "accept.h"
#include "draft.h"
#include "engine.h"

struct SessionStart {
    int32_t  p0      = 0;       // the first verify's position: id_last sits there
    int32_t  id_last = 0;
    int32_t  pending = 0;       // the rs prefix the first verify replays (the server's spec_rs_prev)
    int      m       = 1;       // the first seed decode: catch-up rows + the seed row (<= Drafter::MAXR)
    int32_t  seed_tok[Drafter::MAXR] = {};
    int32_t  seed_pos0 = 0;
    const float * seed_g = nullptr;   // [5120, m] host rows
    uint32_t cseed = 0;         // coupled_seed ^ nonce: draw s of a round is keyed coupled_key(cseed, 0, p0 + 1 + s)
    xe::Mt19937 rng = {};       // the slot's spec_rej_rng
    int      n_draws = 4;       // the rounds' draft count (the request's n_max, --spec-draft-n-max by default)
    // the device draft chain's length as llama_draft_chain_init set it (the server's --spec-draft-n-max; 0 = n_draws): the
    // pruned last-step FFN runs at chain step chain_steps - 1 ONLY -- a round drafting fewer (the budget end) never
    // reaches it, whatever its G (llama_context: last_draw = dchain.step == dchain.n_steps - 1, n_steps fixed at init)
    int      chain_steps = 0;
    int      accept_mode = xe::ACC_BLOCK;   // ACC_PLAIN: the request carries a grammar (the server's plain coupled verify)
    // At the budget end, each round drafts min(n_draws, remaining - 1); with one token left, control returns to the server.
    bool     budget_tail = false;
    // A waiting lazy grammar (xe_start.cut): block verification reads each draft only up to its first id here (<= 16).
    const int32_t * cut = nullptr;
    int      n_cut = 0;
    // the drafter's cell table is the Drafter's ring (Drafter::ring_load: the server's table at the takeover)
};

struct SessionEnd {
    int      rounds = 0;
    int      n_tokens = 0;      // accepted tokens handed to the callback
    int32_t  p0 = 0, id_last = 0, pending = 0;   // what the next round would start from
    int      m = 0;
    int32_t  seed_tok[8] = {};
    int32_t  seed_pos0 = 0;
    uint64_t rng_draws = 0;     // std::mt19937 outputs the accept consumed (the host's engine discards as many)
    int      short_rows = 0;    // verify rows with fewer than K finite candidates (the host would have scanned the row)
    int      tie_rows = 0;      // verify rows whose top-k held tied logits
    int      cut_rounds = 0;    // rounds the server's host re-check would have cut (RF_CUT)
};

struct Session {
    Engine  & e;
    Drafter & dr;
    int K = 20;
    uint32_t * d_exc = nullptr;
    int n_exc = 0;
    xe::Mt19937 * d_rng = nullptr;
    // [0]: the waiting grammar's trigger count (0: no cut), [1..16]: its ids; written by run(), read by the accept kernel
    int32_t * d_cut = nullptr;
    int32_t * d_tid = nullptr, * d_nok = nullptr, * d_did = nullptr, * d_draft = nullptr;
    float * d_tval = nullptr, * d_dval = nullptr, * d_seed_g = nullptr;
    uint64_t * d_tks = nullptr;
    xe::AcceptOut * d_out = nullptr, * h_out = nullptr;
    int32_t * h_nok = nullptr, * h_draft = nullptr;
    // the ids the target's chain biases to -inf (its logit-bias sampler: the request's biases, the vocabulary's suppress
    // tokens): the top-k never admits them (common_sampler topk_skip), sorted
    int32_t * topk_skip = nullptr;
    int       n_topk_skip = 0;
    void set_topk_skip(const int32_t * ids, int n);

    Session(Engine & engine, Drafter & drafter) : e(engine), dr(drafter) {}
    bool init(const char * expf_path);
    // one round's outcome for the host: toks[0..n) in order (the accepted drafts, then the new token); in the plain mode
    // one_mask (bit j: row j had one candidate); the round's drafts[0..n_draws); flags: RF_TIE (a verify row's top-k held
    // tied logits -- the host's full-candidate path may order them differently), RF_SHORT (a row had fewer than K finite
    // candidates). The host may REWRITE toks and n (<= 5) when its own sampler decides the round differently (a grammar
    // that rejects a token, a tie: it re-accepts from the verify's logits, logits_row); the next round starts from what
    // it leaves. false: stop after this round.
    // RF_CUT: the server's host re-check would have cut this round's draft short (a shorter verify the engine cannot run:
    // the round may differ from the server's path)
    static constexpr uint32_t RF_TIE = 1, RF_SHORT = 2, RF_CUT = 4;
    using RoundFn = std::function<bool(int32_t * toks, int & n, uint32_t one_mask, const int32_t * drafts, uint32_t flags)>;
    int round_g = 0;   // the round's draft count (drafts[0..round_g)), valid inside on_round
    // One round's accept launches: top-k, draft records, acceptance, and the next seed's fold rows.
    void issue_accept(int G, int mode);
    // rounds until max_tokens are handed over or on_round returns false; the stream is e.st
    bool run(const SessionStart & s, int max_tokens, const RoundFn & on_round, SessionEnd & out);
    // row r of the last verify's logits (n_vocab floats) to the host -- valid until the next round
    void logits_row(int r, float * dst);
};
