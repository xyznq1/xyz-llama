#pragma once
// xyz-engine accept step on the device: the server's host path -- the target's top-k scan, its sampler chain
// (top_k 20 -> top_p -> dist, llama-sampler.cpp) and block verification of a draft
// (common_sampler_sample_and_accept_n_block, common/sampling.cpp) with the slot's std::mt19937 -- bit for bit: MSVC's
// expf via an exception table (tools/expf_table.cu), float/double IEEE arithmetic in the host's order, no FMA, MSVC
// 14.44's generate_canonical. Compiled without fast math (CMake target xe_exact).
#include <cuda_runtime.h>

#include <cstdint>

namespace xe {

constexpr int ACC_MAX_G  = 8;    // drafts per round
constexpr int ACC_MAX_TK = 32;   // candidates per row

// std::mt19937 (the standard algorithm; MSVC's engine yields the standard sequence)
struct Mt19937 {
    uint32_t mt[624];
    int32_t  idx;
};

struct AcceptParams {
    int   G;            // drafts
    int   tk;           // candidates per row (the chains' top_k, 20)
    float top_p_t;      // the target's top_p (0.95)
    float top_p_d;      // the drafter's (XYZ2_DRAFT_TOP_P, 0.95)
    int   min_keep_t;   // their min_keep (0)
    int   min_keep_d;
    // 0: block verification for a request without grammar; 1: plain coupled verification
    // (common_sampler_sample_and_accept_n: row j's token is the coupled draw keyed coupled_key(cseed, 0, p0 + 1 + j),
    // accepted while it equals draft j) -- what a request with a (tool-call) grammar runs
    int   mode;
    // A waiting lazy grammar (xe_start.cut), device memory: [0] = n (<= 16), [1..n] = its trigger ids. Block verification
    // reads only the draft before the first of them; nullptr or n 0: the whole draft.
    const int32_t * cut;
};

constexpr int ACC_BLOCK = 0;
constexpr int ACC_PLAIN = 1;

struct AcceptOut {
    int32_t n;                          // accepted tokens incl. the last (tau + 1)
    int32_t tokens[ACC_MAX_G + 1];
    // Target and drafter distributions after top_p, retained in host order for acceptance.
    int32_t p_n[ACC_MAX_G + 1];
    int32_t p_id[ACC_MAX_G + 1][ACC_MAX_TK];
    float   p_p[ACC_MAX_G + 1][ACC_MAX_TK];
    int32_t q_n[ACC_MAX_G];
    int32_t q_id[ACC_MAX_G][ACC_MAX_TK];
    float   q_p[ACC_MAX_G][ACC_MAX_TK];
    // plain mode: bit j set when row j's candidates after top_p were ONE (the host's dist then draws once from its
    // own std::mt19937 anyway -- llama_sampler_dist_apply's size-1 branch; the host discards as many)
    int32_t one_mask;
    // the first draw the server's host re-check would NOT reproduce (its own top_p cut over the draw's record, the same
    // coupled selection), -1: none. The server cuts its draft there (a shorter verify); the engine cannot, so the round
    // may differ from the server's path -- reported, never silent
    int32_t cut;
};

// exact top-tk per row of [n_vocab, rows] logits in the host scan's order (logit desc, then id asc; NaN, -inf and the
// skip ids never enter): ids/logits [rows][tk] strongest first; n_ok[r] = candidates found (< tk: the host falls back),
// | TK_TIE when two kept candidates share a logit or the last kept ties the first left out
constexpr int32_t TK_TIE = 0x40000000;
// The threshold bin is selected by a 1024-thread suffix sum.
void topk_rows(cudaStream_t st, const float * logits, int n_vocab, int rows, int tk, const int32_t * skip, int n_skip,
               int32_t * ids, float * vals, int32_t * n_ok, uint64_t * scratch);
size_t topk_rows_scratch(int n_vocab, int rows);

// the accept: t_* the target's candidates [G+1][tk] (topk_rows), d_* each draft's candidates [G][tk] in its record's
// order, draft[G] the drawn tokens; rng advances exactly as the host's; out on the device. keys2: the plain mode's
// coupled keys of rows 0..G (low half first) -- the draft chain's keys plus the bonus row's
void accept(cudaStream_t st, const AcceptParams & prm, const int32_t * t_ids, const float * t_logit, const int32_t * d_ids,
            const float * d_logit, const int32_t * draft, Mt19937 * rng, const uint32_t * exc, int n_exc, AcceptOut * out,
            const uint32_t * keys2 = nullptr);

} // namespace xe
