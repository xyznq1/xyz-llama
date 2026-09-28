#pragma once
// xyz-engine drafter: the xyz2 head (xyz arch in MTP-fusion mode: eh_proj([enorm(emb); hnorm(g)]) -> one gated-attention
// layer over a q4_0 sliding-window cache -> FFN -> output_norm -> the PTQ1 head's 32,768 compact columns) as the server runs
// one decode of it using the fork's kernel sequence.
// The device draft chain (the server's llama_draft_chain, src/models/xyz.cpp): each draw on the GPU (k_draft_sample_v3),
// the next step's token and g taken from it on the device, the last draw through the pruned FFN (the 3+1 head).
#include <cuda_fp16.h>

#include "kernels.h"
#include "model.h"
#include "ring.h"

#include <functional>
#include <unordered_map>
#include <vector>

#define GGML_DRAFT_SAMPLE_MAX_K 32
#define GGML_DRAFT_SAMPLE_OUT   (4 + 3*GGML_DRAFT_SAMPLE_MAX_K)   // ggml.h: one draw's record, int32

struct Drafter {
    static constexpr int MAXS = 8;   // draws per draft (the serve script: --spec-draft-n-max 4)

    const Model & tgt;     // the token table and its rotation come from the target
    const Model & d;
    int   kv_cells = 0;    // the SWA cache's cells (llama: PAD(min(n_ctx, n_swa + n_ubatch), 256))
    int   n_swa    = 4096;
    float eps      = 1e-6f;

    const float * enorm = nullptr, * hnorm = nullptr, * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr,
                * ffn_norm = nullptr, * out_norm = nullptr, * ffn_norm_l = nullptr;
    const void  * eh_proj = nullptr, * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr,
                * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr, * head = nullptr,
                * ffn_gate_l = nullptr, * ffn_up_l = nullptr, * ffn_down_l = nullptr;   // the pruned last-draw FFN
    int   t_q3k = 0;
    int   n_ff = 0, n_ff_l = 0;   // FFN widths (full, pruned; 0 = no pruned FFN)
    int   n_cols = 0;      // head rows (the draft vocabulary)
    const float * t_s5120 = nullptr, * d_s5120 = nullptr;

    // one token's buffers
    float * emb = nullptr, * emb_rot = nullptr, * emb_s = nullptr, * cat = nullptr, * fused = nullptr, * cur = nullptr;
    float * qfull = nullptr, * q = nullptr, * qrot = nullptr, * k = nullptr, * kn = nullptr, * krot = nullptr, * v = nullptr,
          * vrot = nullptr;
    float * fa = nullptr, * fa_rot = nullptr, * gate_c = nullptr, * gated = nullptr, * ffn_inp = nullptr, * ffn_n = nullptr,
          * glu = nullptr, * out = nullptr, * fb = nullptr, * h_rot = nullptr, * logits = nullptr;
    char  * q8 = nullptr, * q8h = nullptr;
    half  * mask = nullptr;
    void  * kc = nullptr, * vc = nullptr;
    struct In { int32_t tok; int32_t pos; int64_t cell; int32_t n_kv; int32_t pad; };
    In    * in_dev = nullptr, * in_host = nullptr;   // [MAXS]: one slot per draw of a draft (no reuse before a sync)

    // the seed decode of a round: m = 1 + catch-up rows (the tokens the target accepted, with their g rows) -- m >= 2 runs
    // the server's unfused multi-row sequence, with the last row drawn. The first seed after a prompt
    // also carries the merged stash (up to its 64 rows) and the deferred boundary row: MAXR = 72 = XE_SEED_MAX
    static constexpr int MAXR = 72;
    struct SeedIn { int32_t tok[MAXR]; int32_t pos[MAXR]; int64_t cell[MAXR]; int32_t n_kv; int32_t m; };
    SeedIn * seed_dev = nullptr, * seed_host = nullptr;
    float * en = nullptr, * hn = nullptr, * attn_o = nullptr, * up = nullptr, * gt = nullptr, * dn = nullptr;

    // the SWA cache's cells: the host mirror of the server's table (which cell each row takes, n_kv) and its device copy
    // (the mask reads it). Each decode carries the table's changes since the previous one (a seq_rm before the seed, its
    // own rows, their purges) as (cell, position) pairs in the input block's pool, applied by the mask kernel.
    Ring ring;
    int32_t * tab = nullptr;         // [kv_cells] device: each cell's position, -1 empty
    struct Delta { int32_t cell; int32_t pos; };
    Delta * dl_host = nullptr, * dl_dev = nullptr;   // the pool, right after the input block (one upload)
    int     dl_cap = 0, dl_n = 0;
    // the server's table (cells [0, n) of pos, the rest empty) and head: the mirror and the device copy
    void ring_load(uint32_t head, const int32_t * pos, uint32_t n);
    int  plan_(int d, const int32_t * ps, int m, int64_t * cells);
    void upload_(cudaStream_t st, int d, const void * h, void * dv, size_t n);

    // the device draft chain
    int32_t  * col_ids  = nullptr;   // [n_cols] column -> token id (the head's d2t)
    int32_t  * rec      = nullptr;   // [MAXS][GGML_DRAFT_SAMPLE_OUT] every draw's record
    int32_t  * draw_out = nullptr;   // the draw's own record (the op's dst)
    int32_t  * col      = nullptr;   // [1] the last draw's column
    int32_t  * steps    = nullptr;   // {0, 1, ..., MAXS-1}: draw s's record row
    uint32_t * keys     = nullptr, * keys_host = nullptr;   // [MAXS][2] the coupled keys, low half first
    int      top_k = 20;
    float    top_p = 0.95f;
    int      n_steps = 4;            // the chain's draws per draft: the last one (a step, s == n_steps-1) takes the pruned FFN

    Drafter(const Model & target, const Model & drafter) : tgt(target), d(drafter) {}
    // bind mode: the server's SWA cache (q4_0 K/V of the drafter's one layer) instead of our own
    void * ext_k = nullptr, * ext_v = nullptr;
    bool init(int cells, cudaStream_t st);
    // one decode into slot `slot`: token `tok` (< 0: the last draw's, on the device) at position `pos` with feedback/feature
    // `g` (device, n_embd floats) -> logits, fb. draw >= 0: then draw into record row `draw` with keys[draw]. last: the
    // pruned FFN (the chain's last draw).
    void step(cudaStream_t st, int slot, int32_t tok, int32_t pos, const float * g, int draw = -1, bool last = false);
    // the round's seed decode: m rows toks[0..m) at pos0.. with g rows (device, [n_embd, m]); the last row's logits are drawn
    // into record row `draw`, its feedback left in fb. m == 1 is the plain (fused) step.
    void seed(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, int draw);
    // THE PROMPT'S CATCH-UP: n > 32 rows (tokens toks[r] at positions pos[r], host; g rows [n_embd, n], device) through
    // the layer as the server's prompt-width drafter graph runs them (MMQ q3_K matmuls, the q4_0 cache converted to f16
    // for the 64-column attention tile); the rows take their cells on the ring. No output row.
    struct PfBufs * pfb = nullptr;
    void prefill(cudaStream_t st, const int32_t * toks, const int32_t * pos, const float * g, int n);
    // a full draft: seed decode of m rows, then steps 1..n-1 from the device draws
    void draft(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, const uint64_t * keys, int n);

    // the same draft as ONE CUDA graph per shape (m, n, every decode's n_kv, the g buffer): the host fills the pinned inputs
    // (keys, seed rows, step slots) in a launch-free pass of draft(), then replays the graph -- its memcpy nodes read those
    // inputs when they run, so every round's values go in without a capture
    bool issue_ = true;                 // false: draft() only fills the host inputs and records the n_kv of each decode
    bool no_copy_ = false;              // true: the decodes do not upload their inputs (one upload of the block precedes)
    // every per-draft input in ONE block (pinned host + device): keys, the seed rows, the step slots -- one copy per draft
    struct DraftIO { uint32_t keys[2*MAXS]; SeedIn seed; In in[MAXS]; int32_t dl_off[MAXS + 1]; };
    DraftIO * io_host = nullptr, * io_dev = nullptr;
    cudaGraphExec_t capture_(cudaStream_t st, const std::function<void()> & issue);
    std::vector<int> kv_trace;
    std::unordered_map<uint64_t, cudaGraphExec_t> dgraphs;
    int n_dcaptures = 0;
    void draft_graph(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, const uint64_t * keys, int n);
};
