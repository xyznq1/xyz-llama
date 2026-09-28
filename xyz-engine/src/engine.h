#pragma once
// xyz-engine: the target's verify pass as its own runtime -- weights in one arena (model.h), every activation, cache and
// recurrent-state buffer allocated once here, and the pass issued as a fixed launch sequence of the fork's kernels
// (kernels.h). One pass = T tokens (T = 5, the served verify batch: last token + 4 drafts).
#include <cuda_fp16.h>

#include <unordered_map>
#include <vector>

#include "kernels.h"
#include "model.h"

struct Layer {
    bool attn = false;
    const float * attn_norm = nullptr, * post_norm = nullptr;
    // GDN
    const void * w_qkvz = nullptr;                 // [attn_qkv; attn_gate]  16384 x 5120 PTQ1_0 (contiguous)
    const void * w_alpha = nullptr, * w_beta = nullptr;   // bf16 48 x 5120
    const float * dt = nullptr, * a = nullptr, * conv_w = nullptr, * ssm_norm = nullptr;
    const void * w_ssm_out = nullptr;              // 5120 x 6144
    // attention
    const void * w_qkv = nullptr;                  // [attn_q; attn_k; attn_v] 14336 x 5120 (contiguous)
    const float * q_norm = nullptr, * k_norm = nullptr;
    const void * w_o = nullptr;                    // 5120 x 6144
    // FFN
    const void * w_gate_up = nullptr;              // [ffn_gate; ffn_up] 34816 x 5120 (contiguous)
    const void * w_down = nullptr;                 // 5120 x 17408
    // the siblings apart, when a merged matrix above is null: its members are not byte-adjacent (the server's buffer in
    // bind mode) -- one launch each, the same values (at <= 8 columns the PTQ1 kernel computes every row the same way
    // whatever the row count, fork llama-graph.cpp MERGED GATE+UP)
    const void * w_qkv_a = nullptr, * w_z = nullptr;             // attn_qkv, attn_gate
    const void * w_q = nullptr, * w_k = nullptr, * w_v = nullptr;
    const void * w_gate = nullptr, * w_up = nullptr;
    // state
    float * conv_state = nullptr, * conv_pack = nullptr;   // [C, 3], [C, P]
    float * ssm_state = nullptr;                           // [128, 128, 48]
    float * gdn_pack[2] = { nullptr, nullptr };            // ping-pong: read the previous pass's, write this pass's
    char * k_cache = nullptr, * v_cache = nullptr;         // xyzkv2 [1024, kv_size]
};

// what one pass reads from the host (one upload)
struct PassIn {
    int32_t tokens[8];
    int32_t pos[32];         // M-RoPE: t, h, w positions, then zeros (4 x T)
    int64_t kv_idx[8];       // cache rows written (= positions: one sequence, compact cells)
    int32_t conv_idx[4];     // rows of [state ++ pack] that precede the batch
    int32_t s_row[1];
    int32_t prefix_n[1];     // tokens replayed from the pack
    int32_t pos0;
    int32_t n_kv;
};

// BIND MODE: the server's memory the engine runs on (per model layer; nullptr where a layer has none). The recurrent
// pointers are sequence 0's row of each tensor; pk is the GDN pack row (the engine ping-pongs it with a scratch row)
struct EngineMem {
    int     kv_size = 0;
    void  * k[128] = {};
    void  * v[128] = {};
    float * conv[128] = {};
    float * ssm[128] = {};
    float * pk[128] = {};
    float * px[128] = {};
};

struct Engine {
    const Model & m;
    int T = 5;               // columns per pass (the buffers' width)
    int Tw = 5;              // THIS pass's columns (<= T): a round that drafts G < 4 verifies G + 1 (the budget end)
    int P = 5;               // replay pack capacity
    int kv_size = 0;         // attention cells
    float eps = 1e-6f;
    cudaStream_t st = nullptr;
    // The verify graph runs the GDN alpha/beta prologue
    // on side stream 0 beside the qkvz matmul; attention's V write and K write on side streams 0 and 1 beside the Q chain.
    // Fork/join by events -- captured into the pass's graph as edges. Same kernels, same buffers: the same bits.
    cudaStream_t side[2] = {};
    cudaEvent_t  ev_fork = nullptr, ev_join[2] = {};

    std::vector<Layer> L;
    const float * s5120 = nullptr, * s6144 = nullptr, * s17408 = nullptr;
    const void * tok_embd = nullptr, * w_out = nullptr;
    const float * out_norm = nullptr;
    eng::MRope rope = {};

    // activations
    float * emb = nullptr, * x = nullptr, * xr = nullptr, * norm_out = nullptr, * rot = nullptr, * mm_out = nullptr;
    float * qkv_mixed = nullptr, * z = nullptr, * gate = nullptr, * beta = nullptr, * conv_silu = nullptr, * qk_norm = nullptr;
    float * gdn_out = nullptr, * rot6k = nullptr, * gate_up = nullptr, * glu_rot = nullptr;
    float * q_full = nullptr, * k_cur = nullptr, * v_cur = nullptr, * q_rot = nullptr, * fa_out = nullptr;
    float * xf = nullptr, * h = nullptr, * h_rot = nullptr, * logits = nullptr;
    float * ones = nullptr;
    half *  mask = nullptr;
    char *  q8 = nullptr;
    PassIn * in_dev = nullptr, * in_host = nullptr;

    // The FC fold produces the drafter's g rows of this pass:
    // fc([the input of layer fc_layer ; output_norm(h)]) -- the head's fc (Q3_K [10240 -> 5120]) over [10240, T]
    const void * fc_w = nullptr;
    int     fc_type  = 0;
    int     fc_layer = -1;
    float * fold_cat = nullptr;   // [10240, T]
    float * g_rows   = nullptr;   // [5120, T] the drafter's g for the pass's positions
    char  * q8f      = nullptr;
    bool set_fold(const void * w, int type, int layer);

    // the device draft chain's round: when set, the pass takes tokens 1..dev_draft_n from the chain's records
    // (record s, int 0: the drawn id) -- the early verify's rows, with no host round trip
    const int32_t * dev_draft_rec = nullptr;
    int dev_draft_n = 0;

    int pass_no = 0;         // passes run (the replay pack parity)
    int64_t n_past = 0;      // tokens committed to the caches

    // CUDA graphs: the whole pass is one graph per (n_kv, pack parity) -- the only host-side values its launches take;
    // everything else a pass varies (tokens, positions, replay counts, cache rows) is read from in_dev
    std::unordered_map<int64_t, cudaGraphExec_t> graphs;
    int n_captures = 0;

    // THE PROMPT PATH (prefill.cu): ubatches of 17..pf_nmax tokens as the server's prompt graph runs them (MMQ matmuls,
    // cuBLAS bf16 gates, the conv chain, the positional-mask attention). prefill_ubatch processes n tokens at positions
    // n_past.. (one sequence, compact cells), advances the caches, the recurrent rows and n_past; the output norm of every
    // row (h_nextn) and the fold layer's input rows stay on the device for the drafter.
    struct PrefillBufs * pf = nullptr;
    int  pf_nmax = 512;
    bool prefill_init();
    bool prefill_ubatch(const int32_t * tokens_host, int n, bool want_logits = false, int pending = 0);
    bool prefill_fold(int n, float * g_dev);   // the xyz encoder over the last prompt ubatch's rows (MMQ fc)
    // a prompt chunk of 1..16 tokens, as the server runs it: the verify-width graph WITHOUT packing (PTQ1 mma matmuls on
    // the q8 twins, mmvf bf16 gates, the unarmed conv + recurrence -- replaying `pending` packed rows first, as either
    // path does for the first batch after a response -- the fused Q/K/V chains, the f16 mask); the last row's logits land
    // in `logits` when want_logits
    bool prefill_small(const int32_t * tokens_host, int n, bool want_logits, int pending = 0);
    float * pf_g_small() const;   // the last verify-width chunk's fold rows g [5120, n] (device; set_fold armed)

    const EngineMem * ext = nullptr;   // bind mode: set before init()
    explicit Engine(const Model & model) : m(model) {}
    bool init(cudaStream_t stream);
    // one verify-shaped pass over T tokens at positions n_past.. with `pending` tokens of the previous pass replayed
    // (all T when every token was kept); logits [n_vocab, T] land in `logits`
    void pass(const int32_t * tokens, int pending, int width = -1);   // width: this pass's columns (default T)
    void issue(int n_kv, int par);   // the pass's launches (captured once per key)
};
