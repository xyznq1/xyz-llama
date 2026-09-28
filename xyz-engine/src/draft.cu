// xyz-engine drafter step: the server's plain drafter decode, one token, and the device draft
// chain (seed draw + steps fed from the previous draw on the device, the last one through the pruned FFN).
#include "draft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static __global__ void k_mul_signs_d(const float * __restrict__ x, const float * __restrict__ s, float * __restrict__ y, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * s[i];
    }
}

// the gated attention's gate half out of [q | gate] per head (ggml_cont of the strided view: an exact copy)
static __global__ void k_gate_cont(const float * __restrict__ qfull, float * __restrict__ gate, const int n_head) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_head*256) {
        gate[i] = qfull[(i / 256)*512 + 256 + (i % 256)];
    }
}

// the SWA mask over the ring (llama_kv_cache's STANDARD window): the device table of cell positions first takes decode d's
// changes (dl[dl_off[d] .. dl_off[d+1]): cell -> position, -1 empty, each cell once), then cell j is visible to row r (at
// pos[r]) iff 0 <= pos_j <= pos[r] and pos[r] - pos_j < n_swa. Every cell's entry is refreshed, not only the n_kv the
// attention reads (a later, wider decode reads the rest). Row 0 writes the table; another row may read either value of a
// changed cell and applies the same change over it.
static __global__ void k_ring_mask(half * __restrict__ mask, int32_t * tab, const Drafter::Delta * __restrict__ dl,
                                   const int32_t * __restrict__ dl_off, const int d, const int32_t * __restrict__ pos,
                                   const int32_t * __restrict__ n_kv_p, const int cells, const int n_swa) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (j >= cells) {
        return;
    }
    int32_t pj = tab[j];
    const int e = dl_off[d + 1];
    for (int i = dl_off[d]; i < e; ++i) {
        if (dl[i].cell == j) {
            pj = dl[i].pos;
        }
    }
    if (r == 0) {
        tab[j] = pj;
    }
    const int n_kv = *n_kv_p;
    if (j < n_kv) {
        const int p = pos[r];
        mask[(int64_t) r*n_kv + j] = (pj >= 0 && pj <= p && p - pj < n_swa) ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// m-row variants: the embedding's signs per row, [enorm ; hnorm] per row (ggml_concat dim 0: a copy), the residual adds
// (k_bin_bcast op_add: one IEEE add per element), and the SWA mask of m rows (row r at pos[r]; causal inside the batch)
static __global__ void k_mul_signs_rows(const float * __restrict__ x, const float * __restrict__ s, float * __restrict__ y,
                                        const int n, const int rows) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n*rows) {
        y[i] = x[i] * s[i % n];
    }
}

static __global__ void k_concat_rows(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst,
                                     const int n, const int rows) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < 2*n*rows) {
        const int r = i / (2*n), c = i % (2*n);
        dst[i] = c < n ? a[r*n + c] : b[r*n + c - n];
    }
}

static __global__ void k_add(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

// a chain step's token: the previous draw's column through the head's column -> id map (the server reads the column's row
// of its compact copy of the target's token_embd -- the same bytes as the id's row of the table)
static __global__ void k_col_tok(const int32_t * __restrict__ col_ids, const int32_t * __restrict__ col, Drafter::In * __restrict__ in) {
    in->tok = col_ids[col[0]];
}

static void * dmalloc(size_t bytes) {
    void * p = nullptr;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemset(p, 0, bytes));
    return p;
}

bool Drafter::init(int cells, cudaStream_t st) {
    eng::fork_runtime_init(st);   // the host runtime the launchers read (device info; the context's cuBLAS handle)
    eng::fa_init();
    kv_cells = cells;
    eps = d.hp.rms_eps;
    const std::string p = "blk.0.";
    enorm     = (const float *) d.get(p + "nextn.enorm.weight").data;
    hnorm     = (const float *) d.get(p + "nextn.hnorm.weight").data;
    eh_proj   = d.get(p + "nextn.eh_proj.weight").data;
    attn_norm = (const float *) d.get(p + "attn_norm.weight").data;
    wq        = d.get(p + "attn_q.weight").data;
    wk        = d.get(p + "attn_k.weight").data;
    wv        = d.get(p + "attn_v.weight").data;
    wo        = d.get(p + "attn_output.weight").data;
    q_norm    = (const float *) d.get(p + "attn_q_norm.weight").data;
    k_norm    = (const float *) d.get(p + "attn_k_norm.weight").data;
    ffn_norm  = (const float *) d.get(p + "ffn_norm.weight").data;
    ffn_gate  = d.get(p + "ffn_gate.weight").data;
    ffn_up    = d.get(p + "ffn_up.weight").data;
    ffn_down  = d.get(p + "ffn_down.weight").data;
    n_ff      = (int) d.get(p + "ffn_up.weight").ne[1];
    if (const Weight * u = d.find(p + "ffn_up.last.weight")) {   // the 3+1 head's pruned last-draw FFN
        ffn_up_l   = u->data;
        ffn_gate_l = d.get(p + "ffn_gate.last.weight").data;
        ffn_down_l = d.get(p + "ffn_down.last.weight").data;
        ffn_norm_l = (const float *) d.get(p + "ffn_norm.last.weight").data;
        n_ff_l     = (int) u->ne[1];
    }
    out_norm  = (const float *) d.get("output_norm.weight").data;
    head      = d.get("output.weight").data;
    n_cols    = (int) d.get("output.weight").ne[1];
    t_q3k     = (int) d.get(p + "attn_q.weight").type;
    // the MMQ activation buffer for the widest batch (a 512-row prompt ubatch; the captured seed decode's <= MAXR rows)
    eng::mmq_reserve(512, std::max<int64_t>(2*5120, n_ff));
    for (size_t i = 0; i < tgt.hp.had_widths.size(); ++i) if (tgt.hp.had_widths[i] == 5120) t_s5120 = tgt.had_signs_dev[i];
    for (size_t i = 0; i < d.hp.had_widths.size(); ++i)   if (d.hp.had_widths[i] == 5120)   d_s5120 = d.had_signs_dev[i];
    if (!t_s5120 || !d_s5120) {
        fprintf(stderr, "drafter: missing a 5120-wide Hadamard sign vector\n");
        return false;
    }
    const auto f = [](size_t n) { return (float *) dmalloc(n*MAXR*sizeof(float)); };   // every row buffer holds MAXR rows
    emb = f(5120); emb_rot = f(5120); emb_s = f(5120); cat = f(10240); fused = f(5120); cur = f(5120);
    qfull = f(12288); q = f(6144); qrot = f(6144); k = f(1024); kn = f(1024); krot = f(1024); v = f(1024); vrot = f(1024);
    fa = f(6144); fa_rot = f(6144); gate_c = f(6144); gated = f(6144); ffn_inp = f(5120); ffn_n = f(5120);
    glu = f((size_t) std::max(n_ff, n_ff_l));
    out = f(5120); fb = f(5120); h_rot = f(5120); logits = f((size_t) n_cols);
    en = f(5120); hn = f(5120); attn_o = f(5120); up = f((size_t) n_ff); gt = f((size_t) n_ff); dn = f(5120);
    q8  = (char *) dmalloc(1u << 20);   // q8_1 of the MMVQ widths (mm_q: up to 6 rows of 17,408; MMQ takes the pool)
    q8h = (char *) dmalloc(1u << 16);
    mask = (half *) dmalloc((size_t) kv_cells*MAXR*sizeof(half));
    // the input block, then the table-change pool: a draft's decodes change at most every cell each
    dl_cap = MAXS*kv_cells;
    const size_t io_bytes = sizeof(DraftIO) + (size_t) dl_cap*sizeof(Delta);
    io_dev = (DraftIO *) dmalloc(io_bytes);
    CK(cudaMallocHost(&io_host, io_bytes));
    memset(io_host, 0, io_bytes);
    dl_host = (Delta *) (io_host + 1);
    dl_dev  = (Delta *) (io_dev + 1);
    tab = (int32_t *) dmalloc((size_t) kv_cells*sizeof(int32_t));
    ring_load(0, nullptr, 0);   // a fresh cache until the server's table is loaded
    seed_dev = &io_dev->seed;
    seed_host = &io_host->seed;
    kc = ext_k;
    vc = ext_v;
    if (kc == nullptr || vc == nullptr) {
        fprintf(stderr, "drafter: bind: missing K/V cache\n");
        return false;
    }
    in_dev  = io_dev->in;
    in_host = io_host->in;

    // the device draft chain: column c is token d2t[c] (common/speculative.cpp: col_ids[head_idx[i]] = head_ids[i])
    const Weight & d2t = d.get("d2t");
    if (d2t.type != GGML_TYPE_I64 || d2t.ne[0] != n_cols) {
        fprintf(stderr, "drafter: d2t is not I64 [%d]\n", n_cols);
        return false;
    }
    std::vector<int64_t> raw((size_t) n_cols);
    CK(cudaMemcpy(raw.data(), d2t.data, raw.size()*sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::vector<int32_t> ids(raw.begin(), raw.end());
    col_ids = (int32_t *) dmalloc(ids.size()*sizeof(int32_t));
    CK(cudaMemcpy(col_ids, ids.data(), ids.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    rec      = (int32_t *) dmalloc((size_t) MAXS*GGML_DRAFT_SAMPLE_OUT*sizeof(int32_t));
    draw_out = (int32_t *) dmalloc(GGML_DRAFT_SAMPLE_OUT*sizeof(int32_t));
    col      = (int32_t *) dmalloc(sizeof(int32_t));
    keys      = io_dev->keys;
    keys_host = io_host->keys;
    int32_t sidx[MAXS];
    for (int s = 0; s < MAXS; ++s) sidx[s] = s;
    steps = (int32_t *) dmalloc(sizeof(sidx));
    CK(cudaMemcpy(steps, sidx, sizeof(sidx), cudaMemcpyHostToDevice));
    return true;
}

void Drafter::ring_load(uint32_t h, const int32_t * p, uint32_t n) {
    ring.init((uint32_t) kv_cells, (uint32_t) n_swa, h, p, n);
    CK(cudaMemcpy(tab, ring.pos.data(), (size_t) kv_cells*sizeof(int32_t), cudaMemcpyHostToDevice));
}

// decode d of a draft (0: the seed, or a first plain step) on the ring: its m rows take the cells the server's find_slot
// would give them; the table's changes since the previous decode go into the pool as this decode's slice; its n_kv
int Drafter::plan_(int d, const int32_t * ps, int m, int64_t * cells) {
    uint32_t idx[MAXR];
    if (!ring.find_slot(m, idx)) {
        fprintf(stderr, "drafter: no cells for %d rows (%zu of %u used)\n", m, ring.used.size(), ring.size);
        abort();
    }
    ring.apply(idx, ps, m);
    for (int r = 0; r < m; ++r) {
        cells[r] = idx[r];
    }
    if (d == 0) {
        dl_n = 0;
        kv_trace.clear();
    }
    io_host->dl_off[d] = dl_n;
    dl_n += ring.take(dl_host + dl_n);
    io_host->dl_off[d + 1] = dl_n;
    const int n_kv = ring.n_kv();
    kv_trace.push_back(n_kv);
    return n_kv;
}

// an eager decode's inputs: its slot, the pool offsets and its slice of the pool
void Drafter::upload_(cudaStream_t st, int d, const void * h, void * dv, size_t n) {
    CK(cudaMemcpyAsync(dv, h, n, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(io_dev->dl_off, io_host->dl_off, sizeof(io_host->dl_off), cudaMemcpyHostToDevice, st));
    const int a = io_host->dl_off[d], b = io_host->dl_off[d + 1];
    if (b > a) {
        CK(cudaMemcpyAsync(dl_dev + a, dl_host + a, (size_t) (b - a)*sizeof(Delta), cudaMemcpyHostToDevice, st));
    }
}

void Drafter::step(cudaStream_t st, int slot, int32_t tok, int32_t pos, const float * g, int draw, bool last) {
    In * ih = in_host + slot;
    In * in = in_dev + slot;
    if (!no_copy_) {   // a capture pass replays the prepare pass's plan: the block already holds it
        ih->tok = tok; ih->pos = pos;
        ih->n_kv = plan_(slot, &pos, 1, &ih->cell);
    }
    const int n_kv = ih->n_kv;
    if (!issue_) {
        return;
    }
    if (!no_copy_) {
        upload_(st, slot, ih, in, sizeof(In));
    }
    if (tok < 0) {
        k_col_tok<<<1, 1, 0, st>>>(col_ids, col, in);
    }
    const size_t row = 1024/32*18;
    enum { SWIGLU = 2 };   // ggml_glu_op: REGLU 0, GEGLU 1, SWIGLU 2

    // embedding (the target's rotated table): rows, butterfly, signs
    eng::get_rows_ptq1(st, tgt.get("token_embd.weight").data, 5120, tgt.hp.n_vocab, &in->tok, 1, emb);
    eng::fwht_block(st, emb, emb_rot, 5, nullptr, 1, nullptr);
    k_mul_signs_d<<<20, 256, 0, st>>>(emb_rot, t_s5120, emb_s, 5120);
    // [enorm(emb) ; hnorm(g)] -> eh_proj
    eng::rms_norm_mul(st, emb_s, enorm, cat, 5120, 1, eps);
    eng::rms_norm_mul(st, g, hnorm, cat + 5120, 5120, 1, eps);
    auto mv = [&](const void * w, const float * x, int K, int n, float * dst, const void * gate = nullptr, int glu = 0,
                  const float * bias = nullptr) {
        eng::mmvq_fast(st, t_q3k, w, x, K, n, dst, q8, gate, glu, bias);
    };
    mv(eh_proj, cat, 10240, 5120, fused);
    eng::rms_norm_mul(st, fused, attn_norm, cur, 5120, 1, eps);
    // Q (with gate), K, V -- one quantization and one launch
    eng::mmvq_qkv(st, 8, t_q3k, cur, 5120, q8, wq, 12288, qfull, wk, 1024, k, wv, 1024, v);
    eng::rms_norm_mul_rope(st, qfull, 512, 24, q_norm, eps, q, &in->pos, 64, d.hp.rope_base, 262144);
    eng::fwht_rows(st, q, qrot, 256, 24);
    eng::rms_norm_mul_rope(st, k, 256, 4, k_norm, eps, kn, &in->pos, 64, d.hp.rope_base, 262144);
    eng::fwht_rows(st, kn, krot, 256, 4);
    eng::fwht_rows(st, v, vrot, 64, 16);
    eng::set_rows_q4_0(st, vrot, 1024, 1, &in->cell, vc, (int64_t) row);
    eng::set_rows_q4_0(st, krot, 1024, 1, &in->cell, kc, (int64_t) row);
    // attention over the window, inverse V rotation, gate
    k_ring_mask<<<dim3((kv_cells + 255)/256, 1), 256, 0, st>>>(mask, tab, dl_dev, io_dev->dl_off, slot, &in->pos, &in->n_kv,
                                                                 kv_cells, n_swa);
    eng::flash_attn_q4_0(st, qrot, 1, 24, kc, vc, n_kv, kv_cells, mask, 1.0f/16.0f, fa);
    eng::fwht_rows(st, fa, fa_rot, 64, 96);
    k_gate_cont<<<24, 256, 0, st>>>(qfull, gate_c, 24);
    eng::sigmoid_gate(st, gate_c, fa_rot, gated, 6144, 1);
    // out-projection + residual, FFN (fused GLU) + residual; the chain's last draw takes the pruned FFN
    const bool pl = last && n_ff_l > 0;
    mv(wo, gated, 6144, 5120, ffn_inp, nullptr, 0, fused);
    eng::rms_norm_mul(st, ffn_inp, pl ? ffn_norm_l : ffn_norm, ffn_n, 5120, 1, eps);
    mv(pl ? ffn_up_l : ffn_up, ffn_n, 5120, pl ? n_ff_l : n_ff, glu, pl ? ffn_gate_l : ffn_gate, SWIGLU);
    mv(pl ? ffn_down_l : ffn_down, glu, pl ? n_ff_l : n_ff, 5120, out, nullptr, 0, ffn_inp);
    // output norm (the feedback) + signs + Hadamard (+ q8 twin) -> the PTQ1 head
    eng::norm_fwht(st, out, nullptr, nullptr, out_norm, d_s5120, fb, h_rot, q8h, 1, eps);
    eng::ptq1_own(st, head, q8h, 5120, n_cols, 1, logits, n_cols, 1);
    if (draw >= 0) {
        eng::draft_draw(st, logits, col_ids, keys + 2*draw, draw_out, steps + draw, rec,
                        (int64_t) GGML_DRAFT_SAMPLE_OUT*sizeof(int32_t), col, n_cols, top_k, top_p);
    }
    CK(cudaGetLastError());
}

void Drafter::seed(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, int draw) {
    if (m == 1) {
        step(st, 0, toks[0], pos0, g, draw, false);
        return;
    }
    SeedIn * sh = seed_host;
    if (!no_copy_) {
        for (int r = 0; r < m; ++r) {
            sh->tok[r] = toks[r]; sh->pos[r] = pos0 + r;
        }
        sh->m = m;
        sh->n_kv = plan_(0, sh->pos, m, sh->cell);
    }
    const int n_kv = sh->n_kv;
    if (!issue_) {
        return;
    }
    if (!no_copy_) {
        upload_(st, 0, sh, seed_dev, sizeof(SeedIn));
    }
    const size_t row = 1024/32*18;
    const int E = 5120;

    // embedding: rows, butterfly, signs; [enorm ; hnorm] -> eh_proj -- every matmul unfused, routed by m (mm_q: MMVQ
    // to 6 rows, MMQ above), the attention by m too (native q4_0 to 32 rows, the f16 tile above)
    eng::get_rows_ptq1(st, tgt.get("token_embd.weight").data, E, tgt.hp.n_vocab, seed_dev->tok, m, emb);
    eng::fwht_block(st, emb, emb_rot, (int64_t) 5*m, nullptr, 1, nullptr);
    k_mul_signs_rows<<<(E*m + 255)/256, 256, 0, st>>>(emb_rot, t_s5120, emb_s, E, m);
    eng::rms_norm_mul(st, emb_s, enorm, en, E, m, eps);
    eng::rms_norm_mul(st, g, hnorm, hn, E, m, eps);
    k_concat_rows<<<(2*E*m + 255)/256, 256, 0, st>>>(en, hn, cat, E, m);
    eng::mm_q(st, t_q3k, eh_proj, cat, 2*E, E, m, fused, q8);
    eng::rms_norm_mul(st, fused, attn_norm, cur, E, m, eps);
    // Q (with gate), K, V
    eng::mm_q(st, t_q3k, wq, cur, E, 12288, m, qfull, q8);
    eng::rms_norm_mul_rope(st, qfull, 512, 24, q_norm, eps, q, seed_dev->pos, 64, d.hp.rope_base, 262144, m, 12288);
    eng::fwht_rows(st, q, qrot, 256, (int64_t) 24*m);
    eng::mm_q(st, t_q3k, wk, cur, E, 1024, m, k, q8);
    eng::rms_norm_mul_rope(st, k, 256, 4, k_norm, eps, kn, seed_dev->pos, 64, d.hp.rope_base, 262144, m, 1024);
    eng::fwht_rows(st, kn, krot, 256, (int64_t) 4*m);
    eng::mm_q(st, t_q3k, wv, cur, E, 1024, m, v, q8);
    eng::fwht_rows(st, v, vrot, 64, (int64_t) 16*m);
    eng::set_rows_q4_0(st, vrot, 1024, m, seed_dev->cell, vc, (int64_t) row);
    eng::set_rows_q4_0(st, krot, 1024, m, seed_dev->cell, kc, (int64_t) row);
    // attention (m query rows), inverse V rotation, gate
    k_ring_mask<<<dim3((kv_cells + 255)/256, m), 256, 0, st>>>(mask, tab, dl_dev, io_dev->dl_off, 0, seed_dev->pos,
                                                                 &seed_dev->n_kv, kv_cells, n_swa);
    eng::flash_attn_q4_0_prompt(st, qrot, m, 24, kc, vc, n_kv, kv_cells, mask, 1.0f/16.0f, fa);
    eng::fwht_rows(st, fa, fa_rot, 64, (int64_t) 96*m);
    k_gate_cont<<<24*m, 256, 0, st>>>(qfull, gate_c, 24*m);
    eng::sigmoid_gate(st, gate_c, fa_rot, gated, 6144, m);
    // out-projection, + residual; FFN: up, gate, SWIGLU, down, + residual (all unfused at m rows)
    eng::mm_q(st, t_q3k, wo, gated, 6144, E, m, attn_o, q8);
    k_add<<<(E*m + 255)/256, 256, 0, st>>>(attn_o, fused, ffn_inp, E*m);
    eng::rms_norm_mul(st, ffn_inp, ffn_norm, ffn_n, E, m, eps);
    eng::mm_q(st, t_q3k, ffn_up, ffn_n, E, n_ff, m, up, q8);
    eng::mm_q(st, t_q3k, ffn_gate, ffn_n, E, n_ff, m, gt, q8);
    eng::silu_gate(st, gt, up, glu, n_ff, m);
    eng::mm_q(st, t_q3k, ffn_down, glu, n_ff, E, m, dn, q8);
    k_add<<<(E*m + 255)/256, 256, 0, st>>>(dn, ffn_inp, out, E*m);
    // the output row (the seed, the last): output norm (feedback) + signs + Hadamard (+ q8 twin) -> head -> draw
    eng::norm_fwht(st, out + (size_t) (m - 1)*E, nullptr, nullptr, out_norm, d_s5120, fb, h_rot, q8h, 1, eps);
    eng::ptq1_own(st, head, q8h, E, n_cols, 1, logits, n_cols, 1);
    if (draw >= 0) {
        eng::draft_draw(st, logits, col_ids, keys + 2*draw, draw_out, steps + draw, rec,
                        (int64_t) GGML_DRAFT_SAMPLE_OUT*sizeof(int32_t), col, n_cols, top_k, top_p);
    }
    CK(cudaGetLastError());
}

void Drafter::draft(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, const uint64_t * ks, int n) {
    for (int s = 0; s < n; ++s) {
        keys_host[2*s]     = (uint32_t) (ks[s] & 0xFFFFFFFFull);
        keys_host[2*s + 1] = (uint32_t) (ks[s] >> 32);
    }
    if (issue_ && !no_copy_) {
        CK(cudaMemcpyAsync(keys, keys_host, (size_t) 2*n*sizeof(uint32_t), cudaMemcpyHostToDevice, st));
    }
    seed(st, toks, pos0, g, m, 0);
    const int32_t P = pos0 + m - 1;   // the seed row's position
    for (int s = 1; s < n; ++s) {
        step(st, s, -1, P + s, fb, s, s == n_steps - 1);   // token and g: the previous draw's, on the device
    }
}

cudaGraphExec_t Drafter::capture_(cudaStream_t st, const std::function<void()> & issue) {
    cudaGraph_t gr = nullptr;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
    issue();
    CK(cudaStreamEndCapture(st, &gr));
    cudaGraphExec_t ge = nullptr;
    CK(cudaGraphInstantiate(&ge, gr, 0));
    CK(cudaGraphDestroy(gr));
    n_dcaptures++;
    return ge;
}

void Drafter::draft_graph(cudaStream_t st, const int32_t * toks, int32_t pos0, const float * g, int m, const uint64_t * ks, int n) {
    // 1. the host inputs of this draft, no launches: the ring places every decode's rows (each decode's n_kv, its slice of
    //    table changes)
    issue_ = false;
    draft(st, toks, pos0, g, m, ks, n);
    issue_ = true;
    // 2. one upload of the input block and the pool's used part; the graphs hold no copies (a host-memory copy node is slow
    //    to launch on WDDM)
    CK(cudaMemcpyAsync(io_dev, io_host, sizeof(DraftIO) + (size_t) dl_n*sizeof(Delta), cudaMemcpyHostToDevice, st));
    // 3. two graphs: the seed decode (m rows) and the chain's steps -- the small first launch starts the GPU sooner, and the
    //    second is issued while the seed runs
    const auto key_of = [](std::initializer_list<uint64_t> v) {
        uint64_t k = 1469598103934665603ull;
        for (uint64_t x : v) k = (k ^ x) * 1099511628211ull;
        return k;
    };
    const uint64_t k_seed = key_of({ 1, (uint64_t) m, (uint64_t) (uintptr_t) g, (uint64_t) kv_trace[0] });
    auto it = dgraphs.find(k_seed);
    if (it == dgraphs.end()) {
        no_copy_ = true;
        it = dgraphs.emplace(k_seed, capture_(st, [&] { seed(st, toks, pos0, g, m, 0); })).first;
        no_copy_ = false;
    }
    CK(cudaGraphLaunch(it->second, st));
    if (n > 1) {
        uint64_t k_steps = key_of({ 2, (uint64_t) n, (uint64_t) n_steps });
        for (size_t i = 1; i < kv_trace.size(); ++i) k_steps = (k_steps ^ (uint64_t) kv_trace[i]) * 1099511628211ull;
        auto is = dgraphs.find(k_steps);
        if (is == dgraphs.end()) {
            const int32_t P = pos0 + m - 1;
            no_copy_ = true;
            is = dgraphs.emplace(k_steps, capture_(st, [&] {
                for (int s = 1; s < n; ++s) {
                    step(st, s, -1, P + s, fb, s, s == n_steps - 1);
                }
            })).first;
            no_copy_ = false;
        }
        CK(cudaGraphLaunch(is->second, st));
    }
}

// ---- the prompt's catch-up -------------------------------------------------------------------------------------------

struct PfBufs {
    int32_t * tok = nullptr, * pos = nullptr;
    int64_t * cell = nullptr;
    half    * mask = nullptr;
    float * emb = nullptr, * emb_rot = nullptr, * emb_s = nullptr, * en = nullptr, * hn = nullptr, * cat = nullptr,
          * fused = nullptr, * cur = nullptr, * qfull = nullptr, * q = nullptr, * qrot = nullptr, * k = nullptr, * kn = nullptr,
          * krot = nullptr, * v = nullptr, * vrot = nullptr, * fa = nullptr, * fa_rot = nullptr, * gate_c = nullptr,
          * gated = nullptr, * attn_o = nullptr, * ffn_inp = nullptr, * ffn_n = nullptr, * up = nullptr, * gt = nullptr,
          * glu = nullptr, * dn = nullptr, * out = nullptr;
};

// the SWA mask of a prompt-width decode from the uploaded table: cell j visible to row r iff 0 <= pos_j <= pos[r] and
// pos[r] - pos_j < n_swa (the host mask's values, llama_kv_cache STANDARD window)
static __global__ void k_pf_ring_mask(half * __restrict__ mask, const int32_t * __restrict__ tab, const int32_t * __restrict__ pos,
                                      const int n_kv, const int n_swa) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (j >= n_kv) {
        return;
    }
    const int32_t pj = tab[j], p = pos[r];
    mask[(int64_t) r*n_kv + j] = (pj >= 0 && pj <= p && p - pj < n_swa) ? __float2half(0.0f) : __float2half(-INFINITY);
}

// a catch-up decode of n prompt rows (2..512), as llama_decode(ctx_dft) runs one ubatch of them: the matmuls routed by
// width (mm_q: MMVQ to 6 rows, MMQ above), the attention as fattn.cu routes it (flash_attn_q4_0_prompt: native q4_0 up
// to 32 rows, the f16 tile above). One row is a different graph (the fused single-token decode) and is not taken.
void Drafter::prefill(cudaStream_t st, const int32_t * toks, const int32_t * pos, const float * g, const int n) {
    constexpr int N = 512;
    if (n < 2 || n > N) {
        fprintf(stderr, "drafter: prefill takes 2..%d rows (%d)\n", N, n);
        abort();
    }
    const int E = 5120;
    if (pfb == nullptr) {
        pfb = new PfBufs();
        PfBufs & b = *pfb;
        const auto f = [&](size_t w) { float * p = nullptr; CK(cudaMalloc(&p, w*N*sizeof(float))); return p; };
        CK(cudaMalloc(&b.tok, N*sizeof(int32_t)));
        CK(cudaMalloc(&b.pos, N*sizeof(int32_t)));
        CK(cudaMalloc(&b.cell, N*sizeof(int64_t)));
        CK(cudaMalloc(&b.mask, (size_t) kv_cells*N*sizeof(half)));
        b.emb = f(E); b.emb_rot = f(E); b.emb_s = f(E); b.en = f(E); b.hn = f(E); b.cat = f(2*E); b.fused = f(E); b.cur = f(E);
        b.qfull = f(12288); b.q = f(6144); b.qrot = f(6144); b.k = f(1024); b.kn = f(1024); b.krot = f(1024); b.v = f(1024);
        b.vrot = f(1024); b.fa = f(6144); b.fa_rot = f(6144); b.gate_c = f(6144); b.gated = f(6144); b.attn_o = f(E);
        b.ffn_inp = f(E); b.ffn_n = f(E); b.up = f((size_t) n_ff); b.gt = f((size_t) n_ff); b.glu = f((size_t) n_ff);
        b.dn = f(E); b.out = f(E);
    }
    PfBufs & b = *pfb;

    // the rows' cells on the ring (the server's find_slot + apply_ubatch); the whole table goes up with the inputs
    std::vector<uint32_t> idx(n);
    if (!ring.find_slot(n, idx.data())) {
        fprintf(stderr, "drafter: no cells for %d rows (%zu of %u used)\n", n, ring.used.size(), ring.size);
        abort();
    }
    ring.apply(idx.data(), pos, n);
    {
        std::vector<Delta> scratch((size_t) kv_cells + 1);
        ring.take(scratch.data());   // the table is uploaded whole below, so no decode replays these changes
    }
    const int n_kv = ring.n_kv();
    kv_trace.push_back(n_kv);
    std::vector<int64_t> cells(n);
    for (int r = 0; r < n; ++r) {
        cells[r] = idx[r];
    }
    CK(cudaMemcpyAsync(b.tok, toks, (size_t) n*sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(b.pos, pos, (size_t) n*sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(b.cell, cells.data(), (size_t) n*sizeof(int64_t), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(tab, ring.pos.data(), (size_t) kv_cells*sizeof(int32_t), cudaMemcpyHostToDevice, st));
    CK(cudaStreamSynchronize(st));   // the host vectors above go out of scope

    const size_t row = 1024/32*18;
    // embedding: rows, butterfly, signs; [enorm ; hnorm] -> eh_proj
    eng::get_rows_ptq1(st, tgt.get("token_embd.weight").data, E, tgt.hp.n_vocab, b.tok, n, b.emb);
    eng::fwht_block(st, b.emb, b.emb_rot, (int64_t) 5*n, nullptr, 1, nullptr);
    k_mul_signs_rows<<<(E*n + 255)/256, 256, 0, st>>>(b.emb_rot, t_s5120, b.emb_s, E, n);
    eng::rms_norm_mul(st, b.emb_s, enorm, b.en, E, n, eps);
    eng::rms_norm_mul(st, g, hnorm, b.hn, E, n, eps);
    k_concat_rows<<<(2*E*n + 255)/256, 256, 0, st>>>(b.en, b.hn, b.cat, E, n);
    eng::mm_q(st, t_q3k, eh_proj, b.cat, 2*E, E, n, b.fused, q8);
    eng::rms_norm_mul(st, b.fused, attn_norm, b.cur, E, n, eps);
    // Q (with gate), K, V: MMQ, the fused norm + NEOX rope, the KV rotations, q4_0 rows into the ring's cells
    eng::mm_q(st, t_q3k, wq, b.cur, E, 12288, n, b.qfull, q8);
    eng::rms_norm_mul_rope(st, b.qfull, 512, 24, q_norm, eps, b.q, b.pos, 64, d.hp.rope_base, 262144, n, 12288);
    eng::fwht_rows(st, b.q, b.qrot, 256, (int64_t) 24*n);
    eng::mm_q(st, t_q3k, wk, b.cur, E, 1024, n, b.k, q8);
    eng::rms_norm_mul_rope(st, b.k, 256, 4, k_norm, eps, b.kn, b.pos, 64, d.hp.rope_base, 262144, n, 1024);
    eng::fwht_rows(st, b.kn, b.krot, 256, (int64_t) 4*n);
    eng::mm_q(st, t_q3k, wv, b.cur, E, 1024, n, b.v, q8);
    eng::fwht_rows(st, b.v, b.vrot, 64, (int64_t) 16*n);
    eng::set_rows_q4_0(st, b.krot, 1024, n, b.cell, kc, (int64_t) row);
    eng::set_rows_q4_0(st, b.vrot, 1024, n, b.cell, vc, (int64_t) row);
    // attention over the first n_kv cells (the SWA mask), inverse V rotation, gate
    k_pf_ring_mask<<<dim3((n_kv + 255)/256, n), 256, 0, st>>>(b.mask, tab, b.pos, n_kv, n_swa);
    eng::flash_attn_q4_0_prompt(st, b.qrot, n, 24, kc, vc, n_kv, kv_cells, b.mask, 1.0f/16.0f, b.fa);
    eng::fwht_rows(st, b.fa, b.fa_rot, 64, (int64_t) 96*n);
    k_gate_cont<<<24*n, 256, 0, st>>>(b.qfull, b.gate_c, 24*n);
    eng::sigmoid_gate(st, b.gate_c, b.fa_rot, b.gated, 6144, n);
    // out-projection + residual; FFN: gate, up, SWIGLU, down + residual
    eng::mm_q(st, t_q3k, wo, b.gated, 6144, E, n, b.attn_o, q8);
    k_add<<<(E*n + 255)/256, 256, 0, st>>>(b.attn_o, b.fused, b.ffn_inp, E*n);
    eng::rms_norm_mul(st, b.ffn_inp, ffn_norm, b.ffn_n, E, n, eps);
    eng::mm_q(st, t_q3k, ffn_gate, b.ffn_n, E, n_ff, n, b.gt, q8);
    eng::mm_q(st, t_q3k, ffn_up, b.ffn_n, E, n_ff, n, b.up, q8);
    eng::silu_gate(st, b.gt, b.up, b.glu, n_ff, n);
    eng::mm_q(st, t_q3k, ffn_down, b.glu, n_ff, E, n, b.dn, q8);
    k_add<<<(E*n + 255)/256, 256, 0, st>>>(b.dn, b.ffn_inp, b.out, E*n);
    CK(cudaGetLastError());
}
