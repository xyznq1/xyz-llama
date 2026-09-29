// xyz-engine C API: the Session bound to the host server's weights and memory.
#define XE_BUILD
#include "xyz_engine.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>

#include "draft.h"
#include "engine.h"
#include "session.h"

struct xe_ctx {
    cudaStream_t st = nullptr;
    Model tgt, dm;
    EngineMem mem;
    std::unique_ptr<Engine>  e;
    std::unique_ptr<Drafter> dr;
    std::unique_ptr<Session> ses;
    // the prompt: a batch's g rows and the drafter's rows (device)
    float * g_all = nullptr, * g_rows = nullptr;
    int     g_cap = 0;
};

extern "C" {

XE_API xe_ctx * xe_create(const xe_bind * b) {
    // RTX 40 series (sm_89) only: that's where we proved the engine's text is the default path's, word for word. On an
    // RTX 3080 Ti the text differed (and it was no faster), so everything else runs the default path, which is exact.
    int dev = 0, cc_major = 0, cc_minor = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
    if (cc_major != 8 || cc_minor != 9) {
        fprintf(stderr, "xyz-engine: only verified on RTX 40 series (sm_89), this GPU is sm_%d%d -- the default path runs\n",
                cc_major, cc_minor);
        return nullptr;
    }
    auto c = std::make_unique<xe_ctx>();
    CK(cudaStreamCreateWithFlags(&c->st, cudaStreamNonBlocking));
    c->tgt.bind_lookup = [b](const char * nm) { return b->tensor(b->user, 0, nm); };
    c->dm.bind_lookup  = [b](const char * nm) { return b->tensor(b->user, 1, nm); };
    // model 2: the host pointer of a target tensor llama keeps on the host -- the input embedding, read in place
    c->tgt.bind_lookup_host = [b](const char * nm) { return b->tensor(b->user, 2, nm); };
    if (!c->tgt.load(b->target_gguf, c->st) || !c->dm.load(b->drafter_gguf, c->st)) {
        return nullptr;
    }
    c->mem.kv_size = b->kv_size;
    for (int il = 0; il < 128; ++il) {
        c->mem.k[il] = b->k[il];   c->mem.v[il] = b->v[il];
        c->mem.conv[il] = b->conv[il]; c->mem.ssm[il] = b->ssm[il]; c->mem.pk[il] = b->pk[il]; c->mem.px[il] = b->px[il];
    }
    c->e = std::make_unique<Engine>(c->tgt);
    c->e->ext = &c->mem;
    if (!c->e->init(c->st)) {
        return nullptr;
    }
    const Weight & fc = c->dm.get("fc.weight");
    if (!c->e->set_fold(fc.data, (int) fc.type, 34)) {
        return nullptr;
    }
    c->dr = std::make_unique<Drafter>(c->tgt, c->dm);
    c->dr->ext_k = b->dk;
    c->dr->ext_v = b->dv;
    c->dr->n_swa = b->d_n_swa;
    if (!c->dr->init(b->d_kv_size, c->st)) {
        return nullptr;
    }
    c->ses = std::make_unique<Session>(*c->e, *c->dr);
    if (!c->ses->init(b->expf_table)) {
        return nullptr;
    }
    fprintf(stderr, "xyz-engine: bound (target %d layers, kv %d cells, drafter swa %d cells)\n", c->tgt.hp.n_layer,
            b->kv_size, b->d_kv_size);
    return c.release();
}

XE_API void xe_free(xe_ctx * c) {
    delete c;
}

XE_API void xe_rebind_rs(xe_ctx * c, float * const * conv, float * const * ssm, float * const * pk, float * const * px) {
    bool moved = false;
    for (int il = 0; il < c->tgt.hp.n_layer; ++il) {
        Layer & l = c->e->L[il];
        if (l.attn) continue;
        moved |= l.conv_state != conv[il] || l.ssm_state != ssm[il] || l.gdn_pack[0] != pk[il] || l.conv_pack != px[il];
        l.conv_state = conv[il]; l.ssm_state = ssm[il]; l.gdn_pack[0] = pk[il]; l.conv_pack = px[il];
    }
    if (moved) {   // the captured passes hold the old rows
        for (auto & kv : c->e->graphs) CK(cudaGraphExecDestroy(kv.second));
        c->e->graphs.clear();
        c->e->graph_used.clear();
    }
}

XE_API int xe_generate(xe_ctx * c, const xe_start * s, int32_t max_tokens, xe_round_fn cb, void * user, xe_end * out) {
    SessionStart ss;
    ss.p0 = s->p0; ss.id_last = s->id_last; ss.pending = s->pending; ss.m = s->m; ss.seed_pos0 = s->seed_pos0;
    ss.seed_g = s->seed_g; ss.cseed = s->cseed; ss.n_draws = s->n_draws;
    ss.accept_mode = s->accept_mode == 1 ? xe::ACC_PLAIN : xe::ACC_BLOCK;
    ss.budget_tail = s->budget_tail != 0;   // no dist_rng: the one-token end goes back to the server
    ss.chain_steps = s->chain_steps;
    if (s->cut != nullptr && s->n_cut > 16) {
        fprintf(stderr, "xyz-engine: xe_generate: %d trigger ids (16 at most) -- refused\n", s->n_cut);
        return 1;
    }
    ss.cut = s->cut; ss.n_cut = s->cut != nullptr ? s->n_cut : 0;
    c->ses->set_topk_skip(s->skip, s->n_skip);
    static_assert(XE_SEED_MAX == Drafter::MAXR, "the seed rows the API carries are the rows the drafter holds");
    for (int k = 0; k < s->m && k < XE_SEED_MAX; ++k) ss.seed_tok[k] = s->seed_tok[k];
    memcpy(ss.rng.mt, s->rng, sizeof(ss.rng.mt));
    ss.rng.idx = 624;   // the textual state is the last 624 outputs: the next draw twists them
    c->e->pass_no = 0;  // the server's pack row is the one the first pass reads (gdn_pack[0])
    c->dr->ring_load((uint32_t) s->d_head, s->d_pos, (uint32_t) s->d_n);
    const int cap0 = c->e->n_captures + c->dr->n_dcaptures;
    SessionEnd se;
    const bool ok = c->ses->run(ss, max_tokens, [&](int32_t * t, int & n, uint32_t one_mask, const int32_t * drafts,
                                                    uint32_t flags) {
        int32_t n32 = n;
        const bool go = cb(user, t, &n32, one_mask, drafts, c->ses->round_g, flags) != 0;
        n = n32;
        return go;
    }, se);
    CK(cudaStreamSynchronize(c->st));
    if (out) {
        out->rounds = se.rounds; out->n_tokens = se.n_tokens; out->p0 = se.p0; out->id_last = se.id_last;
        out->pending = se.pending; out->m = se.m; out->seed_pos0 = se.seed_pos0; out->rng_draws = se.rng_draws;
        out->short_rows = se.short_rows;
        out->tie_rows   = se.tie_rows;
        out->cut_rounds = se.cut_rounds;
        out->captures   = c->e->n_captures + c->dr->n_dcaptures - cap0;
        for (int k = 0; k < 8; ++k) out->seed_tok[k] = se.seed_tok[k];
    }
    return ok ? 0 : 1;
}

XE_API int xe_seed_rows(xe_ctx * c, float * dst, int32_t m) {
    CK(cudaMemcpy(dst, c->ses->d_seed_g, (size_t) m*5120*sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

XE_API void xe_logits_row(xe_ctx * c, int32_t r, float * dst) {
    c->ses->logits_row(r, dst);
}

XE_API void xe_flush_pack(xe_ctx * c) {
    if ((c->e->pass_no & 1) == 0) {
        return;   // the last pack already sits in the server's row
    }
    const size_t bytes = (size_t) (2*16*128 + 48*128 + 48 + 48)*c->e->P*sizeof(float);
    for (int il = 0; il < c->tgt.hp.n_layer; ++il) {
        Layer & l = c->e->L[il];
        if (!l.attn) {
            CK(cudaMemcpyAsync(l.gdn_pack[0], l.gdn_pack[1], bytes, cudaMemcpyDeviceToDevice, c->st));
        }
    }
    CK(cudaStreamSynchronize(c->st));
    c->e->pass_no = 0;
}

// ---- the prompt -----------------------------------------------------------------------------------------------------------

XE_API int xe_prefill(xe_ctx * c, const xe_prefill_in * in, xe_prefill_out * out) {
    Engine  & e  = *c->e;
    Drafter & dr = *c->dr;
    const int E = 5120;
    const int n = in->n, U = in->n_ubatch;
    // ---- what the engine reproduces (declined here = nothing written, the server runs the batch)
    if (n < 1 || U < 17 || U != in->n_ubatch_dft || U > e.pf_nmax || in->p0 < 0) {
        return 1;
    }
    // (a verify-width ubatch inside a prompt-width batch: the server folds no rows in its graph there -- the encoder does,
    // chunked by n_ubatch_dft = n_ubatch, so over the same rows at the same width with the same routing: pf_g_small's rows)
    // the recurrent cell's packed rows (the first batch after a response): the first ubatch replays them
    if (in->pending < 0 || in->pending > e.P) {
        return 5;
    }
    // xyz's ROLLBACK (process(): a batch at or before the deferred boundary is a new prompt over a reused slot -- the
    // server already cut both caches back): the stash keeps its rows below p0 - 1, and the row at p0 - 1 when it carries
    // this batch's first token; the boundary is forgotten. (The server's engine_state trimmed the stash below the old
    // boundary first; p0 - 1 < that boundary here, so the rows this keeps are the ones process() keeps.)
    int32_t n_stash = in->n_stash, pending_pos = in->pending_pos;
    if (pending_pos >= 0 && in->p0 <= pending_pos) {
        int32_t keep = 0;
        while (keep < n_stash && in->stash_pos[keep] < in->p0 - 1) {
            keep++;
        }
        if (keep < n_stash && in->stash_pos[keep] == in->p0 - 1 && in->stash_ids[keep] == in->tokens[0]) {
            keep++;
        }
        n_stash     = keep;
        pending_pos = -1;
    }
    dr.ring_load((uint32_t) in->d_head, in->d_pos, (uint32_t) in->d_n);
    int32_t stash_max_pos = -1;
    for (int k = 0; k < n_stash; ++k) {
        stash_max_pos = std::max(stash_max_pos, in->stash_pos[k]);
    }
    const int32_t dft_pos_max = std::max((int32_t) dr.ring.pos_max(), stash_max_pos);
    const bool bridge = pending_pos >= 0 && pending_pos + 1 == in->p0 && pending_pos > dft_pos_max;
    const int rows = n_stash + (bridge ? 1 : 0) + (n - 1);
    const bool merge = n <= 16 && in->merge_max > 0 && rows <= in->merge_max;
    if (!merge && rows > 0) {
        // the drafter's decode splits at n_ubatch_dft; a one-row piece is the fused single-token graph (not reproduced)
        for (int off = 0; off < rows; off += U) {
            if (std::min(U, rows - off) < 2) {
                return 4;
            }
        }
    }

    // ---- the target, ubatch by ubatch; each ubatch's g rows (the encoder at prompt width, the in-graph fold below it)
    if (c->g_cap < n) {
        if (c->g_all) CK(cudaFree(c->g_all));
        if (c->g_rows) CK(cudaFree(c->g_rows));
        c->g_cap = std::max(n, 2048);
        CK(cudaMalloc(&c->g_all, (size_t) E*c->g_cap*sizeof(float)));
        CK(cudaMalloc(&c->g_rows, (size_t) E*(c->g_cap + 64 + 1)*sizeof(float)));
    }
    e.n_past = in->p0;
    for (int off = 0; off < n; off += U) {
        const int u = std::min(U, n - off);
        const bool last = off + u == n;
        if (u > 16) {
            if (!e.prefill_ubatch(in->tokens + off, u, in->want_logits && last, off == 0 ? in->pending : 0) ||
                    !e.prefill_fold(u, c->g_all + (size_t) E*off)) {
                return -1;
            }
        } else {
            if (!e.prefill_small(in->tokens + off, u, in->want_logits && last, off == 0 ? in->pending : 0)) {
                return -1;
            }
            CK(cudaMemcpyAsync(c->g_all + (size_t) E*off, e.pf_g_small(), (size_t) E*u*sizeof(float), cudaMemcpyDeviceToDevice,
                               c->st));
        }
    }

    // ---- the drafter's rows: [stash ; the bridge ; (tokens[k + 1], g[k]) for k < n - 1]
    std::vector<int32_t> toks, poss;
    std::vector<float> host_g;   // the stash's and the bridge's g rows (host), ahead of the batch's own on the device
    for (int k = 0; k < n_stash; ++k) {
        toks.push_back(in->stash_ids[k]);
        poss.push_back(in->stash_pos[k]);
    }
    host_g.assign(in->stash_g, in->stash_g + (size_t) E*n_stash);
    if (bridge) {
        toks.push_back(in->tokens[0]);
        poss.push_back(pending_pos);
        host_g.insert(host_g.end(), in->pending_g, in->pending_g + E);
    }
    const int n_head_rows = (int) toks.size();
    for (int k = 0; k + 1 < n; ++k) {
        toks.push_back(in->tokens[k + 1]);
        poss.push_back(in->p0 + k);
    }
    if (n_head_rows > 0) {
        CK(cudaMemcpyAsync(c->g_rows, host_g.data(), host_g.size()*sizeof(float), cudaMemcpyHostToDevice, c->st));
    }
    if (n > 1) {
        CK(cudaMemcpyAsync(c->g_rows + (size_t) E*n_head_rows, c->g_all, (size_t) E*(n - 1)*sizeof(float),
                           cudaMemcpyDeviceToDevice, c->st));
    }
    out->decoded_rows = 0;
    out->n_stash = 0;
    if (merge) {   // the catch-up merge: the rows wait for the next seed decode
        out->n_stash = rows;
        for (int k = 0; k < rows; ++k) {
            out->stash_ids[k] = toks[k];
            out->stash_pos[k] = poss[k];
        }
        CK(cudaMemcpyAsync(out->stash_g, c->g_rows, (size_t) E*rows*sizeof(float), cudaMemcpyDeviceToHost, c->st));
    } else {
        for (int off = 0; off < rows; off += U) {   // llama_decode(ctx_dft)'s split
            const int u = std::min(U, rows - off);
            dr.prefill(c->st, toks.data() + off, poss.data() + off, c->g_rows + (size_t) E*off, u);
        }
        out->decoded_rows = rows;
    }

    // ---- what the server keeps: the deferred boundary (the batch's last row), the batch's g rows, the logits
    CK(cudaMemcpyAsync(out->g_last, c->g_all + (size_t) E*(n - 1), (size_t) E*sizeof(float), cudaMemcpyDeviceToHost, c->st));
    CK(cudaMemcpyAsync(out->verify_g, c->g_all, (size_t) E*n*sizeof(float), cudaMemcpyDeviceToHost, c->st));
    if (in->want_logits) {
        CK(cudaMemcpyAsync(out->logits, e.logits, (size_t) c->tgt.hp.n_vocab*sizeof(float), cudaMemcpyDeviceToHost, c->st));
    }
    CK(cudaStreamSynchronize(c->st));
    return 0;
}

XE_API int xe_ring(xe_ctx * c, int32_t * pos, int32_t n, int32_t * head) {
    const Ring & r = c->dr->ring;
    for (int32_t i = 0; i < n && i < (int32_t) r.size; ++i) {
        pos[i] = r.pos[i];
    }
    if (head) {
        *head = (int32_t) r.head;
    }
    return (int) r.used_max_p1();
}

} // extern "C"
