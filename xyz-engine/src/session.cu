// xyz-engine session: see session.h
#include "session.h"

#include <cstddef>
#include <cstdio>
#include <algorithm>

namespace {

uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint64_t coupled_key(uint32_t seed, int32_t seq, int32_t pos) {   // llama_sampler_coupled_key
    return splitmix(((uint64_t) seed << 32) ^ splitmix(((uint64_t) (uint32_t) seq << 32) ^ (uint64_t) (uint32_t) pos));
}

// the accept's draft inputs from the chain's records: ids and logit bits of the top_k slots, the draws
__global__ void k_rec_to_accept_s(const int32_t * __restrict__ rec, const int n, const int K, int32_t * __restrict__ did,
                                  float * __restrict__ dval, int32_t * __restrict__ draft) {
    const int s = blockIdx.x, k = threadIdx.x;
    if (s >= n || k >= K) return;
    const int32_t * r = rec + s*GGML_DRAFT_SAMPLE_OUT;
    did[s*K + k]  = r[4 + k];
    dval[s*K + k] = __int_as_float(r[4 + GGML_DRAFT_SAMPLE_MAX_K + k]);
    if (k == 0) draft[s] = r[0];
}

} // namespace

bool Session::init(const char * expf_path) {
    FILE * fx = fopen(expf_path, "rb");
    uint32_t cnt = 0;
    if (!fx || fread(&cnt, 4, 1, fx) != 1) {
        fprintf(stderr, "session: cannot read %s\n", expf_path);
        if (fx) fclose(fx);
        return false;
    }
    std::vector<uint32_t> ex((size_t) 2*cnt);
    const bool ok = fread(ex.data(), 4, ex.size(), fx) == ex.size();
    fclose(fx);
    if (!ok) return false;
    n_exc = (int) cnt;
    CK(cudaMalloc(&d_exc, ex.size()*4));
    CK(cudaMemcpy(d_exc, ex.data(), ex.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_rng, sizeof(xe::Mt19937)));
    CK(cudaMalloc(&d_cut, 17*sizeof(int32_t)));
    CK(cudaMemset(d_cut, 0, 17*sizeof(int32_t)));
    CK(cudaMalloc(&d_tid, 8*K*4)); CK(cudaMalloc(&d_tval, 8*K*4)); CK(cudaMalloc(&d_nok, 64));
    CK(cudaMalloc(&d_did, 8*K*4)); CK(cudaMalloc(&d_dval, 8*K*4)); CK(cudaMalloc(&d_draft, 64));
    CK(cudaMalloc(&d_tks, xe::topk_rows_scratch(e.m.hp.n_vocab, e.T)));
    CK(cudaMalloc(&d_out, sizeof(xe::AcceptOut)));
    CK(cudaMallocHost(&h_out, sizeof(xe::AcceptOut)));
    CK(cudaMallocHost(&h_nok, 8*sizeof(int32_t)));
    CK(cudaMallocHost(&h_draft, 8*sizeof(int32_t)));
    CK(cudaMalloc(&d_seed_g, (size_t) Drafter::MAXR*5120*sizeof(float)));
    return true;
}

void Session::set_topk_skip(const int32_t * ids, int n) {
    std::vector<int32_t> v(ids, ids + n);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    if (topk_skip != nullptr) {
        CK(cudaFree(topk_skip));
        topk_skip = nullptr;
    }
    n_topk_skip = (int) v.size();
    if (n_topk_skip > 0) {
        CK(cudaMalloc(&topk_skip, v.size()*sizeof(int32_t)));
        CK(cudaMemcpy(topk_skip, v.data(), v.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    }
}

void Session::logits_row(int r, float * dst) {
    CK(cudaMemcpy(dst, e.logits + (size_t) r*e.m.hp.n_vocab, (size_t) e.m.hp.n_vocab*sizeof(float), cudaMemcpyDeviceToHost));
}

// the draft length block verification reads (the accept kernel): the index of the first trigger id, else G
static int cut_len(const SessionStart & s, const int32_t * drafts, const int G) {
    for (int j = 0; j < G; ++j) {
        for (int e = 0; e < s.n_cut && e < 16; ++e) {
            if (drafts[j] == s.cut[e]) {
                return j;
            }
        }
    }
    return G;
}

void Session::issue_accept(const int G, const int mode) {
    cudaStream_t st = e.st;
    xe::topk_rows(st, e.logits, e.m.hp.n_vocab, G + 1, K, topk_skip, n_topk_skip, d_tid, d_tval, d_nok, d_tks);
    k_rec_to_accept_s<<<G, 32, 0, st>>>(dr.rec, G, K, d_did, d_dval, d_draft);
    const xe::AcceptParams prm = { G, K, 0.95f, 0.95f, 0, 0, mode, d_cut };
    xe::accept(st, prm, d_tid, d_tval, d_did, d_dval, d_draft, d_rng, d_exc, n_exc, d_out, dr.keys);
    // the next seed's g rows: the verify's fold rows 0..G (the host path keeps them there too)
    CK(cudaMemcpyAsync(d_seed_g, e.g_rows, (size_t) (G + 1)*5120*sizeof(float), cudaMemcpyDeviceToDevice, st));
}

bool Session::run(const SessionStart & s, int max_tokens, const RoundFn & on_round, SessionEnd & out) {
    cudaStream_t st = e.st;
    const int n_draws = s.n_draws;
    if (n_draws + 1 > e.T || s.m < 1 || s.m > Drafter::MAXR) {
        fprintf(stderr, "session: %d draws (the verify holds %d), seed rows %d\n", n_draws, e.T, s.m);
        return false;
    }
    CK(cudaMemcpyAsync(d_rng, &s.rng, sizeof(s.rng), cudaMemcpyHostToDevice, st));
    {   // the waiting grammar's trigger ids: block verification reads each draft only up to the first
        int32_t hc[17] = {};
        hc[0] = s.accept_mode == xe::ACC_BLOCK && s.cut != nullptr ? std::min(s.n_cut, 16) : 0;
        for (int k = 0; k < hc[0]; ++k) hc[1 + k] = s.cut[k];
        CK(cudaMemcpyAsync(d_cut, hc, sizeof(hc), cudaMemcpyHostToDevice, st));   // pageable: staged before return
    }
    CK(cudaMemcpyAsync(d_seed_g, s.seed_g, (size_t) s.m*5120*sizeof(float), cudaMemcpyHostToDevice, st));
    int     m = s.m;
    int32_t seed_pos0 = s.seed_pos0, p0 = s.p0, id_last = s.id_last, pending = s.pending;
    std::vector<int32_t> seed_tok(s.seed_tok, s.seed_tok + m);
    dr.ring.seq_rm(seed_pos0, -1);   // the server's seq_rm before a seed (the ring holds the server's table)
    dr.n_steps = s.chain_steps > 0 ? s.chain_steps : n_draws;   // fixed for the whole response, as the server's chain
    e.dev_draft_rec = dr.rec;
    e.dev_draft_n   = n_draws;
    std::vector<int32_t> vt(e.T, 0);
    std::vector<uint64_t> keys(n_draws);
    out = SessionEnd();
    bool go = true;
    const bool plain = s.accept_mode == xe::ACC_PLAIN;
    int32_t toks[xe::ACC_MAX_G + 1];
    while (go && out.n_tokens < max_tokens) {
        // this round's draft length: the server's get_n_draft_max at the budget end (min(n_max, n_remaining - 1))
        const int G = s.budget_tail ? std::min(n_draws, max_tokens - out.n_tokens - 1) : n_draws;
        if (G <= 0) {
            break;
        }
        for (int k = 0; k < G; ++k) keys[k] = coupled_key(s.cseed, 0, p0 + 1 + k);
        // the bonus row's key rides in the draft's input block (uploaded with it): the plain verify's row G
        const uint64_t kb = coupled_key(s.cseed, 0, p0 + 1 + G);
        dr.keys_host[2*G]     = (uint32_t) (kb & 0xFFFFFFFFull);
        dr.keys_host[2*G + 1] = (uint32_t) (kb >> 32);
        round_g = G;      // (dr.n_steps stays the chain's length: a round of G < chain_steps draws has no pruned step)
        dr.draft_graph(st, seed_tok.data(), seed_pos0, d_seed_g, m, keys.data(), G);
        vt[0] = id_last;
        e.n_past = p0;
        e.dev_draft_n = G;
        e.pass(vt.data(), pending, G + 1);
        issue_accept(G, s.accept_mode);
        CK(cudaMemcpyAsync(h_out, d_out, offsetof(xe::AcceptOut, p_n), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(&h_out->one_mask, &d_out->one_mask, 2*sizeof(int32_t), cudaMemcpyDeviceToHost, st));   // + cut
        CK(cudaMemcpyAsync(h_nok, d_nok, (size_t) (G + 1)*sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_draft, d_draft, (size_t) G*sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int n = h_out->n;
        for (int k = 0; k < n; ++k) toks[k] = h_out->tokens[k];
        out.rounds++;
        if (!plain) {
            out.rng_draws += 2ull*(uint64_t) (cut_len(s, h_draft, G) + 1);   // Gv + 1 uniforms, two engine outputs each
        }
        uint32_t flags = 0;
        for (int r = 0; r < G + 1; ++r) {
            if ((h_nok[r] & ~xe::TK_TIE) != K) {   // fewer than K candidates: the host's scan would have fallen back
                out.short_rows++;
                flags |= RF_SHORT;
            }
            if (h_nok[r] > 0 && (h_nok[r] & xe::TK_TIE)) {
                out.tie_rows++;
                flags |= RF_TIE;
            }
        }
        if (h_out->cut >= 0) {
            out.cut_rounds++;
            flags |= RF_CUT;
        }
        go = on_round(toks, n, (uint32_t) h_out->one_mask, h_draft, flags);
        out.n_tokens += n;
        const int tau = n - 1;
        // the next round: seed rows = accepted drafts + the new token at p0..p0+tau (fold rows 0..tau, already in place)
        m = tau + 1;
        seed_pos0 = p0;
        seed_tok.assign(toks, toks + m);
        p0 += tau + 1;
        pending = tau + 1;
        id_last = toks[tau];
        dr.ring.seq_rm(seed_pos0, -1);   // the server drops the draft's cells before the next seed
    }
    out.p0 = p0; out.id_last = id_last; out.pending = pending; out.m = m; out.seed_pos0 = seed_pos0;
    for (int k = 0; k < m && k < 8; ++k) out.seed_tok[k] = seed_tok[k];
    return true;
}
