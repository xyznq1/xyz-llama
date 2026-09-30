// xyz-engine: the target forward. The launch order and every kernel instance match the server's verify pass; the graph
// semantics implement the fork's qwen35 builder (src/models/qwen35.cpp,
// delta-net-base.cpp rs_replay, llama-graph.cpp build_hadamard_rotate / build_attn) as its CUDA fusions execute it.
#include "engine.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// ---- the engine's own kernels: exact operations only (a sign flip, an add, the 0/-inf mask) ------------------------------

// the embedding's inverse rotation, second half: x = fwht(row) * signs (llama-graph build_embd_rotation mode 1; the
// fork's k_bin_bcast op_mul computes the same single product)
static __global__ void k_mul_signs(const float * __restrict__ x, const float * __restrict__ s, float * __restrict__ y,
                                   const int n, const int total) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = x[i] * s[i % n];
    }
}

// one half of the fold's concat rows (ggml_concat dim 0 of [5120, T] and [5120, T]: a copy)
static __global__ void k_fold_half(const float * __restrict__ src, float * __restrict__ cat, const int half, const int total) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        cat[(i / 5120)*10240 + half*5120 + i % 5120] = src[i];
    }
}

// tokens[1 + s] = the draw of record s (GGML_DRAFT_SAMPLE_OUT int32 per record, [0] = the selected id)
static __global__ void k_tokens_from_rec(PassIn * in, const int32_t * __restrict__ rec, const int n, const int rec_ints) {
    const int s = threadIdx.x;
    if (s < n) {
        in->tokens[1 + s] = rec[s*rec_ints];
    }
}

static __global__ void k_add(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ y, const int total) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = a[i] + b[i];
    }
}

// the device KQ mask (llama_kv_cache::build_kq_mask_dev): cell j is visible to row r iff its position < pos_r + 0.5 --
// one sequence in compact cells, so cell j holds position j and cells past the batch are empty (masked). Only from the
// last 64-aligned tile start at or below pos_0 + 1: before it every row sees every cell, and flash_attn (vis_pos) reads
// no mask there -- except the vector kernel (full: eng::fa_vec_rule), which reads every cell.
static __global__ void k_kq_mask(half * __restrict__ mask, const PassIn * __restrict__ in, const int n_tok, const bool full) {
    const int n_kv = in->n_kv;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv || (!full && j < (in->pos[0] + 1) / 64 * 64)) {
        return;
    }
    for (int r = 0; r < n_tok; ++r) {
        mask[(int64_t) r*n_kv + j] = j <= in->pos[r] ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// ---- setup ------------------------------------------------------------------------------------------------------------

static void * dev_alloc(size_t bytes) {
    void * p = nullptr;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemset(p, 0, bytes));
    return p;
}

// A merged sibling matmul uses [A; B; C] when the members are contiguous; otherwise the layer launches them separately.
static const void * merged(const Model & m, const std::vector<std::string> & names) {
    const Weight & a = m.get(names[0]);
    const char * end = (const char *) a.data + a.nbytes;
    for (size_t i = 1; i < names.size(); ++i) {
        const Weight & b = m.get(names[i]);
        if ((const char *) b.data != end) {
            return nullptr;
        }
        end = (const char *) b.data + b.nbytes;
    }
    return a.data;
}

bool Engine::init(cudaStream_t stream) {
    st = stream;
    for (int i = 0; i < 2; ++i) {
        CK(cudaStreamCreateWithFlags(&side[i], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_join[i], cudaEventDisableTiming));
    }
    CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
    kv_size = ext->kv_size;
    const Hparams & hp = m.hp;
    eps = hp.rms_eps;
    if (hp.n_embd != 5120 || hp.n_ff != 17408 || hp.n_head != 24 || hp.n_head_kv != 4 || hp.head_dim != 256 ||
            hp.ssm_inner != 6144 || hp.ssm_state != 128 || hp.ssm_groups != 16 || hp.ssm_dt_rank != 48 || hp.ssm_conv != 4 ||
            hp.had_block != 1024) {
        fprintf(stderr, "engine: not the Qwen3.8-27B qwen35 shape the kernels are instantiated for\n");
        return false;
    }
    for (size_t i = 0; i < hp.had_widths.size(); ++i) {
        (hp.had_widths[i] == 5120 ? s5120 : hp.had_widths[i] == 6144 ? s6144 : s17408) = m.had_signs_dev[i];
    }
    if (!s5120 || !s6144 || !s17408) {
        fprintf(stderr, "engine: missing a Hadamard sign vector\n");
        return false;
    }
    eng::fork_runtime_init(st);
    eng::fa_init();
    const int sections[4] = { hp.rope_sections[0], hp.rope_sections[1], hp.rope_sections[2], hp.rope_sections[3] };
    // llama-context cparams for this model: no YaRN (ext 0), attn_factor 1, freq_scale 1, beta 32/1; n_ctx_orig only feeds
    // the YaRN corr dims, which ext_factor 0 never reads
    rope = eng::mrope_params(hp.rope_dims, sections, GGML_ROPE_TYPE_IMROPE, 262144, hp.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    tok_embd = m.get("token_embd.weight").data;
    tok_ilv  = m.get("token_embd.weight").ilv;
    w_out    = m.get("output.weight").data;
    out_norm = (const float *) m.get("output_norm.weight").data;

    const int64_t pack_row = 2*16*128 + 48*128 + 48 + 48;   // ggml_gated_delta_net_pack_row(128, 16, 128, 48, false)
    L.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; ++il) {
        Layer & l = L[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        l.attn      = hp.is_attn(il);
        l.attn_norm = (const float *) m.get(p + "attn_norm.weight").data;
        l.post_norm = (const float *) m.get(p + "post_attention_norm.weight").data;
        l.w_gate_up = merged(m, { p + "ffn_gate.weight", p + "ffn_up.weight" });
        l.w_gate    = m.get(p + "ffn_gate.weight").data;
        l.w_up      = m.get(p + "ffn_up.weight").data;
        l.w_down    = m.get(p + "ffn_down.weight").data;
        if (l.attn) {
            l.w_qkv  = merged(m, { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight" });
            l.w_q    = m.get(p + "attn_q.weight").data;
            l.w_k    = m.get(p + "attn_k.weight").data;
            l.w_v    = m.get(p + "attn_v.weight").data;
            l.q_norm = (const float *) m.get(p + "attn_q_norm.weight").data;
            l.k_norm = (const float *) m.get(p + "attn_k_norm.weight").data;
            l.w_o    = m.get(p + "attn_output.weight").data;
            l.k_cache = (char *) ext->k[il];
            l.v_cache = (char *) ext->v[il];
            if (l.k_cache == nullptr || l.v_cache == nullptr) {
                fprintf(stderr, "engine: bind: layer %d has no K/V\n", il);
                return false;
            }
        } else {
            l.w_qkvz    = merged(m, { p + "attn_qkv.weight", p + "attn_gate.weight" });
            l.w_qkv_a   = m.get(p + "attn_qkv.weight").data;
            l.w_z       = m.get(p + "attn_gate.weight").data;
            l.w_alpha   = m.get(p + "ssm_alpha.weight").data;
            l.w_beta    = m.get(p + "ssm_beta.weight").data;
            l.dt        = (const float *) m.get(p + "ssm_dt.bias").data;
            l.a         = (const float *) m.get(p + "ssm_a").data;
            l.conv_w    = (const float *) m.get(p + "ssm_conv1d.weight").data;
            l.ssm_norm  = (const float *) m.get(p + "ssm_norm.weight").data;
            l.w_ssm_out = m.get(p + "ssm_out.weight").data;
            l.conv_state  = ext->conv[il];
            l.conv_pack   = ext->px[il];
            l.ssm_state   = ext->ssm[il];
            l.gdn_pack[0] = ext->pk[il];
            l.gdn_pack[1] = (float *) dev_alloc((size_t) pack_row*P*sizeof(float));
            if (!l.conv_state || !l.conv_pack || !l.ssm_state || !l.gdn_pack[0]) {
                fprintf(stderr, "engine: bind: layer %d has no recurrent rows\n", il);
                return false;
            }
        }
    }

    const auto f = [&](size_t n) { return (float *) dev_alloc(n*T*sizeof(float)); };
    emb = f(5120); x = f(5120); xr = f(5120); norm_out = f(5120); rot = f(5120); mm_out = f(5120);
    qkv_mixed = f(10240); z = f(6144); gate = f(48); beta = f(48); conv_silu = f(10240); qk_norm = f(4096);
    gdn_out = f(6144); rot6k = f(6144); gate_up = f(2*17408); glu_rot = f(17408);
    q_full = f(12288); k_cur = f(1024); v_cur = f(1024); q_rot = f(6144); fa_out = f(6144);
    xf = f(5120); h = f(5120); h_rot = f(5120); logits = f((size_t) hp.n_vocab);
    ones = (float *) dev_alloc(128*sizeof(float));
    {
        float o[128];
        for (float & v : o) v = 1.0f;
        CK(cudaMemcpy(ones, o, sizeof(o), cudaMemcpyHostToDevice));   // the InnerQ scale_inv llama initialises to 1
    }
    mask = (half *) dev_alloc((size_t) kv_size*T*sizeof(half));
    q8   = (char *) dev_alloc(1u << 20);
    in_dev = (PassIn *) dev_alloc(sizeof(PassIn));
    CK(cudaMallocHost(&in_host, sizeof(PassIn)));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    fprintf(stderr, "engine: %d layers, T %d, %d KV cells, %.2f GiB free after init\n", hp.n_layer, T, kv_size, fr / 1073741824.0);
    return true;
}

// ---- one pass -----------------------------------------------------------------------------------------------------------

bool Engine::set_fold(const void * w, int type, int layer) {
    if (w == nullptr || layer < 1 || layer >= m.hp.n_layer) {
        return false;
    }
    CK(cudaMalloc(&fold_cat, (size_t) 10240*T*sizeof(float)));
    CK(cudaMalloc(&g_rows, (size_t) 5120*T*sizeof(float)));
    CK(cudaMalloc(&q8f, (size_t) 1 << 18));
    fc_w = w; fc_type = type; fc_layer = layer;
    for (auto & kv : graphs) CK(cudaGraphExecDestroy(kv.second));   // captured passes do not have it
    graphs.clear();
    graph_used.clear();
    return true;
}

void Engine::pass(const int32_t * tokens, int pending, int width) {
    const Hparams & hp = m.hp;
    const int w = width > 0 ? width : T;
    if (w > T) {
        fprintf(stderr, "engine: a pass of %d columns (buffers hold %d)\n", w, T);
        exit(1);
    }
    Tw = w;
    const int64_t p0 = n_past;
    const int n_kv = (int) std::min<int64_t>(kv_size, std::max<int64_t>(256, (p0 + w + 255) / 256 * 256));
    if (p0 + w > kv_size) {
        fprintf(stderr, "engine: context full (%lld + %d > %d)\n", (long long) p0, w, kv_size);
        exit(1);
    }
    PassIn & in = *in_host;
    memset(&in, 0, sizeof(in));
    for (int j = 0; j < w; ++j) {   // M-RoPE sections at stride w: the chains read pos[s*n_tokens + t]
        in.tokens[j]       = tokens[j];
        in.pos[j]          = (int32_t) (p0 + j);
        in.pos[w + j]      = (int32_t) (p0 + j);
        in.pos[2*w + j]    = (int32_t) (p0 + j);
        in.pos[3*w + j]    = 0;
        in.kv_idx[j]       = p0 + j;
    }
    for (int w = 0; w < 3; ++w) {
        in.conv_idx[w] = pending + w;
    }
    in.s_row[0]    = 0;
    in.prefix_n[0] = pending;
    in.pos0        = (int32_t) p0;
    in.n_kv        = n_kv;
    CK(cudaMemcpyAsync(in_dev, in_host, sizeof(PassIn), cudaMemcpyHostToDevice, st));
    if (dev_draft_rec != nullptr && dev_draft_n > 0) {
        k_tokens_from_rec<<<1, 32, 0, st>>>(in_dev, dev_draft_rec, dev_draft_n, 4 + 3*32);   // GGML_DRAFT_SAMPLE_OUT
    }
    const int par = pass_no & 1;
    const int64_t key = ((int64_t) n_kv*8 + w)*2 + par;
    auto it = graphs.find(key);
    if (it == graphs.end()) {
        if (graphs.size() >= 12) {   // evict the least recently launched (engine.h graph_used)
            auto lru = graph_used.begin();
            for (auto u = graph_used.begin(); u != graph_used.end(); ++u) {
                if (u->second < lru->second) {
                    lru = u;
                }
            }
            CK(cudaStreamSynchronize(st));   // its last launch may still run
            CK(cudaGraphExecDestroy(graphs.at(lru->first)));
            graphs.erase(lru->first);
            graph_used.erase(lru);
        }
        cudaGraph_t g = nullptr;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
        issue(n_kv, par);
        CK(cudaStreamEndCapture(st, &g));
        cudaGraphExec_t ge = nullptr;
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphDestroy(g));
        it = graphs.emplace(key, ge).first;
        n_captures++;
    }
    graph_used[key] = ++graph_clock;
    CK(cudaGraphLaunch(it->second, st));
    n_past += w;
    pass_no++;
}

void Engine::issue(const int n_kv, const int par) {
    const Hparams & hp = m.hp;

    // embedding: rows of the rotated table, butterfly, then signs
    eng::get_rows_ptq1(st, tok_embd, 5120, hp.n_vocab, in_dev->tokens, Tw, emb, tok_ilv);
    eng::fwht_block(st, emb, xf, (int64_t) 5*Tw, nullptr, 1, nullptr);
    k_mul_signs<<<(5120*Tw + 255)/256, 256, 0, st>>>(xf, s5120, x, 5120, 5120*Tw);
    k_kq_mask<<<(n_kv + 255)/256, 256, 0, st>>>(mask, in_dev, Tw, eng::fa_vec_rule(Tw, n_kv));

    for (int il = 0; il < hp.n_layer; ++il) {
        const Layer & l = L[il];
        // [ffn residual add ->] attn norm -> signs -> Hadamard (+ the un-rotated norm for the GDN's alpha/beta)
        if (il == 0) {
            eng::norm_fwht(st, x, nullptr, nullptr, l.attn_norm, s5120, l.attn ? nullptr : norm_out, rot, q8, Tw, eps);
        } else {
            eng::norm_fwht(st, mm_out, xr, x, l.attn_norm, s5120, l.attn ? nullptr : norm_out, rot, q8, Tw, eps);
        }
        if (il == fc_layer && fc_w != nullptr) {   // the layer's input: the fold's first half
            k_fold_half<<<(5120*Tw + 255)/256, 256, 0, st>>>(x, fold_cat, 0, 5120*Tw);
        }
        if (!l.attn) {
            // The alpha/beta prologue runs on a side stream beside the qkvz matmul; the recurrence waits for it.
            CK(cudaEventRecord(ev_fork, st));
            CK(cudaStreamWaitEvent(side[0], ev_fork, 0));
            eng::gdn_alpha_beta(side[0], l.w_alpha, l.w_beta, norm_out, l.dt, l.a, gate, beta);
            CK(cudaEventRecord(ev_join[0], side[0]));
            if (l.w_qkvz != nullptr) {
                eng::ptq1_matmul(st, l.w_qkvz, q8, 5120, 16384, Tw, qkv_mixed, 10240, z, 10240, 6144);
            } else {
                eng::ptq1_matmul(st, l.w_qkv_a, q8, 5120, 10240, Tw, qkv_mixed, 10240);
                eng::ptq1_matmul(st, l.w_z, q8, 5120, 6144, Tw, z, 6144);
            }
            eng::ConvReplay c = {};
            c.state = (const char *) l.conv_state;         c.state_row_stride = 3*10240*sizeof(float);
            c.pack  = (const char *) l.conv_pack;          c.pack_row_stride  = (size_t) P*10240*sizeof(float);
            c.s_row = in_dev->s_row;                       c.conv_idx         = in_dev->conv_idx;
            c.qkv   = qkv_mixed;                           c.qkv_nb1          = 10240*sizeof(float);
            c.state_dst = l.conv_state;                    c.pack_dst         = l.conv_pack;
            c.pack_dst_nb1 = 10240*sizeof(float);
            c.conv_weight = l.conv_w;                      c.conv_weight_nb1  = 4*sizeof(float);
            c.conv_silu = conv_silu;                       c.qk_norm          = qk_norm;
            c.eps = eps;
            eng::gdn_conv_replay(st, c, Tw);
            eng::GdnArgs g = {};
            g.q = qk_norm;               g.k = qk_norm + 16*128;
            g.sq1 = 128;                 g.sq2 = 32*128;       g.sq3 = (int64_t) 32*128*Tw;
            g.v = conv_silu + 4096;      g.sv1 = 128;          g.sv2 = 10240;       g.sv3 = (int64_t) 10240*Tw;
            g.g = gate;                  g.beta = beta;
            g.s = l.ssm_state;           g.state_out = l.ssm_state;
            g.dst = gdn_out;
            g.prefix = l.gdn_pack[par];  g.prefix_n = in_dev->prefix_n;
            g.prefix_sstride = (int64_t) (2*16*128 + 48*128 + 96) * P;
            g.pack_out = l.gdn_pack[par ^ 1];
            g.n_tokens = Tw;
            CK(cudaStreamWaitEvent(st, ev_join[0], 0));
            eng::gdn_own(st, g);   // the engine's bit-identical recurrence
            eng::gdn_out_chain(st, gdn_out, l.ssm_norm, z, s6144, rot6k, eps, q8, Tw);
            eng::ptq1_matmul(st, l.w_ssm_out, q8, 6144, 5120, Tw, mm_out, 5120);
        } else {
            if (l.w_qkv != nullptr) {
                eng::ptq1_matmul(st, l.w_qkv, q8, 5120, 14336, Tw, q_full, 12288, k_cur, 12288, 1024, v_cur, 13312, 1024);
            } else {
                eng::ptq1_matmul(st, l.w_q, q8, 5120, 12288, Tw, q_full, 12288);
                eng::ptq1_matmul(st, l.w_k, q8, 5120, 1024, Tw, k_cur, 1024);
                eng::ptq1_matmul(st, l.w_v, q8, 5120, 1024, Tw, v_cur, 1024);
            }
            // V and K writes run on side streams beside the Q chain; attention waits for both.
            CK(cudaEventRecord(ev_fork, st));
            CK(cudaStreamWaitEvent(side[0], ev_fork, 0));
            CK(cudaStreamWaitEvent(side[1], ev_fork, 0));
            eng::attn_v_write(side[0], v_cur, 1024, in_dev->kv_idx, l.v_cache, 1024/128*34, 4, Tw);
            CK(cudaEventRecord(ev_join[0], side[0]));
            eng::attn_k_write(side[1], k_cur, 256, 1024, l.k_norm, eps, in_dev->pos, Tw, rope, in_dev->kv_idx, l.k_cache,
                              1024/128*34, 4);
            CK(cudaEventRecord(ev_join[1], side[1]));
            eng::attn_q_chain(st, q_full, 512, 12288, l.q_norm, eps, in_dev->pos, Tw, rope, ones, q_rot, 24);
            CK(cudaStreamWaitEvent(st, ev_join[0], 0));
            CK(cudaStreamWaitEvent(st, ev_join[1], 0));
            eng::flash_attn(st, q_rot, Tw, 24, l.k_cache, l.v_cache, n_kv, kv_size, mask, 1.0f/16.0f, fa_out, in_dev->pos);
            eng::attn_out_chain(st, fa_out, q_full + 256, 512, 12288, ones, s6144, rot6k, q8, Tw);
            eng::ptq1_matmul(st, l.w_o, q8, 6144, 5120, Tw, mm_out, 5120);
        }
        // attn residual -> post-attention norm -> signs -> Hadamard; FFN
        eng::norm_fwht(st, mm_out, x, xr, l.post_norm, s5120, nullptr, rot, q8, Tw, eps);
        if (l.w_gate_up != nullptr) {
            eng::ptq1_matmul(st, l.w_gate_up, q8, 5120, 2*17408, Tw, gate_up, 2*17408);
        } else {
            eng::ptq1_matmul(st, l.w_gate, q8, 5120, 17408, Tw, gate_up, 2*17408);
            eng::ptq1_matmul(st, l.w_up, q8, 5120, 17408, Tw, gate_up + 17408, 2*17408);
        }
        eng::fwht_glu(st, gate_up, glu_rot, (int64_t) 17*Tw, s17408, 17, q8, 17408);
        eng::ptq1_matmul(st, l.w_down, q8, 17408, 5120, Tw, mm_out, 5120);
    }
    // the last residual, the output norm, the LM head
    k_add<<<(5120*Tw + 255)/256, 256, 0, st>>>(mm_out, xr, xf, 5120*Tw);
    eng::rms_norm_mul(st, xf, out_norm, h, 5120, Tw, eps);
    if (fc_w != nullptr) {   // h_nextn: the fold's second half, then the head's fc (unfused mul_mat_vec_q<Q3_K, Tw>)
        k_fold_half<<<(5120*Tw + 255)/256, 256, 0, st>>>(h, fold_cat, 1, 5120*Tw);
        eng::mmvq_n(st, fc_type, fc_w, fold_cat, 10240, 5120, Tw, g_rows, q8f);
    }
    eng::fwht_block(st, h, h_rot, (int64_t) 5*Tw, s5120, 5, q8);
    eng::ptq1_matmul(st, w_out, q8, 5120, hp.n_vocab, Tw, logits, hp.n_vocab);
    CK(cudaGetLastError());
}
