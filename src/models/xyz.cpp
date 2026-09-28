#include "models.h"

void llama_model_xyz::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("XYZ model requires 'extract_layers' in GGUF metadata");
    }
    // The fusion input COUNT is a measured axis, not a constant: two inputs [34, 64]
    // scored 2.541 against three's 2.547 and cut the feature dump by a third, and the trained fc weights h_nextn
    // 1.0047 against ~0.1 for the others. Everything downstream already sizes itself from the list (fc is
    // n_embd_inp_enc() = count x n_embd_tgt wide; the extraction and packing in common/speculative.cpp loop over
    // it), so only this guard held the upstream xyz value of exactly three.
    if (target_layer_ids.empty() || target_layer_ids.size() > 8) {
        throw std::runtime_error("XYZ requires 1..8 entries in 'extract_layers'");
    }
    {
        std::string ids;
        for (size_t i = 0; i < target_layer_ids.size(); ++i) {
            ids += (i ? ", " : "") + std::to_string(target_layer_ids[i]);
        }
        LLAMA_LOG_INFO("%s: XYZ extract_layers = [%s]\n", __func__, ids.c_str());
    }

    uint32_t n_embd_tgt = 0;

    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);
    LLAMA_LOG_INFO("%s: XYZ n_embd_tgt = %u (draft n_embd = %u)\n", __func__, n_embd_tgt, hparams.n_embd);

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    // xyz norm_before_residual (optional, default false)
    // compatible with Readhat xyz speculator model
    ml.get_key(LLM_KV_NORM_BEFORE_RESIDUAL, hparams.norm_before_residual, false);
    if (hparams.norm_before_residual) {
        LLAMA_LOG_INFO("%s: XYZgnorm_before_residual = true\n", __func__);
    }

    // xyz norm_before_fc (optional, default false)
    // compatible with xyz.1 (e.g. nvidia/gpt-oss-120b-Eagle3-v3)
    ml.get_key(LLM_KV_NORM_BEFORE_FC, hparams.norm_before_fc, false);

    // DRAFTER ATTENTION WINDOW (xyz.attention.sliding_window). The head is trained on short chunks but would attend
    // over its whole KV cache at serving time -- 151k positions it never saw in training, ~1.3 ms of a round, for no acceptance
    // gain past ~8k. A window makes each draft step read N cells instead of all of them and shrinks the drafter's cache.
    uint32_t n_swa_draft = 0;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, n_swa_draft, false);
    if (n_swa_draft > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        // set_swa_pattern(n) marks layer il as SWA when `n == 0 || (il % n < n - 1)`: n == 1 would mark nothing (and
        // trip is_swa_any() at load); 0 is the "every layer" value, and the drafter's single layer is the one to window
        hparams.set_swa_pattern(0);
        hparams.n_swa = n_swa_draft;
        LLAMA_LOG_INFO("%s: XYZ draft attention window = %u tokens\n", __func__, hparams.n_swa);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_xyz::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // xyz2 / MTP-fusion mode (train_xyz2.py): the head is the target's MTP block -- eh_proj fuses token embedding and
    // hidden into ONE n_embd vector that is both residual and attention input. Detected from the GGUF's tensors.
    const bool mtp_fusion = ml->get_tensor_meta(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", 0).str().c_str()) != nullptr;
    if (mtp_fusion) {
        LLAMA_LOG_INFO("%s: XYZ in MTP-fusion mode (xyz2 head: eh_proj + gated attention)\n", __func__);
    } else if (n_layer != 1) {
        throw std::runtime_error("multi-layer XYZ requires xyz2 MTP-fusion tensors in block 0");
    }

    // Get vocab size from the d2t tensor in the GGUF file (optional - only needed if xyz has different vocab_size than target)
    // d2t: draft to target vocabulary mapping
    int64_t n_draft_vocab = n_vocab;  // Default: same as target vocab
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0]; // update draft vocab size
        d2t = create_tensor(tn(LLM_TENSOR_D2T), {n_draft_vocab}, 0);
        LLAMA_LOG_INFO("%s: XYZ using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long)n_draft_vocab);
    } else {
        d2t = nullptr; // no d2t, use default vocab size
        LLAMA_LOG_INFO("%s: XYZ without d2t - sharing same vocab_size with target (vocab_size = %lld)\n", __func__, (long long)n_draft_vocab);
    }

    // Feature fusion layer: projects 3 target layers to draft hidden size
    fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), {n_embd_inp, n_embd}, 0);

    // RMSNorm on the fused target features (input to fc), only when norm_before_fc is set.
    if (hparams.norm_before_fc) {
        output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd_inp}, 0);
    }

    // xyz2 v2: a student narrower than the target (d 2048 against the target's 5120) keeps the target's own head, so
    // it projects back up to the target's width first. Optional: a same-width drafter (the heal) has no such tensor
    // and everything below stays at n_embd, exactly as before.
    const ggml_tensor * up_meta = ml->get_tensor_meta(tn(LLM_TENSOR_XYZ_UP, "weight").str().c_str());
    const int64_t n_embd_head_out = up_meta ? up_meta->ne[1] : n_embd;
    if (up_meta) {
        xyz_up = create_tensor(tn(LLM_TENSOR_XYZ_UP, "weight"), {n_embd, n_embd_head_out}, 0);
        LLAMA_LOG_INFO("%s: XYZ narrow student: body %lld -> head %lld (xyz_up)\n",
                       __func__, (long long) n_embd, (long long) n_embd_head_out);
    }

    // Output layer (uses draft vocab size)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd_head_out}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd_head_out, n_draft_vocab}, TENSOR_NOT_REQUIRED);

    // Token embeddings (optional - Llama 3.3 70B XYZ has its own)
    const struct ggml_tensor * tok_embd_meta = ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_target_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_target_vocab}, 0);
        LLAMA_LOG_INFO("%s: XYZ using its own token_embd (vocab = %lld)\n", __func__, (long long)n_target_vocab);
    }

    // Decoder layers
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const bool layer_mtp_fusion = mtp_fusion && i == 0;
        const int64_t n_embd_attn_input = mtp_fusion ? n_embd : 2 * n_embd;

        // input_layernorm: applied to token embeddings
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (layer_mtp_fusion) {
            // The token embedding is the TARGET's (borrowed, n_embd_tgt wide) while the hidden is the drafter's own,
            // so the fusion input is n_embd_tgt + n_embd -- equal to 2*n_embd only for a same-width drafter. Read the
            // width from enorm, which is sized over the embedding it normalises (xyz2 v2: 5120 with a 2048 body).
            const ggml_tensor * en_meta = ml->get_tensor_meta(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i).str().c_str());
            const int64_t n_embd_emb = en_meta ? en_meta->ne[0] : n_embd;
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {n_embd_emb + n_embd, n_embd}, 0);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", i), {n_embd_emb}, 0);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", i), {n_embd}, 0);
        }
        if (mtp_fusion) {
            layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,   "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
            layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,   "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        } else {
            // xyz specific: hidden_norm applied to fused target features
            layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
        }

        // Attention input: the eh_proj output (fusion) or input_embeds_normed + fused_target_normed (classic).
        // Gated attention (qwen35 style) carries [q | gate] per head in attn_q: twice the rows, read from the shape.
        const ggml_tensor * wq_meta = ml->get_tensor_meta(tn(LLM_TENSOR_ATTN_Q, "weight", i).str().c_str());
        const bool gated_attn = wq_meta != nullptr && wq_meta->ne[1] == 2 * n_embd_head_k * n_head;
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd_attn_input, (gated_attn ? 2 : 1) * n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd_attn_input, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd_attn_input, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // 3+1 hybrid: an optional pruned FFN for the chain's LAST draw -- the drafter's own top neurons, fine-tuned
        // (2048 of 17408 keep slot 4 at -0.49% tokens/round for 45% fewer bytes in that pass). Absent = every step as
        // before.
        const ggml_tensor * gl_meta = ml->get_tensor_meta(tn(LLM_TENSOR_FFN_GATE, "last.weight", i).str().c_str());
        if (gl_meta != nullptr) {
            const int64_t n_ff_last = gl_meta->ne[1];
            layer.ffn_norm_last = create_tensor(tn(LLM_TENSOR_FFN_NORM, "last.weight", i), {n_embd}, 0);
            layer.ffn_gate_last = create_tensor(tn(LLM_TENSOR_FFN_GATE, "last.weight", i), {n_embd, n_ff_last}, 0);
            layer.ffn_up_last   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "last.weight", i), {n_embd, n_ff_last}, 0);
            layer.ffn_down_last = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "last.weight", i), {n_ff_last, n_embd}, 0);
            LLAMA_LOG_INFO("%s: [XYZ-31] layer %d: the chain's last draw uses a pruned FFN (%lld of %lld neurons)\n",
                           __func__, i, (long long) n_ff_last, (long long) n_ff);
        }

        // rope_freqs for llama3 rope scaling (optional - only if xyz config has rope_scaling)
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_xyz::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_xyz::graph<true>::build_inp_embd_enc() const {
    ggml_tensor * cur = nullptr;

    // Input: Target model features (3 layers concatenated: low, mid, high)
    // Data will be provided via ubatch->embd in encode_xyz_features()
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// xyz Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in context
template <>
llama_model_xyz::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    cur = build_inp_embd_enc();

    // RMSNorm on the fused target features before fc
    if (hparams.norm_before_fc) {
        cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        cb(cur, "enc_input_norm", -1);
    }

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    // store in t_h_nextn (same as MTP) so can be read via llama_get_embeddings_nextn(ctx_dft)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// xyz Decoder: processes draft tokens using g_embeddings from encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
template <>
llama_model_xyz::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    ggml_tensor * cur;
    ggml_tensor * inpL;

    // xyz Decoder receives:
    // 1. Token embeddings (e.g.from xyz's own tok_embd for Llama 3.3 70B, or target model for Llama 3.1 8B)
    // 2. g_embeddings from encoder
    auto * tok_embd = model.tok_embd;
    if (model.tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "XYZ decoder requires token embeddings (own or from target model)");
        tok_embd = model_other->tok_embd;
    }

    // THE DEVICE DRAFT CHAIN: a chain STEP takes its token and its g from the
    // previous draw, on the device -- the drawn column's row of the draft vocabulary's embedding copy (then the same
    // rotation build_embd_rows applies) and the feedback hidden the previous step left in dchain->g. The batch's own
    // token and embd are placeholders then, and no host input carries them.
    const bool chain_step = dchain_mode == 2 && dchain != nullptr && n_tokens == 1;
    const bool chain_draw = dchain_mode != 0 && dchain != nullptr;

    ggml_tensor * inp_embd = nullptr;
    ggml_tensor * inp_g    = nullptr;
    if (chain_step) {
        inp_embd = build_embd_rotation(ggml_get_rows(ctx0, dchain->embd, dchain->col), dchain->embd_ref);
        inp_g    = ggml_reshape_2d(ctx0, dchain->g, n_embd, 1);
    } else {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tokens);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        inp_embd = build_embd_rows(tok_embd, inp->tokens);
        inp_g    = inp->embd;

        res->add_input(std::move(inp));
    }
    cb(inp_embd, "inp_embd", -1);
    cb(inp_g, "inp_g_embeddings", -1);

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // DRAFTER ATTENTION WINDOW, graph side. When the window is set (GGUF key or XYZ2_DRAFT_SWA above),
    // hparams.swa_type is STANDARD and the memory is a llama_kv_cache_iswa -- and build_attn_inp_kv() asserts
    // swa_type == NONE ("Use llama_kv_cache_iswa for SWA", llama-graph.cpp:3288). That assert is why BOTH previous
    // window attempts died at load without ever producing a number; the env was reaching the server all along.
    // The two build_attn overloads are identical apart from the input type, so one branch here and a dispatching
    // lambda below leave the windowed and unwindowed graphs the same in every other respect.
    const bool draft_swa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    llm_graph_input_attn_kv      * inp_attn      = draft_swa ? nullptr : build_attn_inp_kv();
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = draft_swa ? build_attn_inp_kv_iswa() : nullptr;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    const bool mtp_fusion = model.layers[0].nextn.eh_proj != nullptr;
    for (int il = 0; il < n_layer; ++il) {
        auto build_attn_draft = [&](ggml_tensor * wo, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
            return draft_swa
                ? build_attn(inp_attn_iswa, wo, NULL, nullptr, q, k, v, nullptr, nullptr, nullptr, kq_scale, il)
                : build_attn(inp_attn,      wo, NULL, nullptr, q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
        };
        const bool gated_attn = model.layers[il].wq->ne[1] == 2 * n_embd_head * n_head;

        ggml_tensor * inpSA;
        if (mtp_fusion && il == 0) {
            // xyz2 head: the target's MTP block. eh_proj([enorm(emb) ; hnorm(g)]) is the residual; attn_norm of it
            // is the attention input (src/models/qwen35.cpp build_mtp, same order: embedding first).
            ggml_tensor * e_norm = build_norm(inp_embd, model.layers[il].nextn.enorm, NULL, LLM_NORM_RMS, il);
            cb(e_norm, "xyz2_enorm", il);
            ggml_tensor * h_norm = build_norm(inp_g, model.layers[il].nextn.hnorm, NULL, LLM_NORM_RMS, il);
            cb(h_norm, "xyz2_hnorm", il);
            ggml_tensor * fused = build_lora_mm(model.layers[il].nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
            cb(fused, "xyz2_eh_proj", il);
            inpSA = fused;
            cur = build_norm(fused, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "xyz2_attn_norm", il);
        } else if (mtp_fusion) {
            inpSA = inpL;
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "xyz2_attn_norm", il);
        } else {
            // Apply input_layernorm to the token embeddings
            ggml_tensor * embd_norm = build_norm(inp_embd,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(embd_norm, "embd_norm", il);

            // Apply hidden_norm to inp_g
            ggml_tensor * g_norm = build_norm(inp_g,
                    model.layers[il].attn_norm_2, NULL,
                    LLM_NORM_RMS, -1);
            cb(g_norm, "g_norm", il);

            // norm_before_residual: determines what goes into the residual connection (compatible with Readhat xyz speculator model)
            // - false (default): use raw inp_g for residual
            // - true: use normalized g_norm for residual
            // inpL is the concatenated input (normalized inp_embd + normalized inp_g)
            inpSA = hparams.norm_before_residual ? g_norm : inpL;

            // Concatenate normalized inp_embd and normalized inp_g
            cur = ggml_concat(ctx0, embd_norm, g_norm, il);
            cb(cur, "concat_embd", il);
        }

        // Self-attention
        ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur_full, "Qcur_full", il);

        ggml_tensor * Qcur = nullptr;
        ggml_tensor * gate = nullptr;
        if (gated_attn) {
            // [q | gate] per head: q at offset 0, gate at offset n_embd_head, head stride 2 * n_embd_head
            const size_t esz = ggml_element_size(Qcur_full);
            Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
                    esz * n_embd_head * 2, esz * n_embd_head * 2 * n_head, 0);
            gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
                    esz * n_embd_head * 2, esz * n_embd_head * 2 * n_head, esz * n_embd_head);
            gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
            cb(gate, "xyz2_gate", il);
            if (model.layers[il].attn_q_norm == nullptr) {
                Qcur = ggml_cont(ctx0, Qcur);
            }
        } else {
            Qcur = ggml_reshape_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens);
        }
        if (model.layers[il].attn_q_norm) {
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
        }

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        if (model.layers[il].attn_k_norm) {
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);
        }

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur, "Vcur", il);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE. Fusion mode reproduces the target's MROPE on text: partial NEOX rotary over n_rot dims.
        const int rope_type_l = mtp_fusion ? (int) LLAMA_ROPE_TYPE_NEOX : rope_type;
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type_l, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type_l, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        if (gated_attn) {
            cur = build_attn_draft(nullptr, Qcur, Kcur, Vcur);
            cb(cur, "attn_pregate", il);
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_gated_out", il);
        } else {
            cur = build_attn_draft(model.layers[il].wo, Qcur, Kcur, Vcur);
        }

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // the chain's last draw takes the pruned FFN when the head carries one (3+1 hybrid); every other decode the full one
        const auto & L = model.layers[il];
        const bool ffn_last = dchain_last && chain_step && L.ffn_up_last != nullptr;

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                ffn_last ? L.ffn_norm_last : L.ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                ffn_last ? L.ffn_up_last   : L.ffn_up,   NULL, NULL,
                ffn_last ? L.ffn_gate_last : L.ffn_gate, NULL, NULL,
                ffn_last ? L.ffn_down_last : L.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "xyz_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    // Keep only the rows the caller asked for output on, the way every other decoder graph does
    // (llm_graph_context::build_inp_out_ids + ggml_get_rows, e.g. qwen35.cpp:154/219).
    //
    // Without the selection:
    //   * the head and the d2t scatter would run for EVERY row of the batch, output or not (a catch-up decode with
    //     logits=false on all rows would still pay the 32,768 x 5,120 projection), and
    //   * llama_context copies the FIRST n_outputs rows of t_logits / t_h_nextn, so a batch whose output row is
    //     not row 0 would return the wrong row: a 5-row draft seed pass would sample row 0 (an already-accepted
    //     catch-up position), and tokens/round falls 2.53 -> 2.01.
    // A drafter decode with exactly one output row is the only case where the two agree.
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    ggml_tensor * prenorm = cur;

    // xyz2 v2: the narrow student's hidden goes up to the target's width for the borrowed head. The FEEDBACK stays
    // in the student's own d-space (V2Head.feedback(h) = h; V2Block's hnorm normalises it on the way in), so it is
    // taken before this projection.
    if (model.xyz_up != nullptr) {
        cur = build_lora_mm(model.xyz_up, cur);
        cb(cur, "xyz_up", -1);
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // The hidden fed back as the next draft step's g (read through llama_get_embeddings_nextn):
    //  - classic xyz head: the pre-norm residual (its training convention)
    //  - MTP-fusion head (xyz2): shared_head_norm(h), what the target's own draft-mtp path feeds its MTP block at
    //    depth. Measured on real records: step-2 acc@1 0.760 pre-norm vs 0.800
    //    normalised. The trainer (train_xyz2.py, Xyz2Head.feedback_norm = True) feeds back the same tensor.
    // A narrow student feeds back its own d-space hidden (see the xyz_up comment above); a same-width MTP-fusion
    // head feeds back the normalised hidden, and classic xyz the pre-norm residual.
    const bool fusion_feedback = model.layers[0].nextn.eh_proj != nullptr && model.xyz_up == nullptr;
    ggml_tensor * fb = fusion_feedback ? cur : prenorm;
    ggml_set_output(fb);
    res->t_h_nextn = fb;

    // lm_head - projects to draft vocabulary
    // if the draft has no own output projection, inherit the target model's lm_head
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->output != nullptr && "XYZ decoder requires an output projection (own or from target model)");
        output = model_other->output;
    }
    cur = build_lora_mm(output, cur);

    // COMPACT logits: the head's own [n_draft_vocab] columns, not scattered into a -inf [n_vocab] row (a 993 KB download
    // per draft step for 32,768 finite values). The context copies them to the start of each n_vocab row and the draft
    // sampler maps column -> id through the GGUF's d2t (common/speculative.cpp, head_idx).

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // THE DEVICE DRAFT CHAIN: draw from the output row on the GPU -- the record lands in
    // dchain->rec[step], the drawn column in dchain->col, and this row's feedback hidden in dchain->g, which is exactly
    // what the next chain step reads. The logits are the head's compact columns.
    if (chain_draw && cur->ne[0] == dchain->ids->ne[0] && cur->ne[1] == 1) {
        auto inp_dc = std::make_unique<llm_graph_input_draft_chain>(dchain);
        inp_dc->key  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2);
        inp_dc->step = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(inp_dc->key);
        ggml_set_input(inp_dc->step);

        ggml_tensor * draw = ggml_draft_sample(ctx0, ggml_reshape_1d(ctx0, cur, cur->ne[0]), dchain->ids,
                inp_dc->key, inp_dc->step, dchain->rec, dchain->col, dchain->top_k, dchain->top_p);
        ggml_set_name(draw, "draft_chain_draw");
        ggml_build_forward_expand(gf, draw);

        GGML_ASSERT(fb->ne[0] == dchain->g->ne[0] && fb->ne[1] == 1);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_1d(ctx0, fb, fb->ne[0]), dchain->g));

        res->add_input(std::move(inp_dc));
    }
}
