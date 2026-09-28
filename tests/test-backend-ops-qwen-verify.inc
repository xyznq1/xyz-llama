struct test_gdn_prologue : public test_case {
    std::vector<ggml_tensor *> outputs;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GDN_PROLOGUE";
    }

    std::string vars() override {
        return "tokens=5,channels=10240,heads=32,head_size=128,rank=48,embd=5120";
    }

    bool run_whole_graph() override { return true; }

    std::vector<ggml_tensor *> fusion_test_nodes() override { return outputs; }

    double max_nmse_err() override { return 5e-4; }

    size_t op_size(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return size_t(1) << 36;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        constexpr int64_t n_tokens  = 5;
        constexpr int64_t channels  = 10240;
        constexpr int64_t conv_prev = 3;
        constexpr int64_t head_size = 128;
        constexpr int64_t qk_heads  = 32;
        constexpr int64_t rank      = 48;
        constexpr int64_t n_embd    = 5120;

        outputs.clear();

        ggml_tensor * states = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_prev * channels, 2);
        ggml_set_name(states, "gdn_states");
        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(rows, "gdn_state_row");
        ggml_tensor * gathered = ggml_get_rows(ctx, states, rows);
        ggml_set_name(gathered, "gdn_gathered");
        ggml_tensor * old_steps = ggml_reshape_3d(ctx, gathered, conv_prev, channels, 1);

        ggml_tensor * mixed_cf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, n_tokens, 1);
        ggml_set_name(mixed_cf, "gdn_mixed_cf");
        ggml_tensor * mixed_tf = ggml_transpose(ctx, mixed_cf);
        ggml_tensor * conv_input = ggml_concat(ctx, old_steps, mixed_tf, 0);
        ggml_set_name(conv_input, "gdn_conv_input");

        ggml_tensor * snapshot_store = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_prev * channels, n_tokens);
        for (int64_t t = 1; t <= n_tokens; ++t) {
            ggml_tensor * src = ggml_view_3d(ctx, conv_input, conv_prev, channels, 1,
                    conv_input->nb[1], conv_input->nb[2], t * sizeof(float));
            ggml_tensor * dst = ggml_view_2d(ctx, snapshot_store, conv_prev * channels, 1,
                    snapshot_store->nb[1], (n_tokens - t) * snapshot_store->nb[1]);
            ggml_tensor * snapshot = ggml_cpy(ctx, src, dst);
            ggml_format_name(snapshot, "gdn_prologue_snapshot_%" PRId64, t);
            ggml_build_forward_expand(gf, snapshot);
            outputs.push_back(snapshot);
        }

        ggml_tensor * conv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
        ggml_set_name(conv_weight, "gdn_conv_weight");
        ggml_tensor * conv = ggml_ssm_conv(ctx, conv_input, conv_weight);
        ggml_tensor * conv_silu = ggml_silu(ctx, conv);
        ggml_set_name(conv_silu, "gdn_prologue_conv_silu");

        ggml_tensor * qk = ggml_view_4d(ctx, conv_silu, head_size, qk_heads, n_tokens, 1,
                head_size * sizeof(float), channels * sizeof(float), channels * n_tokens * sizeof(float), 0);
        ggml_tensor * qk_norm = ggml_l2_norm(ctx, qk, 1e-6f);
        ggml_set_name(qk_norm, "gdn_prologue_qk_norm");
        ggml_build_forward_expand(gf, qk_norm);
        outputs.push_back(conv_silu);
        outputs.push_back(qk_norm);

        ggml_tensor * activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(activations, "gdn_activations");
        ggml_tensor * alpha_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, rank);
        ggml_tensor * beta_weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, rank);
        ggml_set_name(alpha_weight, "gdn_alpha_weight");
        ggml_set_name(beta_weight,  "gdn_beta_weight");

        ggml_tensor * alpha = ggml_mul_mat(ctx, alpha_weight, activations);
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rank);
        ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rank);
        ggml_tensor * gate = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), a);
        ggml_set_name(gate, "gdn_prologue_gate");
        ggml_build_forward_expand(gf, gate);
        outputs.push_back(gate);

        ggml_tensor * beta = ggml_mul_mat(ctx, beta_weight, activations);
        beta = ggml_reshape_4d(ctx, beta, 1, rank, n_tokens, 1);
        beta = ggml_sigmoid(ctx, beta);
        ggml_set_name(beta, "gdn_prologue_beta");
        ggml_build_forward_expand(gf, beta);
        outputs.push_back(beta);

        return beta;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x47444e35);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32 && !ggml_is_view_op(t->op)) {
                const int32_t row = 1;
                ggml_backend_tensor_set(t, &row, 0, sizeof(row));
                continue;
            }

            std::vector<float> data(ggml_nelements(t));
            for (float & value : data) {
                value = dist(rng);
            }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
            } else if (t->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> data_bf16(data.size());
                ggml_fp32_to_bf16_row(data.data(), data_bf16.data(), data.size());
                ggml_backend_tensor_set(t, data_bf16.data(), 0, data_bf16.size() * sizeof(ggml_bf16_t));
            } else {
                GGML_ABORT("unexpected GDN prologue test tensor type");
            }
        }
    }

};

static std::vector<std::unique_ptr<test_case>> make_test_cases_qwen_verify() {
    std::vector<std::unique_ptr<test_case>> cases;
    cases.emplace_back(new test_gdn_prologue());
    // Cover decode, verify, bulk generation, and prefill widths.
    for (int64_t n : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128, 512}) {
        for (ggml_type type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S,
                GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K, GGML_TYPE_IQ1_M,
                GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0}) {
            cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 17408, n, 5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 5120, n, 17408, {1, 1}, {1, 1}));
        }
        cases.emplace_back(new test_mul_mat(GGML_TYPE_IQ4_XS, GGML_TYPE_F32, 248320, n, 5120, {1, 1}, {1, 1}));
        // DRAM-bound PTQ1_0 shape.
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 69632, n, 5120, {1, 1}, {1, 1}));
        // Few-CTA PTQ1_0 shape.
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 5120, n, 52224, {1, 1}, {1, 1}));
        if (n == 1) {
            // Drafter q4_0 matmuls.
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32,  5120, 1, 17408, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 17408, 1,  5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 12288, 1,  5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32,  5120, 1, 12288, {1, 1}, {1, 1}));
            // Served Q3_K drafter matmuls.
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32, 17408, 1,  5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32,  5120, 1, 17408, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32, 12288, 1,  5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32,  5120, 1,  6144, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32,  1024, 1,  5120, {1, 1}, {1, 1}));
            cases.emplace_back(new test_mul_mat(GGML_TYPE_Q3_K, GGML_TYPE_F32,  5120, 1, 10240, {1, 1}, {1, 1}));
        }
        for (int64_t slots : {1, 4}) {
            cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 16, 128, n, 1, 3, false, false, slots));
        }
        cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {n + 3, 10240, 1, 1}, {4, 10240, 1, 1}));
        cases.emplace_back(new test_ssm_conv_bias_silu(GGML_TYPE_F32, {n + 3, 10240, 1, 1}, {4, 10240, 1, 1}, false));
        for (int64_t kv : {512, 16384}) {
            cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, n,
                    true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                    {0, 1, 2, 3}, false));
        }
        // Compare served xyzkv2 attention with the f16 reference at depth.
        for (int64_t kv : {32768, 131072}) {
            for (ggml_type kvt : {GGML_TYPE_XYZKV2_0, GGML_TYPE_F16}) {
                cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, n,
                        true, false, 0, 0, GGML_PREC_F32, kvt, kvt,
                        {0, 1, 2, 3}, false));
            }
        }
        // permute {0,2,1,3} creates the served head-interleaved cache layout.
        cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, 131072, n,
                true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0,
                {0, 2, 1, 3}, false));
        // Maximum served context with the same layout.
        cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, 163840, n,
                true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0,
                {0, 2, 1, 3}, false));
        // Drafter q4_0 KV at serving depth in the served layout.
        cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, 131072, n,
                true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                {0, 2, 1, 3}, false));
        cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {5120, n, 1, 1}));
        cases.emplace_back(new test_bin_bcast(ggml_mul, GGML_TYPE_F32, {5120, 1, 1, 1}, {1, (int) n, 1, 1}));
        cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {5120, n, 1, 1}, {1, 1, 1, 1}));
        cases.emplace_back(new test_glu_split(GGML_GLU_OP_SWIGLU, GGML_TYPE_F32, {17408, n, 1, 1}));
    }
    // Served prefill width.
    for (int64_t n : {2048}) {
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 17408, n, 5120, {1, 1}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32,  5120, n, 17408, {1, 1}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 10240, n, 5120, {1, 1}, {1, 1}));
    }

    // Deep-draft verify widths.
    for (int64_t n : {9, 12, 17}) {
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 17408, n, 5120, {1, 1}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 5120, n, 17408, {1, 1}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 69632, n, 5120, {1, 1}, {1, 1}));
        cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, 131072, n,
                true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0,
                {0, 2, 1, 3}, false));
    }
    // The CUDA backend stores PTQ1_0 in 16-row tiles, so every reader must handle
    // these right: a partial last tile (m % 16), 3 k-blocks (MMQ's 2-block steps read one block past the row), batched
    // matrices whose row count is not a tile multiple, a K-view (row stride > row length), a permuted batch (rows intact),
    // a rows-permuted view (must be refused, never misread), and get_rows. Widths cover v1 (<= 32), v2 (32 at >= 16384
    // rows: not here) and MMQ (> 32).
    for (int64_t n : {1, 5, 17, 32, 33, 64}) {
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 100, n, 384, {1, 1}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32,  40, n, 256, {3, 2}, {1, 1}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32,  48, n, 256, {2, 1}, {1, 1}, {0, 1, 2, 3}, 512));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32,  64, n, 256, {2, 3}, {1, 1}, {0, 1, 3, 2}));
        cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32,  64, n, 256, {2, 3}, {1, 1}, {0, 2, 1, 3}));
    }
    cases.emplace_back(new test_mul_mat(GGML_TYPE_PTQ1_0, GGML_TYPE_F32, 16392, 32, 512, {1, 1}, {1, 1}));   // v2 + partial tile (1024 + 8/16)
    for (int m : {5, 16, 100}) {
        cases.emplace_back(new test_get_rows(GGML_TYPE_PTQ1_0, 256, m, 7));
        cases.emplace_back(new test_get_rows(GGML_TYPE_PTQ1_0, 384, m, 7, 2, 1));
    }
    // PTQ1_0 fused gate+up+SWIGLU for the one-column drafter path.
    cases.emplace_back(new test_mul_mat_vec_fusion(GGML_TYPE_PTQ1_0, GGML_GLU_OP_SWIGLU, 1, 17408, 5120,
            false, 1, 1, false, false, true, false, {1, 1}));
    cases.emplace_back(new test_mul_mat_vec_fusion(GGML_TYPE_PTQ1_0, GGML_GLU_OP_SWIGLU, 1, 256, 1024,
            false, 1, 1, false, false, true, false, {4, 2}));
    // Compare the positional KQ vector with the equivalent f16 mask table.
    for (int64_t kv : {4096, 32768, 163840}) {
        for (int64_t nb : {17, 32, 33, 128, 512}) {
            for (bool pos : {false, true}) {
                cases.emplace_back(new test_flash_attn_ext_posmask(kv, nb, pos));
            }
        }
    }
    return cases;
}
