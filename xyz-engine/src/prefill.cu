// xyz-engine prompt path: a ubatch of more than 16 tokens through the target, as the server's prompt graph runs it.
// The server evaluates the graph with its fusions: RMS_NORM->MUL->MUL(signs)->Hadamard, the PTQ1_0 matmuls on
// MMQ, the bf16 decay/beta gates on cuBLAS, the conv chain (concat, state copy, ssm_conv + silu), the recurrence reading and
// writing the state row directly, the gated out-norm, the positional-mask attention. The engine runs the same values on
// its own bit-identical kernels; the bf16 gates above 8 columns use cuBLAS exactly as the server calls it.
#include "engine.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "common.cuh"
#include "convert.cuh"

namespace eng { extern ggml_backend_cuda_context * g_ctx; }

// every activation of one ubatch (NMAX tokens), allocated once
struct PrefillBufs {
    int32_t * tok = nullptr;      // [NMAX]
    int32_t * pos = nullptr;      // [4*NMAX] M-RoPE: t, h, w, then zeros
    int64_t * kv_idx = nullptr;   // [NMAX] the cells written
    float * mask_pos = nullptr;   // [NMAX + kv_size]
    float * emb = nullptr, * xf = nullptr, * x = nullptr, * xr = nullptr, * rot = nullptr, * norm_out = nullptr, * mm_out = nullptr;
    float * a_raw = nullptr, * b_raw = nullptr, * gate = nullptr, * beta = nullptr;
    __nv_bfloat16 * xbf = nullptr;
    float * qkv = nullptr, * z = nullptr, * conv_silu = nullptr, * qk_norm = nullptr, * gdn_out = nullptr, * rot6k = nullptr;
    float * q_full = nullptr, * k_cur = nullptr, * v_cur = nullptr, * qa = nullptr, * fa_out = nullptr;
    float * ffn_g = nullptr, * ffn_u = nullptr, * glu = nullptr, * glu_rot = nullptr;
    float * h = nullptr, * fold_in = nullptr;   // h_nextn [5120, NMAX]; the fold layer's input rows [5120, NMAX]
    half  * mask16 = nullptr;                   // a verify-width chunk's f16 mask [kv_size, 16]
    float * fold_cat_s = nullptr, * g_small = nullptr;   // a verify-width chunk's fold rows [10240, 16] -> g [5120, 16]
    // the replayed prefix of a batch right after a response: the conv window's rows of [state ++ pack], the replay count
    int32_t * conv_idx = nullptr, * prefix_n = nullptr;   // [3] = pending + {0, 1, 2}; [1] = pending
};

// the fold's input rows of a chunk: [the fold layer's input ; h_nextn] per row (llm_graph_context::build_fc_fold's concat)
static __global__ void k_pf_fold_rows(const float * __restrict__ a, const float * __restrict__ h, float * __restrict__ cat,
                                      const int n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (int64_t) 10240*n) {
        const int64_t r = i / 10240, c = i % 10240;
        cat[i] = c < 5120 ? a[r*5120 + c] : h[r*5120 + c - 5120];
    }
}

static void * pf_alloc(size_t bytes) {
    void * p = nullptr;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemset(p, 0, bytes));
    return p;
}

// ---- the engine's own kernels: exact operations only --------------------------------------------------------------------

static __global__ void k_pf_mul_signs(const float * __restrict__ x, const float * __restrict__ s, float * __restrict__ y,
                                      const int n, const int64_t total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = x[i] * s[i % n];
    }
}

static __global__ void k_pf_add(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ y, const int64_t total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = a[i] + b[i];
    }
}

// f32 -> bf16 of the cuBLAS gates' activation (convert.cu convert_unary<float, nv_bfloat16>: ggml_cuda_cast, round to
// nearest even)
static __global__ void k_pf_to_bf16(const float * __restrict__ x, __nv_bfloat16 * __restrict__ y, const int64_t total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        y[i] = ggml_cuda_cast<nv_bfloat16>(x[i]);
    }
}

// ggml-cuda.cu k_add_softplus_mul_f32, verbatim: ADD(x, dt_bias) -> SOFTPLUS -> MUL(ssm_a), the decay gate
static __global__ void k_add_softplus_mul_f32(const float * __restrict__ x, const float * __restrict__ bias,
                                              const float * __restrict__ a, float * __restrict__ dst,
                                              const int64_t ne0, const int64_t ne) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    const int64_t i0 = i % ne0;
    const float s = x[i] + bias[i0];                          // op_add
    const float p = (s > 20.0f) ? s : logf(1.0f + expf(s));   // op_softplus, verbatim from unary.cu
    dst[i] = p * a[i0];                                       // op_mul
}

// the ubatch's inputs from (p0, n): M-RoPE positions (t = h = w = p, then 0), the cells (compact: cell = position), and the
// positional mask -- the rows' p + 0.5, then the first n_kv cells' positions (+inf past the last written cell). This is
// set_input_kq_mask_dev's upload + build_kq_mask_pos's scatter/concat for one sequence in compact cells.
static __global__ void k_pf_inputs(int32_t * __restrict__ pos, int64_t * __restrict__ kv_idx, float * __restrict__ mask_pos,
                                   const int n, const int64_t p0, const int n_kv) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        const int32_t p = (int32_t) (p0 + i);
        pos[i] = p; pos[n + i] = p; pos[2*n + i] = p; pos[3*n + i] = 0;
        kv_idx[i] = p0 + i;
        mask_pos[i] = (float) p + 0.5f;
    }
    if (i < n_kv) {
        mask_pos[n + i] = i < p0 + n ? (float) i : INFINITY;
    }
}

// the f16 mask of a verify-width chunk (llama_kv_cache::build_kq_mask_dev's values: cell j visible to row r iff its position
// < p_r + 0.5; one sequence in compact cells, so cell j holds position j and the cells past the chunk are empty) -- from
// the last 64-aligned tile start at or below p0 + 1 only: flash_attn (vis_pos = the chunk's positions) reads none before
static __global__ void k_pf_mask_f16(half * __restrict__ mask, const int n, const int64_t p0, const int n_kv) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv || j < (p0 + 1) / 64 * 64) {
        return;
    }
    for (int r = 0; r < n; ++r) {
        mask[(int64_t) r*n_kv + j] = j <= p0 + r ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

namespace {

// ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16> for a 2D bf16 weight [K, nrows] and a contiguous f32 activation [K, n]:
// Ada prefers f32 output -> CUBLAS_COMPUTE_32F, C in f32, alpha 1 beta 0, the context's handle (TF32 math mode, its 4 MiB
// workspace, stream 0) -- cuBLAS picks the same algorithm for the same call
void gemm_bf16(cudaStream_t st, __nv_bfloat16 * xbf, const void * w, int64_t K, int64_t nrows, const float * x, int64_t n,
               float * dst) {
    const int64_t total = K*n;
    k_pf_to_bf16<<<(unsigned) ((total + 255)/256), 256, 0, st>>>(x, xbf, total);
    cublasHandle_t h = eng::g_ctx->cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) nrows, (int) n, (int) K,
        &alpha, w, CUDA_R_16BF, (int) K, xbf, CUDA_R_16BF, (int) K,
        &beta, dst, CUDA_R_32F, (int) nrows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// a GDN layer's conv section + recurrence for a prompt chunk, as the server's prompt graph runs them (build_conv_state +
// ssm_conv + l2_norm + gated_delta_net on the fused cache, the state rows read and written in place) -- on the engine's
// two kernels (k_gdn_own.cu; tools/pchain_test.cu: bit-identical at 1..512 rows, fresh and replaying). The conv window
// [state(3) ++ this chunk] per channel, the last 3 steps back into the state, ssm_conv + silu, the q/k L2 norms; the
// recurrence, the final state into the state row. Inputs: b.qkv [10240, n], b.gate / b.beta [48, n]; output: b.gdn_out
// [6144, n]. pending > 0 (the first batch after a response: the recurrent cell's last `pending` tokens are packed, not
// committed): delta-net-base.cpp's rs_replay unarmed path -- the conv window from rows [pending, pending + 3) of
// [state(3) ++ pack(5)] (conv_window_replayed), and the recurrence replaying the packed prefix first (gated_delta_net_ext:
// prefix_n pending, no flags -- replay, then commit through the batch)
void gdn_recurrence(cudaStream_t st, const Layer & l, PrefillBufs & b, int n, float eps, int pending) {
    eng::gdn_conv_prompt(st, l.conv_state, l.conv_pack, pending > 0 ? b.conv_idx : nullptr, b.qkv, n, l.conv_w, b.conv_silu,
                         b.qk_norm, eps);
    eng::GdnArgs a = {};
    a.q = b.qk_norm;  a.k = b.qk_norm + 16*128;  a.v = b.conv_silu + 4096;   // q heads 0..15, k heads 16..31, v after
    a.sq1 = 128;  a.sq2 = 32*128;  a.sq3 = (int64_t) 32*128*n;
    a.sv1 = 128;  a.sv2 = 10240;   a.sv3 = (int64_t) 10240*n;
    a.g = b.gate;  a.beta = b.beta;
    a.s = l.ssm_state;  a.dst = b.gdn_out;  a.state_out = l.ssm_state;
    a.prefix   = pending > 0 ? l.gdn_pack[0] : nullptr;   // the server's pack row
    a.prefix_n = pending > 0 ? b.prefix_n : nullptr;
    a.prefix_sstride = 92640;
    a.n_tokens = n;
    eng::gdn_prompt(st, a);
}

// gdn_recurrence's replay inputs for a pending batch: the conv window's rows of [state ++ pack] and the replay count
void upload_pending(PrefillBufs & b, cudaStream_t st, int pending) {
    if (pending > 0) {
        const int32_t h_idx[4] = { pending, pending + 1, pending + 2, pending };
        CK(cudaMemcpyAsync(b.conv_idx, h_idx, 3*sizeof(int32_t), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(b.prefix_n, h_idx + 3, sizeof(int32_t), cudaMemcpyHostToDevice, st));
    }
}

} // namespace

// ---- setup ------------------------------------------------------------------------------------------------------------

bool Engine::prefill_init() {
    if (pf != nullptr) {
        return true;
    }
    const int64_t N = pf_nmax;
    PrefillBufs * b = new PrefillBufs();
    const auto f = [&](int64_t n) { return (float *) pf_alloc((size_t) n*N*sizeof(float)); };
    b->tok      = (int32_t *) pf_alloc(N*sizeof(int32_t));
    b->pos      = (int32_t *) pf_alloc(4*N*sizeof(int32_t));
    b->kv_idx   = (int64_t *) pf_alloc(N*sizeof(int64_t));
    b->mask_pos = (float *) pf_alloc(((size_t) N + kv_size)*sizeof(float));
    b->emb = f(5120); b->xf = f(5120); b->x = f(5120); b->xr = f(5120); b->rot = f(5120); b->mm_out = f(5120);
    // ONE scratch region for a layer's intermediates: the prompt path is one stream, a layer is GDN or attention, and both
    // sets are dead once their output matmul has written mm_out -- the FFN's four then take the same bytes. 149,184 ->
    // 69,632 floats per token (-155 MiB at 512-token ubatches); the same values in other places. rot6k sits at one offset
    // for both branches.
    float * R = f(4*17408);
    const auto at = [&](int64_t off) { return R + off*N; };
    b->ffn_g = at(0); b->ffn_u = at(17408); b->glu = at(2*17408); b->glu_rot = at(3*17408);
    b->norm_out = at(0);
    b->xbf = (__nv_bfloat16 *) at(5120);                      // 5120 bf16 per token = 2560 floats
    b->a_raw = at(7680); b->b_raw = at(7728); b->gate = at(7776); b->beta = at(7824);
    b->qkv = at(7872); b->z = at(18112); b->conv_silu = at(24256); b->qk_norm = at(34496); b->gdn_out = at(38592);
    b->q_full = at(0); b->k_cur = at(12288); b->v_cur = at(13312); b->qa = at(14336); b->fa_out = at(20480);
    b->rot6k = at(44736);                                      // GDN's set ends at 50,880 floats, attention's at 26,624
    b->h = f(5120); b->fold_in = f(5120);
    b->mask16 = (half *) pf_alloc((size_t) 16*kv_size*sizeof(half));
    b->fold_cat_s = (float *) pf_alloc((size_t) 10240*16*sizeof(float));
    b->g_small    = (float *) pf_alloc((size_t) 5120*16*sizeof(float));
    b->conv_idx   = (int32_t *) pf_alloc(4*sizeof(int32_t));
    eng::mmq_reserve(N, 17408);   // the MMQ activation buffer: N columns of the widest K (the FFN down)
    b->prefix_n   = (int32_t *) pf_alloc(sizeof(int32_t));
    pf = b;
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    fprintf(stderr, "engine: prompt path for %d-token ubatches, %.2f GiB free\n", pf_nmax, fr / 1073741824.0);
    return true;
}

float * Engine::pf_g_small() const { return pf ? pf->g_small : nullptr; }

// ---- one ubatch ---------------------------------------------------------------------------------------------------------

// the xyz encoder of the last prompt ubatch's rows (its process() without the FC fold in the graph): fc over
// [the fold layer's input ; h_nextn] per row at the ubatch's width (routed as the encoder graph's mul_mat is: MMQ, MMVQ at
// a few rows), into g [5120, n] (device)
bool Engine::prefill_fold(const int n, float * g) {
    if (pf == nullptr || fc_w == nullptr || n > pf_nmax) {
        return false;
    }
    float * cat = pf->ffn_g;   // [10240, n] fits the FFN scratch, idle between ubatches
    k_pf_fold_rows<<<(unsigned) ((10240LL*n + 255)/256), 256, 0, st>>>(pf->fold_in, pf->h, cat, n);
    eng::mm_q(st, fc_type, fc_w, cat, 10240, 5120, n, g, q8f);
    CK(cudaGetLastError());
    return true;
}

bool Engine::prefill_ubatch(const int32_t * tokens, const int n, const bool want_logits, const int pending) {
    if (n <= 16 || n > pf_nmax || pending < 0 || pending > P) {
        fprintf(stderr, "engine: prefill_ubatch takes 17..%d tokens, 0..%d pending (%d, %d)\n", pf_nmax, P, n, pending);
        return false;
    }
    if (!prefill_init()) {
        return false;
    }
    PrefillBufs & b = *pf;
    const Hparams & hp = m.hp;
    const int64_t p0 = n_past;
    if (p0 + n > kv_size) {
        fprintf(stderr, "engine: context full (%lld + %d > %d)\n", (long long) p0, n, kv_size);
        return false;
    }
    const int n_kv = (int) std::min<int64_t>(kv_size, std::max<int64_t>(256, (p0 + n + 255) / 256 * 256));

    CK(cudaMemcpyAsync(b.tok, tokens, (size_t) n*sizeof(int32_t), cudaMemcpyHostToDevice, st));
    k_pf_inputs<<<(std::max(n, n_kv) + 255)/256, 256, 0, st>>>(b.pos, b.kv_idx, b.mask_pos, n, p0, n_kv);
    upload_pending(b, st, pending);
    if (p0 == 0) {   // build_rs's state_zero (SCALE by 0) of a fresh row: every recurrent row starts at +0
        for (Layer & l : L) {
            if (!l.attn) {
                CK(cudaMemsetAsync(l.conv_state, 0, (size_t) 3*10240*sizeof(float), st));
                CK(cudaMemsetAsync(l.ssm_state, 0, (size_t) 128*128*48*sizeof(float), st));
            }
        }
    }

    // the PTQ1_0 matmuls as ggml_cuda_mul_mat routes them at this width: MMQ above PTQ1_MMA_MAX_COLS (32) columns, the
    // tensor-core MMVQ kernel at 17..32 (ggml_cuda_should_use_mmvq: ptq1_mma_use), its activation quantized per call
    const auto pmm = [&](const void * w, int K, int nrows, const float * x, float * dst, int dst_stride) {
        if (n > 32) {
            eng::mmq_ptq1(st, w, K, nrows, x, K, n, dst, dst_stride);
        } else {
            eng::ptq1_mm_f32(st, w, x, K, nrows, n, dst, dst_stride, q8);
        }
    };

    // embedding: rows of the rotated table, butterfly, then signs
    eng::get_rows_ptq1(st, tok_embd, 5120, hp.n_vocab, b.tok, n, b.emb, tok_ilv);
    eng::fwht_block(st, b.emb, b.xf, (int64_t) 5*n, nullptr, 1, nullptr);
    k_pf_mul_signs<<<(unsigned) ((5120LL*n + 255)/256), 256, 0, st>>>(b.xf, s5120, b.x, 5120, 5120LL*n);

    for (int il = 0; il < hp.n_layer; ++il) {
        const Layer & l = L[il];
        // [ffn residual ->] attn norm -> signs -> Hadamard (+ the unrotated norm for the GDN's gates)
        if (il == 0) {
            eng::norm_fwht(st, b.x, nullptr, nullptr, l.attn_norm, s5120, l.attn ? nullptr : b.norm_out, b.rot, nullptr, n, eps);
        } else {
            eng::norm_fwht(st, b.mm_out, b.xr, b.x, l.attn_norm, s5120, l.attn ? nullptr : b.norm_out, b.rot, nullptr, n, eps);
        }
        if (il == fc_layer) {   // the fold's first half: this layer's input
            CK(cudaMemcpyAsync(b.fold_in, b.x, (size_t) 5120*n*sizeof(float), cudaMemcpyDeviceToDevice, st));
        }
        if (!l.attn) {
            // decay gate: softplus(alpha . x + dt) * a; beta: sigmoid(beta . x) -- cuBLAS bf16, the fused chain, the unary
            gemm_bf16(st, b.xbf, l.w_alpha, 5120, 48, b.norm_out, n, b.a_raw);
            k_add_softplus_mul_f32<<<(unsigned) ((48LL*n + 255)/256), 256, 0, st>>>(b.a_raw, l.dt, l.a, b.gate, 48, 48LL*n);
            gemm_bf16(st, b.xbf, l.w_beta, 5120, 48, b.norm_out, n, b.b_raw);
            eng::sigmoid(st, b.b_raw, b.beta, 48LL*n);
            pmm(l.w_qkv_a, 5120, 10240, b.rot, b.qkv, 10240);
            pmm(l.w_z, 5120, 6144, b.rot, b.z,   6144);
            gdn_recurrence(st, l, b, n, eps, pending);
            // gated out-norm rms_norm(o) * ssm_norm * silu(z), ssm_out's head permutation (build_hadamard_rotate:
            // [128, 16, 3, n] -> [128, 3, 16, n]), signs, Hadamard -- one chain -- then the matmul
            eng::gdn_out_chain(st, b.gdn_out, l.ssm_norm, b.z, s6144, b.rot6k, eps, nullptr, n);
            pmm(l.w_ssm_out, 6144, 5120, b.rot6k, b.mm_out, 5120);
        } else {
            pmm(l.w_q, 5120, 12288, b.rot, b.q_full, 12288);
            pmm(l.w_k, 5120, 1024, b.rot, b.k_cur,  1024);
            pmm(l.w_v, 5120, 1024, b.rot, b.v_cur,  1024);
            // Q: norm, M-RoPE, the KV rotation's 256-point Hadamard, the xyzkv forward WHT; V: the 64-point Hadamard; K:
            // norm, M-RoPE, the 256-point Hadamard -- both xyzkv2 rows into the cache (the chains: tools/pchain_test.cu)
            eng::attn_q_chain(st, b.q_full, 512, 12288, l.q_norm, eps, b.pos, n, rope, ones, b.qa, 24);
            eng::attn_v_write(st, b.v_cur, 1024, b.kv_idx, l.v_cache, 1024/128*34, 4, n);
            eng::attn_k_write(st, b.k_cur, 256, 1024, l.k_norm, eps, b.pos, n, rope, b.kv_idx, l.k_cache, 1024/128*34, 4);
            // attention over the first n_kv cells, the positional mask
            eng::flash_attn_prefill(st, b.qa, n, 24, l.k_cache, l.v_cache, n_kv, kv_size, b.mask_pos, 1.0f/16.0f, b.fa_out);
            // xyzkv inverse WHT, the V rotation's Hadamard, the gate sigmoid(Q's second half) * x, signs, Hadamard
            eng::attn_out_chain(st, b.fa_out, b.q_full + 256, 512, 12288, ones, s6144, b.rot6k, nullptr, n);
            pmm(l.w_o, 6144, 5120, b.rot6k, b.mm_out, 5120);
        }
        // attn residual -> post-attention norm -> signs -> Hadamard; FFN
        eng::norm_fwht(st, b.mm_out, b.x, b.xr, l.post_norm, s5120, nullptr, b.rot, nullptr, n, eps);
        pmm(l.w_gate, 5120, 17408, b.rot, b.ffn_g, 17408);
        pmm(l.w_up, 5120, 17408, b.rot, b.ffn_u, 17408);
        eng::silu_gate(st, b.ffn_g, b.ffn_u, b.glu, 17408, n);
        eng::fwht_block(st, b.glu, b.glu_rot, (int64_t) 17*n, s17408, 17, nullptr);
        pmm(l.w_down, 17408, 5120, b.glu_rot, b.mm_out, 5120);
    }
    // the last residual and the output norm: h_nextn for every row (the drafter's features)
    k_pf_add<<<(unsigned) ((5120LL*n + 255)/256), 256, 0, st>>>(b.mm_out, b.xr, b.xf, 5120LL*n);
    eng::rms_norm_mul(st, b.xf, out_norm, b.h, 5120, n, eps);
    if (want_logits) {   // the output row (inp_out_ids: the last): its rotation with the q8 twin, the LM head, one column
        eng::fwht_block(st, b.h + (size_t) 5120*(n - 1), b.xf, 5, s5120, 5, q8);
        eng::ptq1_matmul(st, w_out, q8, 5120, hp.n_vocab, 1, logits, hp.n_vocab);
    }
    CK(cudaGetLastError());
    n_past += n;
    return true;
}

// ---- a verify-width chunk (1..16 tokens) --------------------------------------------------------------------------------

bool Engine::prefill_small(const int32_t * tokens, const int n, const bool want_logits, const int pending) {
    if (n < 1 || n > 16 || pending < 0 || pending > P) {
        fprintf(stderr, "engine: prefill_small takes 1..16 tokens, 0..%d pending (%d, %d)\n", P, n, pending);
        return false;
    }
    if (!prefill_init()) {
        return false;
    }
    PrefillBufs & b = *pf;
    const Hparams & hp = m.hp;
    const int64_t p0 = n_past;
    if (p0 + n > kv_size) {
        fprintf(stderr, "engine: context full (%lld + %d > %d)\n", (long long) p0, n, kv_size);
        return false;
    }
    const int n_kv = (int) std::min<int64_t>(kv_size, std::max<int64_t>(256, (p0 + n + 255) / 256 * 256));

    CK(cudaMemcpyAsync(b.tok, tokens, (size_t) n*sizeof(int32_t), cudaMemcpyHostToDevice, st));
    k_pf_inputs<<<(std::max(n, n_kv) + 255)/256, 256, 0, st>>>(b.pos, b.kv_idx, b.mask_pos, n, p0, n_kv);
    k_pf_mask_f16<<<(n_kv + 255)/256, 256, 0, st>>>(b.mask16, n, p0, n_kv);
    upload_pending(b, st, pending);
    if (p0 == 0) {
        for (Layer & l : L) {
            if (!l.attn) {
                CK(cudaMemsetAsync(l.conv_state, 0, (size_t) 3*10240*sizeof(float), st));
                CK(cudaMemsetAsync(l.ssm_state, 0, (size_t) 128*128*48*sizeof(float), st));
            }
        }
    }

    eng::get_rows_ptq1(st, tok_embd, 5120, hp.n_vocab, b.tok, n, b.emb, tok_ilv);
    eng::fwht_block(st, b.emb, b.xf, (int64_t) 5*n, nullptr, 1, nullptr);
    k_pf_mul_signs<<<(unsigned) ((5120LL*n + 255)/256), 256, 0, st>>>(b.xf, s5120, b.x, 5120, 5120LL*n);

    float * gate_up = b.ffn_g;   // [34816, n]
    for (int il = 0; il < hp.n_layer; ++il) {
        const Layer & l = L[il];
        if (il == 0) {
            eng::norm_fwht(st, b.x, nullptr, nullptr, l.attn_norm, s5120, l.attn ? nullptr : b.norm_out, b.rot, q8, n, eps);
        } else {
            eng::norm_fwht(st, b.mm_out, b.xr, b.x, l.attn_norm, s5120, l.attn ? nullptr : b.norm_out, b.rot, q8, n, eps);
        }
        if (il == fc_layer) {
            CK(cudaMemcpyAsync(b.fold_in, b.x, (size_t) 5120*n*sizeof(float), cudaMemcpyDeviceToDevice, st));
        }
        if (!l.attn) {
            // the gates: the bf16 [5120 x 48] matmuls as ggml_cuda_mul_mat routes them -- mmvf while the fork's
            // ggml_cuda_should_use_mmvf admits the width (a BF16 weight of <= 512 rows: <= 8 columns; the server sets no
            // GGML_CUDA_MMVF_MAX_N), else cuBLAS (mmf refuses 48 rows: not a multiple of MMF_ROWS_PER_BLOCK); then the
            // fused decay chain, the sigmoid
            const auto gate_mm = [&](const void * w, float * dst) {
                if (n > 8) {
                    gemm_bf16(st, b.xbf, w, 5120, 48, b.norm_out, n, dst);
                    return;
                }
                eng::mmvf_bf16(st, w, b.norm_out, 5120, 48, n, dst);   // mmvf's kernel (k_small.cu, tools/pchain_test.cu)
            };
            gate_mm(l.w_alpha, b.a_raw);
            k_add_softplus_mul_f32<<<(unsigned) ((48LL*n + 255)/256), 256, 0, st>>>(b.a_raw, l.dt, l.a, b.gate, 48, 48LL*n);
            gate_mm(l.w_beta, b.b_raw);
            eng::sigmoid(st, b.b_raw, b.beta, 48LL*n);
            if (l.w_qkvz != nullptr) {
                eng::ptq1_matmul(st, l.w_qkvz, q8, 5120, 16384, n, b.qkv, 10240, b.z, 10240, 6144);
            } else {
                eng::ptq1_matmul(st, l.w_qkv_a, q8, 5120, 10240, n, b.qkv, 10240);
                eng::ptq1_matmul(st, l.w_z, q8, 5120, 6144, n, b.z, 6144);
            }
            gdn_recurrence(st, l, b, n, eps, pending);
            eng::gdn_out_chain(st, b.gdn_out, l.ssm_norm, b.z, s6144, b.rot6k, eps, q8, n);
            eng::ptq1_matmul(st, l.w_ssm_out, q8, 6144, 5120, n, b.mm_out, 5120);
        } else {
            if (l.w_qkv != nullptr) {
                eng::ptq1_matmul(st, l.w_qkv, q8, 5120, 14336, n, b.q_full, 12288, b.k_cur, 12288, 1024, b.v_cur, 13312, 1024);
            } else {
                eng::ptq1_matmul(st, l.w_q, q8, 5120, 12288, n, b.q_full, 12288);
                eng::ptq1_matmul(st, l.w_k, q8, 5120, 1024, n, b.k_cur, 1024);
                eng::ptq1_matmul(st, l.w_v, q8, 5120, 1024, n, b.v_cur, 1024);
            }
            eng::attn_q_chain(st, b.q_full, 512, 12288, l.q_norm, eps, b.pos, n, rope, ones, b.qa, 24);
            eng::attn_v_write(st, b.v_cur, 1024, b.kv_idx, l.v_cache, 1024/128*34, 4, n);
            eng::attn_k_write(st, b.k_cur, 256, 1024, l.k_norm, eps, b.pos, n, rope, b.kv_idx, l.k_cache, 1024/128*34, 4);
            eng::flash_attn(st, b.qa, n, 24, l.k_cache, l.v_cache, n_kv, kv_size, b.mask16, 1.0f/16.0f, b.fa_out, b.pos);
            eng::attn_out_chain(st, b.fa_out, b.q_full + 256, 512, 12288, ones, s6144, b.rot6k, q8, n);
            eng::ptq1_matmul(st, l.w_o, q8, 6144, 5120, n, b.mm_out, 5120);
        }
        eng::norm_fwht(st, b.mm_out, b.x, b.xr, l.post_norm, s5120, nullptr, b.rot, q8, n, eps);
        if (l.w_gate_up != nullptr) {
            eng::ptq1_matmul(st, l.w_gate_up, q8, 5120, 2*17408, n, gate_up, 2*17408);
        } else {
            eng::ptq1_matmul(st, l.w_gate, q8, 5120, 17408, n, gate_up, 2*17408);
            eng::ptq1_matmul(st, l.w_up, q8, 5120, 17408, n, gate_up + 17408, 2*17408);
        }
        eng::fwht_glu(st, gate_up, b.glu_rot, (int64_t) 17*n, s17408, 17, q8, 17408);
        eng::ptq1_matmul(st, l.w_down, q8, 17408, 5120, n, b.mm_out, 5120);
    }
    k_pf_add<<<(unsigned) ((5120LL*n + 255)/256), 256, 0, st>>>(b.mm_out, b.xr, b.xf, 5120LL*n);
    eng::rms_norm_mul(st, b.xf, out_norm, b.h, 5120, n, eps);
    if (fc_w != nullptr) {   // the FC fold: concat_cont followed by the fc matmul, MMVQ to 6 rows and MMQ above
        k_pf_fold_rows<<<(unsigned) ((10240LL*n + 255)/256), 256, 0, st>>>(b.fold_in, b.h, b.fold_cat_s, n);
        eng::mm_q(st, fc_type, fc_w, b.fold_cat_s, 10240, 5120, n, b.g_small, q8f);
    }
    if (want_logits) {   // the last row only (inp_out_ids): its rotation with the q8 twin, the LM head
        eng::fwht_block(st, b.h + (size_t) 5120*(n - 1), b.xf, 5, s5120, 5, q8);
        eng::ptq1_matmul(st, w_out, q8, 5120, hp.n_vocab, 1, logits, hp.n_vocab);
    }
    CK(cudaGetLastError());
    n_past += n;
    return true;
}
