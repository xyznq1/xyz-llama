#pragma once
// xyz-engine's kernel launchers. Every kernel they run is the engine's own (src/k_*.cu, src/fa, src/mmq), each
// bit-identical to the corresponding kernel in the parent llama.cpp fork at that shape.
// cuBLAS handles the prompt path's bf16 gates above 8 columns exactly as the server calls it.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace eng {

// the attention layers' M-RoPE parameters as rope_multi computes them on the host (mrope_params)
struct MRope {
    int   n_dims;
    int   n_offs;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    int   sections[4];
    bool  is_imrope;
};

// ---- k_hadamard.cu: the engine's own rotation kernels -----------------------------------------------------------------
// [ADD ->] RMS_NORM -> MUL(w) -> signs -> 1024-block Hadamard over 5120-wide rows, + the q8_1 twin: rms_norm_mul_fwht_f32_5k_split
// <b != null, norm_out != null>, 5 CTAs per row. x_out = a + b (the residual) when b is given.
void norm_fwht(cudaStream_t st, const float * a, const float * b, float * x_out, const float * w, const float * signs,
               float * norm_out, float * dst, char * q8, int nrows, float eps);
// fwht_cuda_block<1024, 256, signs != null, false>: rows of 1024; block r uses signs[(r % n_blk)*1024 ...]
void fwht_block(cudaStream_t st, const float * src, float * dst, int64_t rows, const float * signs, int n_blk, char * q8);
// fwht_cuda_block<1024, 256, true, true>: silu(gate) * up from a [2*glu_nc, tokens] SWIGLU input, signed, transformed
void fwht_glu(cudaStream_t st, const float * glu_src, float * dst, int64_t rows, const float * signs, int n_blk, char * q8,
              int64_t glu_nc);
// gdn_out_chain_f32<true>: gated RMS norm (ssm_norm, silu(z)), the ssm_out head permutation, signs, Hadamard, q8 twin
void gdn_out_chain(cudaStream_t st, const float * x, const float * w, const float * gate, const float * signs, float * dst,
                   float eps, char * q8, int n_tokens);
// attn_out_chain_f32: xyzkv inverse WHT, the V rotation's 64-point Hadamard, sigmoid(gate), signs, Hadamard, q8 twin
void attn_out_chain(cudaStream_t st, const float * x, const float * gate, int64_t gate_s1, int64_t gate_s2,
                    const float * scale_inv, const float * signs, float * dst, char * q8, int n_tokens);

// ---- k_ptq1_own.cu: the PTQ1_0 matmuls (the fork's mul_mat_ptq1_mma arithmetic, bit-identical to its v1), 1..32 columns:
// dst[r + c*dst_stride] = W[r] . x[c] for the PTQ1_0 ILV16 matrix W (nrows x K) and the q8_1 twin x (ncols columns).
// Rows >= hi_row0 go to hi (a sibling matmul merged into the launch), rows >= hi2_row0 to hi2.
// to 8: variant 0 = one k-block per warp iteration, 1 = two (5 CTAs/SM), 2 = two (6/SM), 3 = four (4/SM); 9..32: 2 or 4
// 8-column tiles per CTA (variant ignored)
void ptq1_own(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst, int dst_stride,
              int variant, float * hi = nullptr, int hi_row0 = 0, int hi_stride = 0,
              float * hi2 = nullptr, int hi2_row0 = 0, int hi2_stride = 0);

// the engine's PTQ1 matmul at 1..32 columns, routed as the server routes mul_mat_ptq1_mma: the one-to-four-tile kernel,
// or the v2 kernel where ptq1_mma_version picks v2
void ptq1_matmul(cudaStream_t st, const void * w, const void * q8, int K, int nrows, int ncols, float * dst, int dst_stride,
                 float * hi = nullptr, int hi_row0 = 0, int hi_stride = 0,
                 float * hi2 = nullptr, int hi2_row0 = 0, int hi2_stride = 0);

// ---- k_gdn_pro.cu (kernels copied verbatim from fork ggml-cuda.cu) --------------------------------------------------
// alpha/beta prologue: gate[t][h] = softplus(alpha_h . x_t + dt[h]) * a[h], beta[t][h] = sigmoid(beta_h . x_t), 5 tokens
void gdn_alpha_beta(cudaStream_t st, const void * alpha_w, const void * beta_w, const float * x, const float * dt,
                    const float * a, float * gate, float * beta);
// the rs_replay conv section: window [state(3) ++ pack(P)] rows conv_idx[0..2] + this batch, commit + park, ssm_conv,
// silu, and the q/k L2 norms. n_tokens = 5.
struct ConvReplay {
    const char *    state;             // conv state table (rows of [C, 3], channel-fastest)
    const char *    pack;              // conv pack table (rows of [C, P])
    const int32_t * s_row;             // the cell row this batch reads
    size_t          state_row_stride;
    size_t          pack_row_stride;
    const int32_t * conv_idx;          // [3]: rows of [state ++ pack] that precede this batch
    const float *   qkv;               // this batch's inputs [C, n]
    size_t          qkv_nb1;
    float *         state_dst;         // commit
    float *         pack_dst;          // park
    size_t          pack_dst_nb1;
    const float *   conv_weight;       // [4, C]
    size_t          conv_weight_nb1;
    float *         conv_silu;         // [C, n]
    float *         qk_norm;           // [128, 32, n]
    float           eps;
};
void gdn_conv_replay(cudaStream_t st, const ConvReplay & a, int n_tokens);

// ---- k_gdn_own.cu: the recurrence (the fork's gated_delta_net.cu arithmetic) -------------------------------------------
// with prefix replay (armed mode: replay prefix_n[0] packed tokens, commit the state after them, run this batch from
// there, pack this batch's inputs)
struct GdnArgs {
    const float * q; const float * k; const float * v;   // q/k: [128, 16, n] rows of qk_norm; v: rows of conv_silu
    int64_t sq1, sq2, sq3, sv1, sv2, sv3;                 // strides in floats
    const float * g; const float * beta;                  // [48, n]
    const float * s;                                      // committed state [128, 128, 48]
    float * dst;                                          // attention output [128, 48, n]
    float * state_out;                                    // the committed state's destination
    const float * prefix; const int32_t * prefix_n;       // pack table row (P packed tokens) and the replay count
    int64_t prefix_sstride;
    float * pack_out;                                     // this batch's pack
    int n_tokens;
};
void gdn_own(cudaStream_t st, const GdnArgs & a);
// a prompt chunk's recurrence (the fused cache, flags 0): replay a.prefix_n[0] packed tokens of a.prefix if given, run
// the chunk, commit the final state into a.state_out (may be a.s: in place); no pack
void gdn_prompt(cudaStream_t st, const GdnArgs & a);
// a prompt chunk's conv section: the window [3 old steps ++ qkv [10240, n]] (old steps: state [3][10240], or with conv_idx
// rows conv_idx[0..2] of [state ++ pack [5][10240]]), the 4-tap conv + silu -> conv_silu [10240, n], steps n..n+2 back
// into state, the 32 q/k heads' L2 norms -> qk_norm [128, 32, n]
void gdn_conv_prompt(cudaStream_t st, float * state, const float * pack, const int32_t * conv_idx, const float * qkv, int n,
                     const float * conv_w, float * conv_silu, float * qk_norm, float eps);

// ---- k_attn_chain.cu: the engine's own attention chains -------------------------------------------------------------
MRope mrope_params(int n_dims, const int sections[4], int mode, int n_ctx_orig, float freq_base, float freq_scale,
                   float ext_factor, float attn_factor, float beta_fast, float beta_slow);
// Q: RMS norm (q_norm) + M-RoPE + the KV rotation's 256-point Hadamard + xyzkv forward WHT -> dst [256, 24, n]
void attn_q_chain(cudaStream_t st, const float * x, int64_t sx1, int64_t sx2, const float * w, float eps, const int32_t * pos,
                  int n_tokens, const MRope & rp, const float * scale_inv, float * dst, int n_heads);
// K: the same prologue, xyzkv2-quantized into cache rows idx[t]
void attn_k_write(cudaStream_t st, const float * x, int64_t sx1, int64_t sx2, const float * w, float eps, const int32_t * pos,
                  int n_tokens, const MRope & rp, const int64_t * idx, char * cache, int64_t row_bytes, int n_heads);
// V: the V rotation's 64-point Hadamard, xyzkv2-quantized into cache rows idx[t]
void attn_v_write(cudaStream_t st, const float * v, int64_t sv2, const int64_t * idx, char * cache, int64_t row_bytes,
                  int n_heads, int n_tokens);

// ---- k_fa.cu: the engine's own attention (src/fa/fa.cuh) -------------------------------------------------------------
// fa_init(): once, outside any capture (shared-memory limits, occupancies, the stream-K fixup buffer)
void fa_init();
// q [256, n_head, n] (the Q chain's output), K/V xyzkv2 caches [1024, kv_size], mask f16 [n_kv, n], dst [256, n_head, n];
// n <= 32 rows (the tiles 1x8, 2x8, 4x8 pair-packed, 8x8). vis_pos (device, optional): the first row's position when
// every row sees cells 0 .. *vis_pos (a causal batch of one sequence in compact cells) -- the KV tiles wholly inside that
// prefix skip their all-zero mask (not copied, not added; bit-identical), and the mask there need not be written
void flash_attn(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                int n_kv, int kv_size, const half * mask, float scale, float * dst, const int * vis_pos = nullptr);

// the (F16, F16) 64-column kernel's occupancy the engine sizes grids by above 32 rows (after fa_init)
int fa_twin_occupancy();

// ---- k_mmq.cu: the engine's own MMQ (src/mmq/mmq.cuh), the prompt path's and the drafter's batch matmuls ------------
// mmq_reserve: the activation buffer for up to max_cols x max_K (and, once, the fixup buffer and the kernels' shared-memory
// limits) -- call outside any capture before a captured launch needs it
void mmq_reserve(int64_t max_cols, int64_t max_K);
// dst[r + c*dst_stride] = W[r] . x[c], W PTQ1_0 [K, nrows], x f32 [K, ncols] at column stride x_stride floats
void mmq_ptq1(cudaStream_t st, const void * w, int64_t K, int64_t nrows, const float * x, int64_t x_stride, int64_t ncols,
              float * dst, int64_t dst_stride);
// the same for a weight of type `type` (PTQ1_0 or Q3_K: the drafter's matmuls at prompt width)
void mmq(cudaStream_t st, int type, const void * w, int64_t K, int64_t nrows, const float * x, int64_t x_stride,
         int64_t ncols, float * dst, int64_t dst_stride);
// the prompt path (> 16 rows): the 64-column tile (ncols1 8 x ncols2 8) with the positional mask mask_pos, f32
// [n_tokens + n_kv] (llama_kv_cache::build_kq_mask_pos: the rows' p + 0.5, then the cells' positions, +inf when empty);
// its grid from the plain tile's occupancy up to 32 rows, the f16 twin's above (the fork's rule)
void flash_attn_prefill(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                        int n_kv, int kv_size, const float * mask_pos, float scale, float * dst);
// the drafter's attention at prompt width: to 32 rows as flash_attn_q4_0; above, where the server converts the q4_0 cache
// to f16 and runs the f16 64-column tile, the same tile on the same values natively (ncols1 8 x ncols2 8, the f16
// kernel's grid); mask f16 [n_kv, n_tokens]
void flash_attn_q4_0_prompt(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache,
                            const void * v_cache, int n_kv, int kv_size, const half * mask, float scale, float * dst);

// drafter step: q4_0 K/V, one token (ncols1 2 x ncols2 8)
void flash_attn_q4_0(cudaStream_t st, const float * q, int n_tokens, int n_head, const void * k_cache, const void * v_cache,
                     int n_kv, int kv_size, const half * mask, float scale, float * dst);

// ---- drafter kernels (k_mmvq.cu, k_rope.cu, k_fwht.cu, k_setrows.cu, k_unary.cu) --------------------------------------
// one column of a quantized weight (Q3_K), the activation quantized to q8_1 into q8; gate + glu_op = the fused GLU (w =
// the UP weight), x_bias = the fused residual add
// One-column Q3_K matvecs at their selected launch shapes.
void mmvq_fast(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, float * dst, char * q8,
               const void * gate = nullptr, int glu_op = 0, const float * x_bias = nullptr);
// one quantization of x and ONE launch for the three matrices of a drafter step's Q, K and V: every output bit three mmvq's
void mmvq_qkv(cudaStream_t st, int rows, int type, const float * x, int K, char * q8, const void * wq, int nq, float * q,
              const void * wk, int nk, float * k, const void * wv, int nv, float * v);
// fused qk RMS_NORM + MUL + ROPE (NEOX): nrows heads of 256 at row stride sx1 floats, one token at pos[0]
void rms_norm_mul_rope(cudaStream_t st, const float * x, int64_t sx1, int nrows, const float * w, float eps, float * dst,
                       const int32_t * pos, int n_dims, float freq_base, int n_ctx_orig, int n_tok = 1, int64_t sx2 = 0);
// the KV rotation: n-point Hadamard (n = 64 or 256, no signs) over rows
void fwht_rows(cudaStream_t st, const float * src, float * dst, int n, int64_t rows);
// SET_ROWS into a q4_0 cache
void set_rows_q4_0(cudaStream_t st, const float * src, int64_t ncols, int n_tok, const int64_t * idx, void * cache,
                   int64_t row_bytes);
// dst = sigmoid(gate) * x over rows of n
// dst = sigmoid(x) over k values (a GDN layer's beta)
void sigmoid(cudaStream_t st, const float * x, float * dst, int64_t k);
// dst [nrows, ncols] = W x for a bf16 weight [K, nrows] and f32 columns [K, ncols <= 8] (mmvf's no-fusion path: the GDN
// gates of a verify-width prompt chunk)
void mmvf_bf16(cudaStream_t st, const void * w, const float * x, int K, int nrows, int ncols, float * dst);
void sigmoid_gate(cudaStream_t st, const float * gate, const float * x, float * dst, int64_t n, int64_t rows);
// dst = silu(gate) * up over rows of n (the SWIGLU of an unfused FFN: ggml_swiglu_split(gate, up))
void silu_gate(cudaStream_t st, const float * gate, const float * up, float * dst, int64_t n, int64_t rows);
// ncols columns of x [K, ncols] into the plain q8_1 blocks (quantize_q8_1<false>, rows padded to MATRIX_ROW_PADDING)
void q8_1_quantize(cudaStream_t st, const float * x, int K, int ncols, char * q8);
// ncols columns of a quantized weight, no fusion (a multi-row drafter decode): mul_mat_vec_q<type, ncols, 0, 0, 0>
void mmvq_n(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, int ncols, float * dst, char * q8);
// the same unfused mul_mat at any width, routed as the server routes it (mmvq_n while ggml_cuda_should_use_mmvq, else mmq)
void mm_q(cudaStream_t st, int type, const void * w, const float * x, int K, int nrows, int ncols, float * dst, char * q8);
// PTQ1_0 [K, nrows] x f32 [K, ncols <= 32] on the tensor-core MMVQ kernel, quantizing the activation itself (no twin):
// the engine's quantizer + ptq1_matmul (k_ptq1_own.cu)
void ptq1_mm_f32(cudaStream_t st, const void * w, const float * x, int K, int nrows, int ncols, float * dst, int dst_stride,
                 char * q8);

// ---- k_topk.cu (fork top-k.cu): the device draft chain's draw ---------------------------------------------------------
void draft_draw(cudaStream_t st, const float * logits, const int32_t * col_ids, const uint32_t * key2, int32_t * out,
                const int32_t * step, int32_t * rec, int64_t rec_nb1, int32_t * col_out, int n, int top_k, float top_p);

// ---- k_misc.cu (fork getrows.cu, norm.cu) ---------------------------------------------------------------------------
// ilv = false: the table in file order (Weight::ilv -- the server's host table read in place)
void get_rows_ptq1(cudaStream_t st, const void * table, int64_t n_embd, int64_t n_rows, const int32_t * ids, int n, float * dst,
                   bool ilv = true);
// rms_norm_f32<1024, true, false>: dst = rms_norm(x) * w, rows of ncols
void rms_norm_mul(cudaStream_t st, const float * x, const float * w, float * dst, int ncols, int nrows, float eps);

// ---- fork_shims.cu ----------------------------------------------------------------------------------------------------
void fork_runtime_init(cudaStream_t main_stream);   // the device info, context and pool the fork's host helpers expect

} // namespace eng
