#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// Signed variant, for weights stored in a ROTATED BASIS (PrismML's Ternary Bonsai 2: the GGUF's
// prism.hadamard.* keys declare a normalized Sylvester-Walsh-Hadamard over the input dimension in
// 1024-wide blocks with an explicit +/-1 sign vector). The sign is applied to the INPUT, before the
// butterfly, folded into the same load as the 1/sqrt(n) scale -- a separate elementwise pass would
// cost another full read/write of the activation for nothing.
//
// Block width comes from dst->ne[0], i.e. from a RESHAPE in the graph rather than a parameter, so a
// 5120-wide activation is transformed as five independent 1024-blocks. `signs` must be contiguous F32
// with ne[0] a whole multiple of that width; block b of a row uses signs[b*N .. b*N+N).
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst, void * q8_out = nullptr);

// The same with the FFN's SWIGLU folded into the load: glu_src is the [2*nc, tokens] merged gate+up output and the
// transformed row is silu(first half) * second half, bit-identical to the unary_gated_op_kernel launch it replaces.
// 1024-wide blocks only. Returns false when the shapes do not fit.
bool ggml_cuda_op_fwht_signed_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu_src,
                                  const ggml_tensor * signs, ggml_tensor * dst, void * q8_out = nullptr);

bool ggml_cuda_op_gdn_out_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                               const ggml_tensor * mul_tensor, const ggml_tensor * silu_tensor,
                               const ggml_tensor * signs, ggml_tensor * dst, void * q8_out = nullptr,
                               bool permuted = true);   // false: heads are already in order

// [ADD ->] RMS_NORM -> MUL(w) -> MUL(signs) -> 1024-block Hadamard in one launch, bit-identical to the three kernels.
// add: the residual ADD feeding the norm (written too), or null; norm_out: the MUL(w) result when something else also
// reads it, or null; dst: the Hadamard MUL_MAT's output ([1024, rows]). Returns false when the shapes do not fit.
bool ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * norm,
                                const ggml_tensor * w, const ggml_tensor * signs, ggml_tensor * norm_out, ggml_tensor * dst,
                                void * q8_out = nullptr);

// Fuse XYZKV_WHT(inverse, 128) -> Hadamard-64 -> sigmoid(gate) * . -> MUL(signs) ->
// 1024-block Hadamard (MUL_MAT hint) in one launch, writing the last Hadamard's output (dst) and its q8_1 twin. gate: the
// [256, 24, tokens] view of Qcur_full the graph CONTs. false = shapes not handled (nothing launched).
bool ggml_cuda_op_attn_out_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * xyzkv, const ggml_tensor * had,
                                 const ggml_tensor * gate, const ggml_tensor * signs, ggml_tensor * dst, void * q8_out);
