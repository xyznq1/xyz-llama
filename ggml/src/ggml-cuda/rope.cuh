#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// Fuse RMS_NORM + MUL + ROPE(M-RoPE) + Hadamard-256 + XYZKV_WHT(fwd, 128) in one launch,
// writing the XYZKV_WHT output. false = shapes/modes not handled (nothing launched).
bool ggml_cuda_op_attn_q_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                               const ggml_tensor * rope, const ggml_tensor * had, ggml_tensor * xyzkv);
