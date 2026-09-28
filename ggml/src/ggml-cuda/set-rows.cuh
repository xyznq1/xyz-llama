#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The attention cache writes use one launch each because the xyzkv2 store reads
// this file's InnerQ state). K: RMS_NORM -> MUL -> ROPE(M-RoPE) -> Hadamard-256 (MUL_MAT hint) -> SET_ROWS(xyzkv2);
// V: its source -> Hadamard-64 (MUL_MAT hint) -> SET_ROWS(xyzkv2). false = shapes not handled (nothing launched).
bool ggml_cuda_op_attn_k_write(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                               const ggml_tensor * rope, const ggml_tensor * had, ggml_tensor * set_rows);
bool ggml_cuda_op_attn_v_write(ggml_backend_cuda_context & ctx, const ggml_tensor * v, const ggml_tensor * had,
                               ggml_tensor * set_rows);
