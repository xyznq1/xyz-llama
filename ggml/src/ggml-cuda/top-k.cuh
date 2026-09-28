#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_DRAFT_SAMPLE runs the speculative drafter's coupled draw for one row on the GPU.
void ggml_cuda_op_draft_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
