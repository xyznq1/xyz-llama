#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// Hoisted IQ2/IQ3 decoders support up to 16 columns. Other types keep the limit above.
#define MMVQ_MAX_BATCH_SIZE_WIDE 16

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    const ggml_tensor * dst_hi = nullptr, const ggml_tensor * dst_hi2 = nullptr);

// Two MUL_MATs a = A*x, b = B*x whose PTQ1_0 weights sit back to back ([A; B]) as ONE launch writing both outputs.
// Bit-identical to the two launches. Returns false, launching nothing, when the pair does not qualify.
bool ggml_cuda_mul_mat_vec_q_ptq1_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);

// The pair test alone (no launch; valid before allocation too), for graph_optimize's fusion plan.
bool ggml_cuda_ptq1_pair_mergeable(const ggml_tensor * a, const ggml_tensor * b);

// Three sibling matmuls [A; B; C] share one launch when their layouts match.
// create_tensor_qkv, adjacent nodes since qwen35.cpp expands them in a row) as ONE launch writing all three outputs.
bool ggml_cuda_ptq1_triple_mergeable(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c);
bool ggml_cuda_mul_mat_vec_q_ptq1_triple(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * c);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
