// Native xyzkv2 MMA flash-attention instances for the Qwen3.5 hybrid shape.
//
// DKQ = DV = 256 (n_embd_k_gqa 1024 over 4 KV heads), and gqa_ratio 24/4 = 6 selects ncols2 = 8.
// ncols1 then follows Q->ne[1]: 1 for plain decode, up to 8 for a width-5 speculative verify.
//
// These read K and V directly from the xyzkv2 cache.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_XYZKV_CASE(256, 256, 1, 8, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0);
DECL_FATTN_MMA_XYZKV_CASE(256, 256, 2, 8, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0);
DECL_FATTN_MMA_XYZKV_CASE(256, 256, 4, 8, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0);
DECL_FATTN_MMA_XYZKV_CASE(256, 256, 8, 8, GGML_TYPE_XYZKV2_0, GGML_TYPE_XYZKV2_0);
