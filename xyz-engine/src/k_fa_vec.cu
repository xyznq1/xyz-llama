// One-row attention before Ada (cc < 890): over a quantized cache the server takes the vector kernel there
// (fattn.cu ggml_cuda_get_best_fattn_kernel: Turing tensor cores, cc < Ada, quantized K/V, Q->ne[1] == 1 and
// K->ne[1] % FATTN_KQ_STRIDE == 0), where Ada takes the MMA tile the engine's own attention (k_fa.cu) implements. This
// runs that kernel through the fork's own launcher -- launch_fattn's occupancy-sized parallel blocks and combine -- on
// tensor descriptors of the engine's buffers. The kernel reads the mask over all n_kv cells, so every cell must be written.
// A separate file: fattn-common.cuh and fa/fa.cuh do not compile together.
#include "common.cuh"
#include "fattn-vec.cuh"

#include "kernels.h"

namespace eng {

extern ggml_backend_cuda_context * g_ctx;   // fork_shims.cu

bool fa_vec_rule(const int n_tokens, const int n_kv) {
    static const int cc = ggml_cuda_info().devices[0].cc;
    return cc < GGML_CUDA_CC_ADA_LOVELACE && n_tokens == 1 && n_kv % FATTN_KQ_STRIDE == 0;
}

namespace {

constexpr int D         = 256;
constexpr int N_HEAD_KV = 4;

// q [256, n_head] f32 (one token), K/V [1024 per cell, kv_size cells] of `type`, mask f16 [n_kv], dst [256, n_head] f32 --
// the views build_attn_mha hands ggml_flash_attn_ext (Q and the caches permuted to [D, tokens | cells, heads])
template <ggml_type type>
void vec_launch(cudaStream_t st, const float * q, const int n_head, const void * k, const void * v, const int n_kv,
                const int kv_size, const half * mask, const float scale, float * dst) {
    const size_t ts = ggml_type_size(type);
    const size_t rb = ggml_row_size(type, D);   // one head of a cell

    ggml_tensor tq = {}, tk = {}, tv = {}, tm = {}, td = {};
    tq.type = GGML_TYPE_F32;
    tq.data = (void *) q;
    tq.ne[0] = D;   tq.ne[1] = 1;                       tq.ne[2] = n_head;           tq.ne[3] = 1;
    tq.nb[0] = 4;   tq.nb[1] = (size_t) D*n_head*4;     tq.nb[2] = (size_t) D*4;     tq.nb[3] = (size_t) D*n_head*4;

    for (ggml_tensor * t : { &tk, &tv }) {
        t->type = type;
        t->ne[0] = D;   t->ne[1] = n_kv;               t->ne[2] = N_HEAD_KV;   t->ne[3] = 1;
        t->nb[0] = ts;  t->nb[1] = rb*N_HEAD_KV;       t->nb[2] = rb;          t->nb[3] = rb*N_HEAD_KV*kv_size;
    }
    tk.data = (void *) k;
    tv.data = (void *) v;

    tm.type = GGML_TYPE_F16;
    tm.data = (void *) mask;
    tm.ne[0] = n_kv;   tm.ne[1] = 1;                     tm.ne[2] = 1;                  tm.ne[3] = 1;
    tm.nb[0] = 2;      tm.nb[1] = (size_t) n_kv*2;       tm.nb[2] = (size_t) n_kv*2;    tm.nb[3] = (size_t) n_kv*2;

    td.type = GGML_TYPE_F32;
    td.op   = GGML_OP_FLASH_ATTN_EXT;
    td.data = dst;
    td.ne[0] = D;   td.ne[1] = n_head;   td.ne[2] = 1;   td.ne[3] = 1;
    td.nb[0] = 4;   td.nb[1] = (size_t) D*4;   td.nb[2] = (size_t) D*n_head*4;   td.nb[3] = (size_t) D*n_head*4;
    const float params[3] = { scale, 0.0f, 0.0f };   // scale, max_bias, logit_softcap
    memcpy(td.op_params, params, sizeof(params));
    td.src[0] = &tq; td.src[1] = &tk; td.src[2] = &tv; td.src[3] = &tm;

    // ggml_cuda_flash_attn_ext_vec_case at Q->ne[1] == 1 without a softcap: the one-column instance
    cudaStream_t & s0 = g_ctx->streams[0][0];
    const cudaStream_t prev = s0;
    s0 = st;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, 1, type, type, false>(*g_ctx, &td);
    s0 = prev;
}

} // namespace

void flash_attn_vec(cudaStream_t st, const float * q, int n_head, const void * k_cache, const void * v_cache, int n_kv,
                    int kv_size, const half * mask, float scale, float * dst) {
    vec_launch<GGML_TYPE_XYZKV2_0>(st, q, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
}

void flash_attn_vec_q4_0(cudaStream_t st, const float * q, int n_head, const void * k_cache, const void * v_cache, int n_kv,
                         int kv_size, const half * mask, float scale, float * dst) {
    vec_launch<GGML_TYPE_Q4_0>(st, q, n_head, k_cache, v_cache, n_kv, kv_size, mask, scale, dst);
}

} // namespace eng
