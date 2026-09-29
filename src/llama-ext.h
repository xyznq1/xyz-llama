#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// Run a draft model's feature fusion in the target graph (the xyz drafter's fc on the target's features). The getter
// returns NULL when the current decode did not compute a complete fold, so callers can use their encoder path.
LLAMA_API bool    llama_set_fc_fold(struct llama_context * ctx_tgt, const struct llama_model * model_dft,
                                    const int32_t * layers, int32_t n_layers);
LLAMA_API float * llama_get_fc_fold(struct llama_context * ctx_tgt);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// ---- xyz-engine binding: the device tensors an external decode engine runs on ------------------------
// the model's tensor by GGUF name (nullptr if absent)
LLAMA_API const struct ggml_tensor * llama_model_tensor_ext(const struct llama_model * model, const char * name);
// a context's memory, per model layer (nullptr where a layer has none): stream 0 K/V of its KV cache (for an iswa memory,
// the SWA cache's) and the recurrent memory's conv state r, GDN state s, GDN pack pk, conv pack px
struct llama_engine_mem {
    int32_t n_layer;
    int32_t kv_size;
    const struct ggml_tensor * k[128];
    const struct ggml_tensor * v[128];
    int32_t rs_size;
    int32_t rs_row;          // the recurrent row holding sequence 0 (its tail cell), -1 if none
    int32_t pack_tokens;
    const struct ggml_tensor * r[128];
    const struct ggml_tensor * s[128];
    const struct ggml_tensor * pk[128];
    const struct ggml_tensor * px[128];
};
LLAMA_API bool llama_engine_mem_get(struct llama_context * ctx, struct llama_engine_mem * out);
// sequence 0's cell positions in the KV cache (iswa: the SWA cache): out[i] = position of cell i, -1 empty; returns the
// number of cells up to the last used one (<= n written)
LLAMA_API int32_t llama_engine_kv_cells(struct llama_context * ctx, int32_t * out, int32_t n);
// ---- xyz-engine hand-over: the memory metadata an external engine's rounds change, read at the takeover
// and written back as the server's own path would have left it (the tensors are the context's own, bound by pointer)
// sequence 0's full cell table (pos[i]: position of cell i, -1 empty; n >= the cache size) and head; returns the size
LLAMA_API int32_t llama_engine_kv_table(struct llama_context * ctx, int32_t * pos, int32_t n, int32_t * head);
// sequence 0's cells := pos[0..n) (-1 empty; cells past n empty), head := head (the drafter's SWA cache)
LLAMA_API bool    llama_engine_kv_set_table(struct llama_context * ctx, const int32_t * pos, int32_t n, int32_t head);
// the target's KV after the engine's verifies: cells [p0, p1) hold positions p0..p1-1 (cell = position, cells below
// p0 already do), nothing of sequence 0 at or past p1; head := p1. false if a cell in the range holds anything else
LLAMA_API bool    llama_engine_kv_commit(struct llama_context * ctx, int32_t p0, int32_t p1);
// the target's recurrent cell of sequence 0 (rs_replay): its position and committed position -- pos - pos_commit rows of
// the pack are pending, replayed by the next decode
LLAMA_API bool    llama_engine_rs_get(struct llama_context * ctx, int32_t * pos, int32_t * pos_commit);
LLAMA_API bool    llama_engine_rs_set(struct llama_context * ctx, int32_t pos, int32_t pos_commit);
// a fresh sequence 0 gets the recurrent cell find_slot would give it (true: it has one)
LLAMA_API bool    llama_engine_rs_claim(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// true for a grammar sampler that still waits for its lazy trigger (no constraint applies yet); false otherwise
LLAMA_API bool llama_sampler_grammar_awaiting(const struct llama_sampler * smpl);
