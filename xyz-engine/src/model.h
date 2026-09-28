#pragma once
// xyz-engine model: GGUF metadata and weights laid out exactly as the fork's CUDA
// readers expect (PTQ1_0 re-packed to ILV16 after upload, the other types verbatim).
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct DeviceArena {
    char * base = nullptr;
    size_t size = 0, used = 0;
    void   init(size_t bytes);
    void * alloc(size_t bytes, size_t align = 256);
};

struct Weight {
    std::string name;
    ggml_type   type = GGML_TYPE_F32;
    int64_t     ne[4] = {1, 1, 1, 1};
    size_t      nbytes = 0;
    void      * data = nullptr;   // device
    bool        bound = false;     // data is the host process's own tensor (bind mode), not the engine's
};

struct Hparams {
    std::string arch;
    int   n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_vocab = 0;
    int   fa_interval = 0;                                     // every fa_interval-th layer is full attention
    int   ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0, ssm_inner = 0;
    float rms_eps = 1e-6f, rope_base = 10000.f;
    int   rope_dims = 0, rope_sections[4] = {0, 0, 0, 0};
    int   had_block = 0;                                       // Hadamard rotation (prism.hadamard.*)
    std::vector<int>   had_widths;                             // sign vector widths, concatenated in had_signs
    std::vector<float> had_signs;
    bool  is_attn(int il) const { return fa_interval > 0 && (il + 1) % fa_interval == 0; }
};

struct Model {
    Hparams hp;
    std::unordered_map<std::string, Weight> w;
    DeviceArena arena;
    std::vector<float *> had_signs_dev;                        // one device sign vector per width (had_widths order)
    // Every tensor the callback returns is the server's device copy; tensors kept on the host are loaded from the file.
    std::function<const void *(const char * name)> bind_lookup;
    bool load(const char * path, cudaStream_t st);
    const Weight & get(const std::string & name) const;
    const Weight * find(const std::string & name) const;
};
