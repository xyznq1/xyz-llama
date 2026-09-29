// xyz-engine model loader: GGUF metadata and any unbound weights. PTQ1_0 weights are re-packed in place to the
// fork's ILV16 layout (ggml-cuda.cu k_ptq1_ilv_pack_tile, common.cuh ptq1_ilv_block): every PTQ1 reader in the fork
// addresses through that layout, so the engine must hold the same bytes where the fork holds them.
#include "model.h"

#include <cstdio>
#include <cstring>

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#include "gguf.h"

void DeviceArena::init(size_t bytes) {
    CK(cudaMalloc(&base, bytes));
    size = bytes;
    used = 0;
}

void * DeviceArena::alloc(size_t bytes, size_t align) {
    used = (used + align - 1) / align * align;
    if (used + bytes > size) {
        fprintf(stderr, "arena: out of space (%zu + %zu > %zu)\n", used, bytes, size);
        exit(1);
    }
    void * p = base + used;
    used += bytes;
    return p;
}

// The fork's ILV16 repack uses one CTA per complete 16-row tile staged in shared memory.
static __global__ void k_ptq1_ilv_pack_tile(uint32_t * __restrict__ data, const int S, const int64_t matrix_words) {
    extern __shared__ uint32_t ptq1_ilv_tile[];
    constexpr int wpb = (int) (sizeof(block_ptq1_0) / 4);
    const int nw = 16*S*wpb;
    uint32_t * p = data + blockIdx.y*matrix_words + (int64_t) blockIdx.x*nw;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        ptq1_ilv_tile[i] = p[i];
    }
    __syncthreads();
    for (int o = threadIdx.x; o < nw; o += blockDim.x) {   // o = ILV word (kb*16 + r)*7 + w
        const int kb  = o / (16*wpb);
        const int rem = o - kb*16*wpb;
        const int r   = rem / wpb;
        const int w   = rem - r*wpb;
        p[o] = ptq1_ilv_tile[(r*S + kb)*wpb + w];
    }
}

// a host range the device reads in place: the device alias of pinned + mapped memory (the server's pinned buffer), else
// its pages registered read-only and mapped; nullptr when neither works (the caller copies the tensor instead)
static void * map_host(const void * p, const size_t bytes) {
    cudaPointerAttributes a = {};
    if (cudaPointerGetAttributes(&a, p) == cudaSuccess && a.type == cudaMemoryTypeHost && a.devicePointer != nullptr) {
        return a.devicePointer;
    }
    (void) cudaGetLastError();
    const uintptr_t page = 4096;
    const uintptr_t lo   = (uintptr_t) p & ~(page - 1);
    const uintptr_t hi   = ((uintptr_t) p + bytes + page - 1) & ~(page - 1);
    const cudaError_t e = cudaHostRegister((void *) lo, hi - lo, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (e != cudaSuccess && e != cudaErrorHostMemoryAlreadyRegistered) {
        (void) cudaGetLastError();
        return nullptr;
    }
    (void) cudaGetLastError();
    void * d = nullptr;
    if (cudaHostGetDevicePointer(&d, const_cast<void *>(p), 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return d;
}

static void ptq1_ilv_pack(const Weight & t, cudaStream_t st) {
    const int     S      = (int) (t.ne[0] / QK_PTQ1_0);
    const int64_t ntiles = t.ne[1] / 16;
    const int64_t nmat   = t.ne[2]*t.ne[3];
    if (ntiles == 0) {
        return;
    }
    const int64_t matrix_words = t.ne[1]*S*(int64_t) (sizeof(block_ptq1_0) / 4);
    const size_t  smem = 16*(size_t) S*sizeof(block_ptq1_0);
    if (smem > 48*1024) {
        CK(cudaFuncSetAttribute(k_ptq1_ilv_pack_tile, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    }
    if (smem > 99*1024) {
        fprintf(stderr, "ptq1_ilv_pack: %s needs %zu B of shared memory per tile (the copy path is not ported)\n", t.name.c_str(), smem);
        exit(1);
    }
    k_ptq1_ilv_pack_tile<<<dim3((unsigned) ntiles, (unsigned) nmat, 1), 256, smem, st>>>((uint32_t *) t.data, S, matrix_words);
    CK(cudaGetLastError());
}

const Weight * Model::find(const std::string & name) const {
    auto it = w.find(name);
    return it == w.end() ? nullptr : &it->second;
}

const Weight & Model::get(const std::string & name) const {
    const Weight * p = find(name);
    if (!p) {
        fprintf(stderr, "model: missing tensor %s\n", name.c_str());
        exit(1);
    }
    return *p;
}

static int kv_i(const gguf_context * g, const std::string & key, int def = 0) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(g, id);
        default: return def;
    }
}

static float kv_f(const gguf_context * g, const std::string & key, float def) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id < 0 ? def : gguf_get_val_f32(g, id);
}

bool Model::load(const char * path, cudaStream_t st) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = {/*no_alloc =*/ true, /*ctx =*/ &meta};
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        fprintf(stderr, "model: cannot read %s\n", path);
        return false;
    }
    hp.arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    const std::string a = hp.arch + ".";
    hp.n_layer     = kv_i(g, a + "block_count");
    hp.n_embd      = kv_i(g, a + "embedding_length");
    hp.n_ff        = kv_i(g, a + "feed_forward_length");
    hp.n_head      = kv_i(g, a + "attention.head_count");
    hp.n_head_kv   = kv_i(g, a + "attention.head_count_kv");
    hp.head_dim    = kv_i(g, a + "attention.key_length");
    hp.fa_interval = kv_i(g, a + "full_attention_interval");
    hp.ssm_conv    = kv_i(g, a + "ssm.conv_kernel");
    hp.ssm_state   = kv_i(g, a + "ssm.state_size");
    hp.ssm_groups  = kv_i(g, a + "ssm.group_count");
    hp.ssm_dt_rank = kv_i(g, a + "ssm.time_step_rank");
    hp.ssm_inner   = kv_i(g, a + "ssm.inner_size");
    hp.rms_eps     = kv_f(g, a + "attention.layer_norm_rms_epsilon", 1e-6f);
    hp.rope_base   = kv_f(g, a + "rope.freq_base", 10000.f);
    hp.rope_dims   = kv_i(g, a + "rope.dimension_count");
    {
        const int64_t id = gguf_find_key(g, (a + "rope.dimension_sections").c_str());
        if (id >= 0) {
            const int32_t * s = (const int32_t *) gguf_get_arr_data(g, id);
            for (int i = 0; i < 4 && i < (int) gguf_get_arr_n(g, id); ++i) hp.rope_sections[i] = s[i];
        }
    }
    // the model's rotation keys are PrismML's (prism.hadamard.*); the xyz drafter carries the same data as xyz.hadamard.*
    const std::string had = gguf_find_key(g, "prism.hadamard.block_size") >= 0 ? "prism.hadamard." : "xyz.hadamard.";
    hp.had_block = kv_i(g, had + "block_size");
    {
        const int64_t iw = gguf_find_key(g, (had + "sign_widths").c_str());
        const int64_t iv = gguf_find_key(g, (had + "sign_values").c_str());
        if (iw >= 0 && iv >= 0) {
            const int32_t * wd = (const int32_t *) gguf_get_arr_data(g, iw);
            for (size_t i = 0; i < gguf_get_arr_n(g, iw); ++i) hp.had_widths.push_back(wd[i]);
            const size_t n = gguf_get_arr_n(g, iv);
            const gguf_type ty = gguf_get_arr_type(g, iv);
            const void * d = gguf_get_arr_data(g, iv);
            for (size_t i = 0; i < n; ++i) {
                float v = ty == GGUF_TYPE_INT8 ? ((const int8_t *) d)[i] : ty == GGUF_TYPE_INT32 ? (float) ((const int32_t *) d)[i]
                        : ((const float *) d)[i];
                hp.had_signs.push_back(v);
            }
        }
    }

    // tensors: sizes first (one arena), placement, then a streamed read of the file into it
    const int64_t n_t = gguf_get_n_tensors(g);
    size_t total = 0;
    std::vector<Weight> ws(n_t);
    std::unordered_map<std::string, int64_t> idx;
    for (int64_t i = 0; i < n_t; ++i) {
        ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        Weight & wt = ws[i];
        const void * bp = bind_lookup(t->name);
        void * mapped = nullptr;
        if (bp == nullptr && bind_lookup_host && t->type == GGML_TYPE_PTQ1_0 && strcmp(t->name, "token_embd.weight") == 0) {
            if (const void * hp_ptr = bind_lookup_host(t->name)) {
                mapped = map_host(hp_ptr, ggml_nbytes(t));
            }
        }
        if (bp != nullptr) {
            wt.data  = const_cast<void *>(bp);
            wt.bound = true;
        } else if (mapped != nullptr) {   // 0.26 GiB of VRAM the engine used to copy the table into
            wt.data  = mapped;
            wt.bound = true;
            wt.ilv   = false;
        } else {
            total += (ggml_nbytes(t) + 255) / 256 * 256;
        }
        wt.name = t->name;
        wt.type = t->type;
        for (int d = 0; d < 4; ++d) wt.ne[d] = t->ne[d];
        wt.nbytes = ggml_nbytes(t);
        idx[wt.name] = i;
    }
    size_t signs_bytes = hp.had_signs.size()*sizeof(float) + 256*hp.had_widths.size();
    arena.init(total + signs_bytes + (1 << 20));
    // Placement: the fork launches sibling matmuls as ONE matrix ([attn_qkv; attn_gate], [attn_q; attn_k; attn_v],
    // [ffn_gate; ffn_up]) because llama places them back to back in creation order; the engine places them the same way,
    // byte-adjacent (each member's size is a whole number of 256-byte units: 16-row tiles of 5120-wide PTQ1_0 rows).
    static const std::vector<std::vector<std::string>> groups = {
        { "attn_qkv.weight", "attn_gate.weight" }, { "attn_q.weight", "attn_k.weight", "attn_v.weight" },
        { "ffn_gate.weight", "ffn_up.weight" } };
    for (int64_t i = 0; i < n_t; ++i) {
        if (ws[i].data != nullptr) {
            continue;
        }
        const std::string & nm = ws[i].name;
        const size_t dot = nm.find('.', 4);
        const std::string prefix = nm.compare(0, 4, "blk.") == 0 && dot != std::string::npos ? nm.substr(0, dot + 1) : "";
        const std::string role   = prefix.empty() ? nm : nm.substr(prefix.size());
        const std::vector<std::string> * grp = nullptr;
        for (const auto & gr : groups) {
            for (const auto & r : gr) {
                if (r == role) grp = &gr;
            }
        }
        if (grp == nullptr || prefix.empty()) {
            ws[i].data = arena.alloc(ws[i].nbytes);
            continue;
        }
        for (size_t k = 0; k < grp->size(); ++k) {
            const auto it = idx.find(prefix + (*grp)[k]);
            if (it == idx.end()) {
                fprintf(stderr, "model: %s has no sibling %s\n", nm.c_str(), (*grp)[k].c_str());
                return false;
            }
            Weight & wk = ws[it->second];
            if (k > 0 && arena.used % 256 != 0) {
                fprintf(stderr, "model: %s cannot follow its sibling byte-adjacent (%zu bytes)\n", wk.name.c_str(), arena.used);
                return false;
            }
            wk.data = arena.alloc(wk.nbytes);
        }
    }
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "model: cannot open %s\n", path);
        return false;
    }
    const size_t data_off = gguf_get_data_offset(g);
    const size_t chunk = 64u << 20;
    void * pinned = nullptr;
    CK(cudaMallocHost(&pinned, 2*chunk));
    int buf = 0;
    cudaEvent_t done[2];
    CK(cudaEventCreate(&done[0])); CK(cudaEventCreate(&done[1]));
    CK(cudaEventRecord(done[0], st)); CK(cudaEventRecord(done[1], st));
    size_t n_ptq1 = 0;
    for (int64_t i = 0; i < n_t; ++i) {
        const Weight & wt = ws[i];
        const char * name = wt.name.c_str();
        if (wt.data == nullptr) {
            continue;
        }
        if (wt.bound) {
            w[wt.name] = wt;
            continue;   // the server's own copy (bind mode)
        }
        _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
        for (size_t o = 0; o < wt.nbytes; o += chunk) {
            const size_t n = std::min(chunk, wt.nbytes - o);
            CK(cudaEventSynchronize(done[buf]));              // this half of the staging buffer is free again
            char * h = (char *) pinned + buf*chunk;
            if (fread(h, 1, n, f) != n) {
                fprintf(stderr, "model: short read in %s\n", name);
                return false;
            }
            CK(cudaMemcpyAsync((char *) wt.data + o, h, n, cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(done[buf], st));
            buf ^= 1;
        }
        if (wt.type == GGML_TYPE_PTQ1_0) {
            ptq1_ilv_pack(wt, st);                            // queued behind the upload that completed it
            n_ptq1++;
        }
        w[wt.name] = wt;
    }
    fclose(f);
    // the Hadamard sign vectors, one per width
    size_t off = 0;
    for (int wd : hp.had_widths) {
        float * d = (float *) arena.alloc(wd*sizeof(float));
        CK(cudaMemcpyAsync(d, hp.had_signs.data() + off, wd*sizeof(float), cudaMemcpyHostToDevice, st));
        had_signs_dev.push_back(d);
        off += wd;
    }
    CK(cudaStreamSynchronize(st));
    CK(cudaFreeHost(pinned));
    hp.n_vocab = (int) get("output.weight").ne[1];
    gguf_free(g);
    ggml_free(meta);
    fprintf(stderr, "model: %s | %lld tensors, %.2f GiB on the device (%zu PTQ1_0 re-packed ILV16)\n", hp.arch.c_str(),
            (long long) n_t, arena.used / 1073741824.0, n_ptq1);
    return true;
}
