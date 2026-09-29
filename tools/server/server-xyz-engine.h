#pragma once
// xyz-engine: the device-side speculative decode loop, loaded at run time from xyz_engine.dll when XYZ_ENGINE=1 (Windows).
// It runs a response's rounds (seed decode, draft chain, verify, fold, accept) on the server's own weights and memory, one
// host sync per round, byte-identical to the server's own path; the server keeps prefill, prompt caching, streaming and
// every stop rule. The types mirror include/xyz_engine.h of the engine's repository.
#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

extern "C" {

typedef struct xe_ctx xe_ctx;
typedef const void * (*xe_tensor_fn)(void * user, int model, const char * name);

struct xe_bind {
    const char * target_gguf;
    const char * drafter_gguf;
    const char * expf_table;
    xe_tensor_fn tensor;
    void *       user;
    int32_t kv_size;
    void *  k[128];
    void *  v[128];
    float * conv[128];
    float * ssm[128];
    float * pk[128];
    float * px[128];
    int32_t d_kv_size;
    void *  dk;
    void *  dv;
    int32_t d_n_swa;
};

#define XE_SEED_MAX 72

struct xe_start {
    int32_t  p0;
    int32_t  id_last;
    int32_t  pending;
    int32_t  m;
    int32_t  seed_tok[XE_SEED_MAX];
    int32_t  seed_pos0;
    const float * seed_g;
    uint32_t cseed;
    uint32_t rng[624];
    const int32_t * d_pos;
    int32_t  d_n;
    int32_t  d_head;
    int32_t  n_draws;
    int32_t  accept_mode;
    const int32_t * skip;
    int32_t  n_skip;
    int32_t  budget_tail;
    int32_t  chain_steps;
    const int32_t * cut;    // a waiting lazy grammar's trigger ids: block verification cuts each draft before the first
    int32_t  n_cut;         // (<= 16; 0: no cut)
};

#define XE_RF_TIE   1u
#define XE_RF_SHORT 2u
#define XE_RF_CUT   4u
typedef int (*xe_round_fn)(void * user, int32_t * toks, int32_t * n, uint32_t one_mask, const int32_t * drafts,
                           int32_t n_drafts, uint32_t flags);

struct xe_end {
    int32_t  rounds;
    int32_t  n_tokens;
    int32_t  p0, id_last, pending;
    int32_t  m;
    int32_t  seed_tok[8];
    int32_t  seed_pos0;
    uint64_t rng_draws;
    int32_t  short_rows;
    int32_t  tie_rows;
    int32_t  cut_rounds;
    int32_t  captures;
};

// the prompt: one server batch (xyz_engine.h xe_prefill)
struct xe_prefill_in {
    const int32_t * tokens;
    int32_t n, p0;
    int32_t n_ubatch;
    int32_t n_ubatch_dft;
    int32_t want_logits;
    const int32_t * d_pos;
    int32_t d_n, d_head;
    const int32_t * stash_ids;
    const int32_t * stash_pos;
    const float *   stash_g;
    int32_t n_stash;
    int32_t pending_pos;
    const float * pending_g;
    int32_t merge_max;
    int32_t pending;
};
struct xe_prefill_out {
    float *   logits;
    float *   g_last;
    float *   verify_g;
    int32_t * stash_ids;
    int32_t * stash_pos;
    float *   stash_g;
    int32_t   n_stash;
    int32_t   decoded_rows;
};

} // extern "C"

struct xyz_engine_dll {
    void * h = nullptr;
    std::string dir;   // the directory the DLL was loaded from (its expf table sits next to it)
    xe_ctx * (*create)(const xe_bind *) = nullptr;
    void     (*destroy)(xe_ctx *) = nullptr;
    void     (*rebind_rs)(xe_ctx *, float * const *, float * const *, float * const *, float * const *) = nullptr;
    int      (*generate)(xe_ctx *, const xe_start *, int32_t, xe_round_fn, void *, xe_end *) = nullptr;
    int      (*seed_rows)(xe_ctx *, float *, int32_t) = nullptr;
    void     (*flush_pack)(xe_ctx *) = nullptr;
    int      (*ring)(xe_ctx *, int32_t *, int32_t, int32_t *) = nullptr;
    void     (*logits_row)(xe_ctx *, int32_t, float *) = nullptr;
    int      (*prefill)(xe_ctx *, const xe_prefill_in *, xe_prefill_out *) = nullptr;   // optional (an older DLL: null)

    bool load(const std::string & path, std::string & err) {
#ifdef _WIN32
        HMODULE m = LoadLibraryA(path.c_str());
        if (m == nullptr) {
            err = "LoadLibrary(" + path + ") failed, error " + std::to_string((unsigned long) GetLastError());
            return false;
        }
        h = (void *) m;
        char buf[MAX_PATH] = {};
        if (GetModuleFileNameA(m, buf, MAX_PATH) > 0) {
            dir = buf;
            dir.erase(dir.find_last_of("\\/") + 1);
        }
        const auto sym = [&](const char * name) -> FARPROC {
            FARPROC p = GetProcAddress(m, name);
            if (p == nullptr && err.empty()) {
                err = std::string("missing symbol ") + name;
            }
            return p;
        };
        create     = (decltype(create))     sym("xe_create");
        destroy    = (decltype(destroy))    sym("xe_free");
        rebind_rs  = (decltype(rebind_rs))  sym("xe_rebind_rs");
        generate   = (decltype(generate))   sym("xe_generate");
        seed_rows  = (decltype(seed_rows))  sym("xe_seed_rows");
        flush_pack = (decltype(flush_pack)) sym("xe_flush_pack");
        ring       = (decltype(ring))       sym("xe_ring");
        logits_row = (decltype(logits_row)) sym("xe_logits_row");
        prefill    = (decltype(prefill))    GetProcAddress(m, "xe_prefill");
        return err.empty();
#else
        err = "xyz-engine: Windows only (" + path + ")";
        return false;
#endif
    }
};
