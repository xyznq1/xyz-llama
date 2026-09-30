// The host runtime the fork's launchers expect (definitions that live in the fork's ggml-cuda.cu, which the engine does not
// compile): device info, current device, error reporting, and the context's memory pool. The engine owns all of it.
#include "common.cuh"

#include "kernels.h"

#include <cstdio>
#include <cstdlib>

static ggml_cuda_device_info g_info = {};
static bool                  g_info_ok = false;

// ggml-cuda.cu ggml_cuda_init, the CUDA branch (the fields the kernels' launchers read)
const ggml_cuda_device_info & ggml_cuda_info() {
    if (!g_info_ok) {
        g_info_ok = true;
        CUDA_CHECK(cudaGetDeviceCount(&g_info.physical_device_count));
        g_info.device_count = 1;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        auto & d = g_info.devices[0];
        d.physical_device      = 0;
        d.virtual_index        = 0;
        d.physical_share_count = 1;
        d.integrated           = false;
        d.vmm                  = false;
        d.nsm                  = prop.multiProcessorCount;
        d.smpb                 = prop.sharedMemPerBlock;
        d.warp_size            = prop.warpSize;
        int coop = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
        d.supports_cooperative_launch = coop != 0;
        d.smpbo      = prop.sharedMemPerBlockOptin;
        d.cc         = 100*prop.major + 10*prop.minor;
        d.total_vram = prop.totalGlobalMem;
    }
    return g_info;
}

void ggml_cuda_set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    return 0;
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  in %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    abort();
}

// The context's pool: one device buffer, allocations stacked and released in reverse order -- every pass requests the
// same sequence, so every pass gets the same addresses (what a captured CUDA graph needs).
struct engine_pool : public ggml_cuda_pool {
    char * base = nullptr;
    size_t size = 0, top = 0;
    static constexpr int MAXN = 64;
    size_t marks[MAXN];
    int    n = 0;
    explicit engine_pool(size_t bytes) : size(bytes) { CUDA_CHECK(cudaMalloc(&base, bytes)); }
    void * alloc(size_t want, size_t * actual) override {
        const size_t sz = (want + 255) / 256 * 256;
        if (top + sz > size || n == MAXN) {
            fprintf(stderr, "engine pool: out of space (%zu + %zu > %zu, %d live)\n", top, sz, size, n);
            abort();
        }
        marks[n++] = top;
        void * p = base + top;
        top += sz;
        *actual = sz;
        return p;
    }
    void free(void * ptr, size_t) override {
        if (n == 0 || base + marks[n - 1] != (char *) ptr) {
            fprintf(stderr, "engine pool: out-of-order free\n");
            abort();
        }
        top = marks[--n];
    }
};

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int stream_no) {
    GGML_UNUSED(device);
    GGML_UNUSED(stream_no);
    return std::unique_ptr<ggml_cuda_pool>(new engine_pool(64u << 20));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

// Host entry points the compiled fork files REFERENCE on paths the engine never takes (activation quantizing when no q8
// twin exists, f16 KV materialisation, sparse attention). Reaching one is a wiring bug, so each aborts by name.
#include "convert.cuh"

[[noreturn]] static void not_wired(const char * what) {
    fprintf(stderr, "xyz-engine: %s reached -- not a path of the verify pass\n", what);
    abort();
}
void ggml_cuda_flash_attn_ext_compact_mask(const ggml_tensor *, int32_t *, int32_t, cudaStream_t) {
    not_wired("ggml_cuda_flash_attn_ext_compact_mask");
}
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type) {
    not_wired("ggml_get_to_fp16_cuda");
}
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type) {
    not_wired("ggml_get_to_fp16_nc_cuda");
}

namespace eng {
ggml_backend_cuda_context * g_ctx = nullptr;   // used by k_fattn.cu

void fork_runtime_init(cudaStream_t main_stream) {
    ggml_cuda_info();
    if (g_ctx == nullptr) {
        g_ctx = new ggml_backend_cuda_context(0);
    }
    g_ctx->streams[0][0] = main_stream;
    g_ctx->curr_stream_no = 0;
}
} // namespace eng
