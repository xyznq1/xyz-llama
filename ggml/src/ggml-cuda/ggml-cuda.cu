#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/xyzkv-wht.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <set>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
#if defined(GGML_USE_HIP)
        info.devices[id].integrated = prop.integrated;
#else
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (ptq1_q8_twin & tw : ptq1_q8_twins) {
        if (tw.q8 != nullptr) {
            CUDA_CHECK(cudaFree(tw.q8));
        }
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_cuda_ptq1_ilv_fresh(const ggml_tensor * t);
static void ggml_cuda_ptq1_ilv_forget(const void * begin, size_t size);

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    ggml_cuda_ptq1_ilv_forget(ctx->dev_ptr, buffer->size);
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    if (tensor->type == GGML_TYPE_PTQ1_0) {
        ggml_cuda_ptq1_ilv_fresh(tensor);
    }
    return GGML_STATUS_SUCCESS;
}

// ILV16 upload / download (the layout: common.cuh). The device copy of a PTQ1_0 tensor is PACKED once all its bytes have
// arrived: a freshly allocated tensor (init_tensor) collects uploads -- the model loader streams big ones in chunks --
// and the in-place repack is queued behind the write that completes it, on the same stream. After that, a write covering
// the whole tensor is uploaded and packed again, and any other write (a byte range, a view's bytes) is applied to the
// row-major image -- download, unpack, patch, upload, pack -- so bytes mean what they mean on every other backend.
// Downloads restore row-major on the host. State per base-tensor device pointer, dropped with its buffer.
static constexpr size_t                         GGML_CUDA_PTQ1_ILV_PACKED = SIZE_MAX;
static std::mutex                               ggml_cuda_ptq1_ilv_mutex;
static std::unordered_map<const void *, size_t> ggml_cuda_ptq1_ilv_state; // bytes received while fresh, or _PACKED

// in place, one CTA per whole 16-row tile of one matrix: the tile is the same 16*S blocks in both layouts, staged whole
// in shared memory, written back as [kb][16 rows]
static __global__ void k_ptq1_ilv_pack_tile(uint32_t * __restrict__ data, const int S, const int64_t matrix_words) {
    extern __shared__ uint32_t ptq1_ilv_tile[];
    constexpr int wpb = (int) (sizeof(block_ptq1_0) / 4);
    const int nw = 16*S*wpb;
    uint32_t * p = data + blockIdx.y*matrix_words + (int64_t) blockIdx.x*nw;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        ptq1_ilv_tile[i] = p[i];
    }
    __syncthreads();
    for (int o = threadIdx.x; o < nw; o += blockDim.x) { // o = ILV word (kb*16 + r)*7 + w
        const int kb  = o / (16*wpb);
        const int rem = o - kb*16*wpb;
        const int r   = rem / wpb;
        const int w   = rem - r*wpb;
        p[o] = ptq1_ilv_tile[(r*S + kb)*wpb + w];
    }
}

// the same permutation out of place (src = a row-major copy), for tiles larger than shared memory
static __global__ void k_ptq1_ilv_pack_copy(const uint32_t * __restrict__ src, uint32_t * __restrict__ dst, const int S,
                                            const int64_t matrix_words, const int64_t packed_words, const int64_t n) {
    constexpr int wpb = (int) (sizeof(block_ptq1_0) / 4);
    const int64_t tile_words = 16*(int64_t) S*wpb;
    for (int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t m   = i / packed_words;                  // matrix
        const int64_t rm  = i - m*packed_words;
        const int64_t t   = rm / tile_words;                   // whole tile
        const int64_t o   = rm - t*tile_words;                 // ILV word in the tile
        const int64_t kb  = o / (16*wpb);
        const int64_t rem = o - kb*16*wpb;
        const int64_t r   = rem / wpb;
        const int64_t w   = rem - r*wpb;
        const int64_t base = m*matrix_words + t*tile_words;
        dst[base + o] = src[base + (r*S + kb)*wpb + w];
    }
}

static void ggml_cuda_ptq1_ilv_pack(const ggml_tensor * t, const int device, cudaStream_t stream) {
    GGML_ASSERT(ggml_is_contiguous(t));
    const int     S      = (int) (t->ne[0] / QK_PTQ1_0);
    const int64_t ntiles = t->ne[1] / 16;
    const int64_t nmat   = t->ne[2]*t->ne[3];
    if (ntiles == 0 || nmat == 0) {
        return;
    }
    const int64_t matrix_words = t->ne[1]*S*(int64_t) (sizeof(block_ptq1_0) / 4);
    const size_t  smem         = 16*(size_t) S*sizeof(block_ptq1_0);
    if (smem <= ggml_cuda_info().devices[device].smpbo && nmat <= 65535) {
        if (smem > 48*1024) {
            CUDA_CHECK(cudaFuncSetAttribute(k_ptq1_ilv_pack_tile, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        }
        k_ptq1_ilv_pack_tile<<<dim3((unsigned) ntiles, (unsigned) nmat, 1), 256, smem, stream>>>((uint32_t *) t->data, S, matrix_words);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // ponytail: a whole-tensor temporary, only for K > ~29k or > 65535 matrices (none in our models) -- chunk it if one comes
    const size_t nbytes = ggml_nbytes(t);
    void * tmp = nullptr;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMalloc(&tmp, nbytes));
    CUDA_CHECK(cudaMemcpyAsync(tmp, t->data, nbytes, cudaMemcpyDeviceToDevice, stream));
    const int64_t packed_words = ntiles*16*S*(int64_t) (sizeof(block_ptq1_0) / 4);
    const int64_t n            = nmat*packed_words;
    const int     nblk         = (int) std::min<int64_t>((n + 255) / 256, 65535);
    k_ptq1_ilv_pack_copy<<<nblk, 256, 0, stream>>>((const uint32_t *) tmp, (uint32_t *) t->data, S, matrix_words, packed_words, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
}

// The readers take a PTQ1_0 operand as whole matrices of its base tensor: rows nb[1] apart and never permuted with the
// matrices, matrices a whole number of rows apart, and a view must keep the base's rows and start on a matrix of it.
static bool ggml_cuda_ptq1_ilv_operand_ok(const ggml_tensor * t) {
    const size_t rows_bytes = t->ne[1]*t->nb[1];
    for (int d = 2; d < 4; ++d) {
        if (t->ne[d] > 1 && (t->nb[d] < rows_bytes || t->nb[d] % t->nb[1] != 0)) {
            return false;
        }
    }
    const ggml_tensor * b = t->view_src;
    return b == nullptr || (b->nb[1] == t->nb[1] && b->ne[1] == t->ne[1] && t->view_offs % (b->ne[1]*b->nb[1]) == 0);
}

// row-major again, on the host copy of a packed tensor
static void ggml_cuda_ptq1_ilv_unpack_host(const ggml_tensor * t, uint8_t * buf) {
    const int64_t S  = t->ne[0] / QK_PTQ1_0;
    const size_t  bs = sizeof(block_ptq1_0);
    const int64_t ntiles = t->ne[1] / 16;
    std::vector<uint8_t> tile(16*S*bs);
    for (int64_t m = 0; m < t->ne[2]*t->ne[3]; ++m) {
        for (int64_t ti = 0; ti < ntiles; ++ti) {
            uint8_t * p = buf + (m*t->ne[1] + ti*16)*S*bs;
            memcpy(tile.data(), p, tile.size());
            for (int64_t kb = 0; kb < S; ++kb) {
                for (int64_t r = 0; r < 16; ++r) {
                    memcpy(p + (r*S + kb)*bs, tile.data() + (kb*16 + r)*bs, bs);
                }
            }
        }
    }
}

// a view's base tensor, with `off` moved into the base's bytes
static const ggml_tensor * ggml_cuda_ptq1_ilv_base(const ggml_tensor * t, size_t & off) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    off += (size_t) ((const char *) t->data - (const char *) base->data);
    return base;
}

static bool ggml_cuda_ptq1_ilv_is_packed(const void * p) {
    std::lock_guard<std::mutex> lock(ggml_cuda_ptq1_ilv_mutex);
    const auto it = ggml_cuda_ptq1_ilv_state.find(p);
    return it != ggml_cuda_ptq1_ilv_state.end() && it->second == GGML_CUDA_PTQ1_ILV_PACKED;
}

// init_tensor: a (re)allocated PTQ1_0 tensor holds raw bytes until its upload completes
static void ggml_cuda_ptq1_ilv_fresh(const ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(ggml_cuda_ptq1_ilv_mutex);
    ggml_cuda_ptq1_ilv_state[t->data] = 0;
}

// free_buffer: forget the tensors that lived in [begin, begin + size)
static void ggml_cuda_ptq1_ilv_forget(const void * begin, const size_t size) {
    const char * b = (const char *) begin;
    std::lock_guard<std::mutex> lock(ggml_cuda_ptq1_ilv_mutex);
    for (auto it = ggml_cuda_ptq1_ilv_state.begin(); it != ggml_cuda_ptq1_ilv_state.end();) {
        const char * p = (const char *) it->first;
        it = p >= b && p < b + size ? ggml_cuda_ptq1_ilv_state.erase(it) : std::next(it);
    }
}

// the row-major bytes of a PTQ1_0 base tensor (synchronizes `stream`)
static std::vector<uint8_t> ggml_cuda_ptq1_ilv_image(const ggml_tensor * base, cudaStream_t stream) {
    std::vector<uint8_t> buf(ggml_nbytes(base));
    CUDA_CHECK(cudaMemcpyAsync(buf.data(), base->data, buf.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ggml_cuda_ptq1_ilv_is_packed(base->data)) {
        ggml_cuda_ptq1_ilv_unpack_host(base, buf.data());
    }
    return buf;
}

// write the row-major bytes [offset, offset + size) of a PTQ1_0 tensor or view: queued on `stream`, which is
// synchronized only when a packed tensor had to be patched
static void ggml_cuda_ptq1_ilv_write(const ggml_tensor * t, const void * data, size_t offset, const size_t size,
                                     const int device, cudaStream_t stream) {
    const ggml_tensor * base   = ggml_cuda_ptq1_ilv_base(t, offset);
    const size_t        nbytes = ggml_nbytes(base);
    GGML_ASSERT(offset + size <= nbytes);
    bool pack  = false;
    bool patch = false;
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_ptq1_ilv_mutex);
        size_t & st = ggml_cuda_ptq1_ilv_state[base->data];   // never init'ed here: fresh, counted from 0
        if (st == GGML_CUDA_PTQ1_ILV_PACKED) {
            pack  = offset == 0 && size == nbytes;
            patch = !pack;
        } else {
            st  += size;
            pack = st >= nbytes;
            if (pack) {
                st = GGML_CUDA_PTQ1_ILV_PACKED;
            }
        }
    }
    ggml_cuda_set_device(device);
    if (patch) {
        std::vector<uint8_t> img = ggml_cuda_ptq1_ilv_image(base, stream);
        memcpy(img.data() + offset, data, size);
        CUDA_CHECK(cudaMemcpyAsync(base->data, img.data(), nbytes, cudaMemcpyHostToDevice, stream));
        ggml_cuda_ptq1_ilv_pack(base, device, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));   // img goes out of scope
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) base->data + offset, data, size, cudaMemcpyHostToDevice, stream));
    if (pack) {
        ggml_cuda_ptq1_ilv_pack(base, device, stream);
    }
}

// the row-major bytes [offset, offset + size) of a PTQ1_0 tensor or view (synchronizes `stream`)
static void ggml_cuda_ptq1_ilv_read(const ggml_tensor * t, void * data, size_t offset, const size_t size, cudaStream_t stream) {
    const ggml_tensor * base = ggml_cuda_ptq1_ilv_base(t, offset);
    const std::vector<uint8_t> img = ggml_cuda_ptq1_ilv_image(base, stream);
    GGML_ASSERT(offset + size <= img.size());
    memcpy(data, img.data() + offset, size);
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    if (tensor->type == GGML_TYPE_PTQ1_0) {
        const std::vector<uint8_t> bytes(size, value);
        ggml_cuda_ptq1_ilv_write(tensor, bytes.data(), offset, size, ctx->device, cudaStreamPerThread);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (tensor->type == GGML_TYPE_PTQ1_0) {
        ggml_cuda_ptq1_ilv_write(tensor, data, offset, size, ctx->device, cudaStreamPerThread);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (tensor->type == GGML_TYPE_PTQ1_0) {
        ggml_cuda_ptq1_ilv_read(tensor, data, offset, size, cudaStreamPerThread);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    GGML_ASSERT(tensor->type != GGML_TYPE_PTQ1_0 && "PTQ1_0 ILV16: no 2D uploads");
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    GGML_ASSERT(tensor->type != GGML_TYPE_PTQ1_0 && "PTQ1_0 ILV16: no 2D downloads");

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type == GGML_TYPE_PTQ1_0 || dst->type == GGML_TYPE_PTQ1_0) {
        return false;   // through get/set, so the ILV16 state follows the bytes' meaning, not their order
    }
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

// The device can read pinned host allocations through their UVA pointers, so a
// small upload from one can run as a kernel (ggml_backend_cuda_set_tensor_async). Registered at allocation, dropped at free.
struct ggml_cuda_pinned_alloc {
    uintptr_t host;
    size_t    size;
    uintptr_t dev;
};
static std::mutex                          g_cuda_pinned_mutex;
static std::vector<ggml_cuda_pinned_alloc> g_cuda_pinned_allocs;

static void ggml_cuda_pinned_register(void * host, size_t size) {
    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess || dev == nullptr) {
        (void) cudaGetLastError();   // not mapped: its uploads stay DMA copies
        return;
    }
    std::lock_guard<std::mutex> lock(g_cuda_pinned_mutex);
    g_cuda_pinned_allocs.push_back({ (uintptr_t) host, size, (uintptr_t) dev });
}

static void ggml_cuda_pinned_unregister(void * host) {
    std::lock_guard<std::mutex> lock(g_cuda_pinned_mutex);
    for (size_t k = 0; k < g_cuda_pinned_allocs.size(); ++k) {
        if (g_cuda_pinned_allocs[k].host == (uintptr_t) host) {
            g_cuda_pinned_allocs.erase(g_cuda_pinned_allocs.begin() + k);
            return;
        }
    }
}

// the device address of [p, p + size) when it lies inside one registered pinned allocation, else nullptr
static const void * ggml_cuda_pinned_dev_ptr(const void * p, size_t size) {
    const uintptr_t a = (uintptr_t) p;
    std::lock_guard<std::mutex> lock(g_cuda_pinned_mutex);
    for (const ggml_cuda_pinned_alloc & r : g_cuda_pinned_allocs) {
        if (a >= r.host && a + size <= r.host + r.size) {
            return (const void *) (r.dev + (a - r.host));
        }
    }
    return nullptr;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_cuda_pinned_unregister(buffer->context);
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_cuda_pinned_register(ptr, size);
    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        // PTQ1_0 (ILV16, common.cuh) needs the matrix shape, which only the _nc converter takes
        if (ggml_is_contiguously_allocated(src0) && src0->type != GGML_TYPE_PTQ1_0) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
        compute_type = GGML_TYPE_F32;
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_XYZKV_WHT:
            ggml_cuda_xyzkv_wht(ctx, dst);
            break;
        case GGML_OP_DRAFT_SAMPLE:
            ggml_cuda_op_draft_sample(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

// Read a small pinned host upload in place while preserving stream order.
static __global__ void k_upload_small(const char * __restrict__ src, char * __restrict__ dst, const int64_t n) {
    const int64_t i = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*16;
    if (i >= n) {
        return;
    }
    if (i + 16 <= n && (((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
        *(int4 *) (dst + i) = __ldcv((const int4 *) (src + i));
        return;
    }
    for (int64_t k = i; k < n && k < i + 16; ++k) {
        dst[k] = __ldcv(src + k);
    }
}

// Issue every queued small upload in one launch. blockIdx.y picks the upload and the x blocks stride it.
// the same 16-byte __ldcv reads as k_upload_small, in the same stream position the first queued upload would have had
static __global__ void k_upload_multi(const ggml_backend_cuda_context::up_batch b, const int n_uploads) {
    const int      j   = blockIdx.y;
    if (j >= n_uploads) {
        return;
    }
    const char *   src = b.src[j];
    char *         dst = b.dst[j];
    const int64_t  n   = b.n[j];
    const bool     vec = ((((uintptr_t) src) | ((uintptr_t) dst)) & 15) == 0;
    for (int64_t i = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*16; i < n; i += (int64_t) gridDim.x*blockDim.x*16) {
        if (vec && i + 16 <= n) {
            *(int4 *) (dst + i) = __ldcv((const int4 *) (src + i));
        } else {
            for (int64_t k = i; k < n && k < i + 16; ++k) {
                dst[k] = __ldcv(src + k);
            }
        }
    }
}

struct ggml_cuda_upload_batch {
    ggml_backend_cuda_context::up_batch data = {};
    int                                 count = 0;
    int64_t                             max_size = 0;
};

static ggml_cuda_upload_batch ggml_cuda_upload_take(ggml_backend_cuda_context * cuda_ctx) {
    ggml_cuda_upload_batch batch;
    batch.data         = cuda_ctx->up_pend;
    batch.count        = cuda_ctx->n_up_pend;
    batch.max_size     = cuda_ctx->up_pend_max;
    cuda_ctx->n_up_pend   = 0;
    cuda_ctx->up_pend_max = 0;
    return batch;
}

static cudaKernelNodeParams ggml_cuda_upload_params(ggml_cuda_upload_batch & batch, void ** args) {
    constexpr int bs = 256;
    const int64_t n_thr = (batch.max_size + 15) / 16;
    const unsigned nbx = (unsigned) std::max<int64_t>(1, std::min<int64_t>((n_thr + bs - 1) / bs, 64));
    args[0] = &batch.data;
    args[1] = &batch.count;
    cudaKernelNodeParams params = {};
    params.func        = (void *) k_upload_multi;
    params.gridDim     = dim3(nbx, (unsigned) std::max(batch.count, 1), 1);
    params.blockDim    = dim3(bs, 1, 1);
    params.kernelParams = args;
    return params;
}

static void ggml_cuda_upload_launch(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_upload_batch & batch) {
    if (batch.count == 0) {
        return;
    }
    void * args[2];
    const cudaKernelNodeParams params = ggml_cuda_upload_params(batch, args);
    k_upload_multi<<<params.gridDim, params.blockDim, 0, cuda_ctx->stream(cuda_ctx->device, 0)>>>(batch.data, batch.count);
    CUDA_CHECK(cudaGetLastError());
}

// Issue queued uploads on stream 0 before this backend submits or waits.
// on it, so the uploads keep their place in the stream order
static void ggml_cuda_upload_flush(ggml_backend_cuda_context * cuda_ctx) {
    if (cuda_ctx->n_up_pend == 0) {
        return;
    }
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_upload_batch batch = ggml_cuda_upload_take(cuda_ctx);
    ggml_cuda_upload_launch(cuda_ctx, batch);
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (tensor->type == GGML_TYPE_PTQ1_0) {
        ggml_cuda_upload_flush(cuda_ctx);
        ggml_cuda_ptq1_ilv_write(tensor, data, offset, size, cuda_ctx->device, cuda_ctx->stream());
        return;
    }
    if (size > 0 && size <= 256*1024) {
        const void * dev_src = ggml_cuda_pinned_dev_ptr(data, size);
        if (dev_src != nullptr && cuda_ctx->curr_stream_no == 0) {
            if (cuda_ctx->n_up_pend == ggml_backend_cuda_context::UP_MAX) {
                ggml_cuda_upload_flush(cuda_ctx);
            }
            const int j = cuda_ctx->n_up_pend++;
            cuda_ctx->up_pend.src[j] = (const char *) dev_src;
            cuda_ctx->up_pend.dst[j] = (char *) tensor->data + offset;
            cuda_ctx->up_pend.n[j]   = (int64_t) size;
            cuda_ctx->up_pend_max    = std::max<int64_t>(cuda_ctx->up_pend_max, (int64_t) size);
            return;
        }
        if (dev_src != nullptr) {
            ggml_cuda_upload_flush(cuda_ctx);
            ggml_cuda_set_device(cuda_ctx->device);
            const int64_t n_thr = ((int64_t) size + 15) / 16;
            constexpr int bs = 256;
            k_upload_small<<<(unsigned) ((n_thr + bs - 1) / bs), bs, 0, cuda_ctx->stream()>>>(
                (const char *) dev_src, (char *) tensor->data + offset, (int64_t) size);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

// Store a small download directly into the mapped pinned destination.
static __global__ void k_download_small(const char * __restrict__ src, char * __restrict__ dst, const int64_t n) {
    const int64_t i = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*16;
    if (i >= n) {
        return;
    }
    if (i + 16 <= n && (((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
        *(int4 *) (dst + i) = *(const int4 *) (src + i);
        return;
    }
    for (int64_t k = i; k < n && k < i + 16; ++k) {
        dst[k] = src[k];
    }
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    ggml_cuda_upload_flush(cuda_ctx);

    const bool ilv = tensor->type == GGML_TYPE_PTQ1_0;
    if (!ilv && size > 0 && size <= 256*1024) {
        void * dev_dst = (void *) ggml_cuda_pinned_dev_ptr(data, size);
        if (dev_dst != nullptr) {
            ggml_cuda_set_device(cuda_ctx->device);
            const int64_t n_thr = ((int64_t) size + 15) / 16;
            constexpr int bs = 256;
            k_download_small<<<(unsigned) ((n_thr + bs - 1) / bs), bs, 0, cuda_ctx->stream()>>>(
                (const char *) tensor->data + offset, (char *) dev_dst, (int64_t) size);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    if (tensor->type == GGML_TYPE_PTQ1_0) {
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_ptq1_ilv_read(tensor, data, offset, size, cuda_ctx->stream());
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->type != GGML_TYPE_PTQ1_0 && "PTQ1_0 ILV16: no 2D copies");

    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->type != GGML_TYPE_PTQ1_0 && "PTQ1_0 ILV16: no 2D copies");

    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    if (src->type == GGML_TYPE_PTQ1_0 || dst->type == GGML_TYPE_PTQ1_0) {
        return false;   // ILV16 state: through get/set (ggml_backend_cuda_buffer_cpy_tensor)
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;
    ggml_cuda_upload_flush(cuda_ctx_src);
    ggml_cuda_upload_flush(cuda_ctx_dst);

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    const ggml_tensor * first = cgraph->nodes[0];
    if (cgraph->n_nodes == 0) {
        return first;
    }
    const ggml_tensor * last = cgraph->nodes[cgraph->n_nodes - 1];
    uintptr_t key = (uintptr_t) first;
    key ^= (uintptr_t) cgraph->n_nodes * (uintptr_t) 0x9E3779B97F4A7C15ull;
    key ^= (uintptr_t) (last->ne[1] + 1) * (uintptr_t) 0xC2B2AE3D27D4EB4Full;
    // Include input batch size because draft and catch-up graphs can share node and output shapes.
    key ^= (uintptr_t) (first->ne[1] + 1) * (uintptr_t) 0x165667B19E3779F9ull;
    // Include MUL_MAT weight pointers to distinguish same-shape draft steps.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        const ggml_tensor * w    = node->op == GGML_OP_MUL_MAT ? node->src[0] : nullptr;
        if (w != nullptr && w->buffer != nullptr && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            key = (key ^ (uintptr_t) w->data) * (uintptr_t) 0x100000001B3ull;
        }
    }
    return (const void *) key;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (!res && cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// A fused launch requires disjoint read and write ranges across CTAs.
// Planning keeps every declared range alive until the fused group's last output is allocated.
// allow: (write, read) pairs that may overlap because the kernel reads and writes that element in the same thread
// (an in-place ADD fused into a chain).
struct ggml_cuda_fusion_plan {
    struct group { std::vector<const ggml_tensor *> reads, writes; };
    std::vector<group> groups;
    bool matched = false;   // set by the guard in plan mode: the matcher just called claims its nodes
};
static thread_local ggml_cuda_fusion_plan * g_ggml_cuda_fusion_plan = nullptr;

static bool ggml_cuda_fused_ranges_ok(std::initializer_list<const ggml_tensor *> writes,
                                      std::initializer_list<const ggml_tensor *> reads, const char * who,
                                      std::initializer_list<std::pair<const ggml_tensor *, const ggml_tensor *>> allow = {}) {
    if (g_ggml_cuda_fusion_plan != nullptr) {
        ggml_cuda_fusion_plan::group g;
        for (const ggml_tensor * t : writes) { if (t != nullptr) { g.writes.push_back(t); } }
        for (const ggml_tensor * t : reads)  { if (t != nullptr) { g.reads.push_back(t); } }
        g_ggml_cuda_fusion_plan->groups.push_back(std::move(g));
        g_ggml_cuda_fusion_plan->matched = true;
        return false;   // plan mode: record, launch nothing
    }
    auto overlaps = [](const ggml_tensor * p, const ggml_tensor * q) {
        const char * p0 = (const char *) p->data;
        const char * q0 = (const char *) q->data;
        return p0 < q0 + ggml_nbytes(q) && q0 < p0 + ggml_nbytes(p);
    };
    auto allowed = [&](const ggml_tensor * w, const ggml_tensor * r) {
        for (const auto & a : allow) {
            if (a.first == w && a.second == r) {
                return true;
            }
        }
        return false;
    };
    const ggml_tensor * hit_w = nullptr;
    const ggml_tensor * hit_o = nullptr;
    for (auto w = writes.begin(); hit_w == nullptr && w != writes.end(); ++w) {
        if (*w == nullptr) {
            continue;
        }
        for (const ggml_tensor * r : reads) {
            if (r != nullptr && overlaps(*w, r) && !allowed(*w, r)) {
                hit_w = *w; hit_o = r;
                break;
            }
        }
        for (auto w2 = std::next(w); hit_w == nullptr && w2 != writes.end(); ++w2) {
            if (*w2 != nullptr && overlaps(*w, *w2)) {
                hit_w = *w; hit_o = *w2;
            }
        }
    }
    if (hit_w != nullptr) {
        // the first few per matcher, with the colliding tensors: which placement defeats which fusion
        static std::mutex m;
        static std::map<std::string, int> said;
        std::lock_guard<std::mutex> lock(m);
        if (said[who]++ < 4) {
            GGML_LOG_WARN("%s: fused launch declined -- output %s [%s] overlaps %s [%s] (placement race)\n", who,
                hit_w->name, ggml_op_name(hit_w->op), hit_o->name, ggml_op_name(hit_o->op));
        }
    }
    return hit_w == nullptr;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// Launch-count fusions for the Qwen3.5 Gated-DeltaNet decode graph:
// (A) is pure copying, (B) applies the same three f32 operations in the same order as the three kernels it replaces.

// Widest token count instantiated by the fused GDN prologue.
static constexpr int ggml_cuda_gdn_max_tokens = 16;

struct ggml_cuda_gdn_conv_prologue_args {
    const char * state;
    const int32_t * state_row;
    size_t state_row_stride;
    float * gathered;
    const char * new_steps;
    size_t new_nb0;
    size_t new_nb1;
    float * conv_input;
    float * snapshots[ggml_cuda_gdn_max_tokens];
    int snapshot_start[ggml_cuda_gdn_max_tokens];
    const char * conv_weight;
    size_t conv_weight_nb1;
    float * conv_silu;
    float * qk_norm;
    float eps;
};

template <int n_tokens>
static __global__ void k_gdn_conv_prologue(ggml_cuda_gdn_conv_prologue_args args) {
    constexpr int channels = 10240;
    constexpr int channels_per_block = 128;
    constexpr int conv_width = 4;
    constexpr int old_steps = conv_width - 1;
    constexpr int qk_blocks = 32;

    const int tid = threadIdx.x;
    const int channel = blockIdx.x * channels_per_block + tid;
    const float * state_row = (const float *) (args.state + (size_t) args.state_row[0] * args.state_row_stride);

    float window[old_steps + n_tokens];
#pragma unroll
    for (int j = 0; j < old_steps; ++j) {
        window[j] = state_row[channel * old_steps + j];
        args.gathered[channel * old_steps + j] = window[j];
        args.conv_input[channel * (old_steps + n_tokens) + j] = window[j];
    }
#pragma unroll
    for (int j = 0; j < n_tokens; ++j) {
        window[old_steps + j] = *(const float *) (args.new_steps + (size_t) j * args.new_nb0 + (size_t) channel * args.new_nb1);
        args.conv_input[channel * (old_steps + n_tokens) + old_steps + j] = window[old_steps + j];
    }

#pragma unroll
    for (int snapshot = 0; snapshot < n_tokens; ++snapshot) {
#pragma unroll
        for (int j = 0; j < old_steps; ++j) {
            args.snapshots[snapshot][channel * old_steps + j] = window[args.snapshot_start[snapshot] + j];
        }
    }

    float weight[conv_width];
#pragma unroll
    for (int j = 0; j < conv_width; ++j) {
        weight[j] = *(const float *) (args.conv_weight + (size_t) channel * args.conv_weight_nb1 + j * sizeof(float));
    }

    __shared__ float conv_shared[n_tokens][channels_per_block];
#pragma unroll
    for (int token = 0; token < n_tokens; ++token) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < conv_width; ++j) {
            sum += window[token + j] * weight[j];
        }
        const float value = ggml_cuda_op_silu_single(sum);
        args.conv_silu[token * channels + channel] = value;
        conv_shared[token][tid] = value;
    }
    __syncthreads();

    if (blockIdx.x < qk_blocks && tid < WARP_SIZE) {
#pragma unroll
        for (int token = 0; token < n_tokens; ++token) {
            float sum = 0.0f;
#pragma unroll
            for (int col = tid; col < channels_per_block; col += WARP_SIZE) {
                const float value = conv_shared[token][col];
                sum += value * value;
            }
            sum = warp_reduce_sum<WARP_SIZE>(sum);
            const float scale = rsqrtf(fmaxf(sum, args.eps * args.eps));
#pragma unroll
            for (int col = tid; col < channels_per_block; col += WARP_SIZE) {
                args.qk_norm[(token * qk_blocks + blockIdx.x) * channels_per_block + col] =
                    scale * conv_shared[token][col];
            }
        }
    }
}

// Replay layout:
//   state row [C, W] and pack row [C, P], channel-fastest, both at cell s_copy[0]
//   old window w = row conv_idx[w] of [state (W rows) ++ pack (P rows)]
//   commit: state row at kv_head <- old window; park: pack row at kv_head <- this batch's inputs [C, n]
//   conv over [old window, inputs] (4 taps), silu; l2 norm of the q/k heads (channels 0..4095)
// One thread per channel: its reads of the state and pack rows precede its writes to the kv_head rows, and no other
// thread touches that channel, so reading and writing the same cell is ordered without a barrier.
// The conv sum, zero bias add, silu, and l2 norm keep the graph's f32 operation order.
struct ggml_cuda_gdn_conv_replay_args {
    const char *    state;             // conv state table (cell rows)
    const char *    pack;              // pack table (cell rows)
    const int32_t * s_row;             // s_copy[0]: the cell this sequence reads
    size_t          state_row_stride;
    size_t          pack_row_stride;
    const int32_t * conv_idx;          // W rows into [state ++ pack]
    const char *    qkv;               // this batch's inputs [C, n], channel-fastest
    size_t          qkv_nb1;
    float *         state_dst;         // commit: state row at kv_head, [C, W]
    float *         pack_dst;          // park: pack row at kv_head, [C, n]
    size_t          pack_dst_nb1;
    const char *    conv_weight;
    size_t          conv_weight_nb1;
    float *         conv_silu;         // [C, n]
    float *         qk_norm;           // [128, 32, n]
    float           eps;
    float           bias0;             // runtime 0.0f: ssm_conv_f32's absent bias, added as it does
};

template <int n_tokens>
static __global__ void k_gdn_conv_replay(ggml_cuda_gdn_conv_replay_args args) {
    constexpr int channels           = 10240;
    constexpr int channels_per_block = 128;
    constexpr int conv_width         = 4;
    constexpr int old_steps          = conv_width - 1;
    constexpr int qk_blocks          = 32;

    const int tid     = threadIdx.x;
    const int channel = blockIdx.x * channels_per_block + tid;
    const int srow    = args.s_row[0];
    const float * st  = (const float *) (args.state + (size_t) srow * args.state_row_stride);
    const float * pk  = (const float *) (args.pack  + (size_t) srow * args.pack_row_stride);

    float window[old_steps + n_tokens];
#pragma unroll
    for (int w = 0; w < old_steps; ++w) {
        const int r = args.conv_idx[w];
        window[w] = r < old_steps ? st[r * channels + channel] : pk[(r - old_steps) * channels + channel];
    }
#pragma unroll
    for (int j = 0; j < n_tokens; ++j) {
        window[old_steps + j] = *(const float *) (args.qkv + (size_t) j * args.qkv_nb1 + (size_t) channel * sizeof(float));
    }
    // commit and park AFTER every read of this channel (the kv_head cell may be the cell just read)
#pragma unroll
    for (int w = 0; w < old_steps; ++w) {
        args.state_dst[w * channels + channel] = window[w];
    }
#pragma unroll
    for (int j = 0; j < n_tokens; ++j) {
        *(float *) ((char *) args.pack_dst + (size_t) j * args.pack_dst_nb1 + (size_t) channel * sizeof(float)) =
            window[old_steps + j];
    }

    float weight[conv_width];
#pragma unroll
    for (int j = 0; j < conv_width; ++j) {
        weight[j] = *(const float *) (args.conv_weight + (size_t) channel * args.conv_weight_nb1 + j * sizeof(float));
    }

    __shared__ float conv_shared[n_tokens][channels_per_block];
#pragma unroll
    for (int token = 0; token < n_tokens; ++token) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < conv_width; ++j) {
            sum += window[token + j] * weight[j];
        }
        sum += args.bias0;
        const float value = ggml_cuda_op_silu_single(sum);
        args.conv_silu[token * channels + channel] = value;
        conv_shared[token][tid] = value;
    }
    __syncthreads();

    if (blockIdx.x < qk_blocks && tid < WARP_SIZE) {
#pragma unroll
        for (int token = 0; token < n_tokens; ++token) {
            float sum = 0.0f;
#pragma unroll
            for (int col = tid; col < channels_per_block; col += WARP_SIZE) {
                const float value = conv_shared[token][col];
                sum += value * value;
            }
            sum = warp_reduce_sum<WARP_SIZE>(sum);
            const float scale = rsqrtf(fmaxf(sum, args.eps * args.eps));
#pragma unroll
            for (int col = tid; col < channels_per_block; col += WARP_SIZE) {
                args.qk_norm[(token * qk_blocks + blockIdx.x) * channels_per_block + col] =
                    scale * conv_shared[token][col];
            }
        }
    }
}

static __global__ void k_gdn_alpha_beta_prologue(
        const nv_bfloat16 * alpha_weight, const nv_bfloat16 * beta_weight, const float * activations,
        const float * dt, const float * a, float * gate, float * beta) {
    constexpr int n_embd = 5120;
    constexpr int rank = 48;
    constexpr int n_tokens = 5;
    constexpr int block_size = 256;
    constexpr int ncols2 = n_embd / 2;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const nv_bfloat162 * alpha2 = (const nv_bfloat162 *) (alpha_weight + row * n_embd);
    const nv_bfloat162 * beta2  = (const nv_bfloat162 *) (beta_weight  + row * n_embd);
    const float2 * activations2 = (const float2 *) activations;
    float alpha_sum[n_tokens] = { 0.0f };
    float beta_sum[n_tokens]  = { 0.0f };

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 alpha_value = alpha2[col2];
        const nv_bfloat162 beta_value  = beta2[col2];
#pragma unroll
        for (int token = 0; token < n_tokens; ++token) {
            const float2 x = activations2[token * ncols2 + col2];
            ggml_cuda_mad(alpha_sum[token], alpha_value.x, x.x);
            ggml_cuda_mad(alpha_sum[token], alpha_value.y, x.y);
            ggml_cuda_mad(beta_sum[token], beta_value.x, x.x);
            ggml_cuda_mad(beta_sum[token], beta_value.y, x.y);
        }
    }

    __shared__ float alpha_warp[WARP_SIZE];
    __shared__ float beta_warp[WARP_SIZE];
    if (tid < WARP_SIZE) {
        alpha_warp[tid] = 0.0f;
        beta_warp[tid] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int token = 0; token < n_tokens; ++token) {
        alpha_sum[token] = warp_reduce_sum<WARP_SIZE>(alpha_sum[token]);
        beta_sum[token] = warp_reduce_sum<WARP_SIZE>(beta_sum[token]);
        alpha_warp[tid / WARP_SIZE] = alpha_sum[token];
        beta_warp[tid / WARP_SIZE] = beta_sum[token];
        __syncthreads();
        if (tid < WARP_SIZE) {
            alpha_sum[token] = warp_reduce_sum<WARP_SIZE>(alpha_warp[tid]);
            beta_sum[token] = warp_reduce_sum<WARP_SIZE>(beta_warp[tid]);
        }
        __syncthreads();
    }

    if (tid < n_tokens) {
        const float biased = alpha_sum[tid] + dt[row];
        const float softplus = biased > 20.0f ? biased : logf(1.0f + expf(biased));
        gate[tid * rank + row] = softplus * a[row];
        beta[tid * rank + row] = 1.0f / (1.0f + expf(-beta_sum[tid]));
    }
}

static const ggml_tensor * ggml_cuda_gdn_unwrap_view(const ggml_tensor * tensor) {
    while (tensor != nullptr && (tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_TRANSPOSE ||
            tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE)) {
        tensor = tensor->src[0];
    }
    return tensor;
}

static int ggml_cuda_gdn_next_compute(const ggml_cgraph * cgraph, int index) {
    while (index < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[index])) {
        ++index;
    }
    return index;
}

// Synchronization events for the GDN alpha/beta side stream.
static cudaEvent_t g_gdn_ab_fork[GGML_CUDA_MAX_DEVICES] = {};
static cudaEvent_t g_gdn_ab_join[GGML_CUDA_MAX_DEVICES] = {};
// The attention Q chain, V write, and K write use streams 0, 1, and 2.
static cudaEvent_t g_attn_fork[GGML_CUDA_MAX_DEVICES]  = {};
static cudaEvent_t g_attn_join1[GGML_CUDA_MAX_DEVICES] = {};
static cudaEvent_t g_attn_join2[GGML_CUDA_MAX_DEVICES] = {};

// the GDN layer's alpha projection: MUL_MAT(blk.N.ssm_alpha.weight (bf16), attn_norm)
static bool ggml_cuda_is_gdn_alpha_mm(const ggml_tensor * node, int * layer) {
    int l = -1;
    if (node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || node->src[0]->type != GGML_TYPE_BF16 ||
            sscanf(node->src[0]->name, "blk.%d.", &l) != 1) {
        return false;
    }
    char want[64];
    snprintf(want, sizeof(want), "blk.%d.ssm_alpha.weight", l);
    if (strcmp(node->src[0]->name, want) != 0) {
        return false;
    }
    if (layer) {
        *layer = l;
    }
    return true;
}

static int ggml_cuda_try_fuse_gdn_conv_prologue(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    constexpr int channels = 10240;
    constexpr int old_steps = 3;
    // n_tokens is a runtime dimension provided by the concat.
    int n_tokens = 0;

    const ggml_tensor * gathered = cgraph->nodes[i];
    if (gathered->op != GGML_OP_GET_ROWS || gathered->type != GGML_TYPE_F32 ||
            gathered->src[0]->type != GGML_TYPE_F32 || gathered->src[1]->type != GGML_TYPE_I32 ||
            ggml_nelements(gathered) != old_steps * channels || ggml_nelements(gathered->src[1]) != 1 ||
            gathered->src[0]->nb[0] != sizeof(float) || !ggml_is_contiguous(gathered)) {
        return 0;
    }

    int index = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_CONCAT) {
        return 0;
    }
    const ggml_tensor * concat = cgraph->nodes[index];
    n_tokens = (int) concat->ne[0] - old_steps;
    if (n_tokens < 1 || n_tokens > ggml_cuda_gdn_max_tokens ||
            (n_tokens > 10 && n_tokens != 12 && n_tokens != 16)) {
        return 0;                                  // no instantiation for this width: decline, as before
    }
    if (concat->type != GGML_TYPE_F32 || concat->op_params[0] != 0 ||
            concat->ne[1] != channels ||
            concat->ne[2] != 1 || concat->ne[3] != 1 || !ggml_is_contiguous(concat)) {
        return 0;
    }
    const ggml_tensor * old_source = nullptr;
    const ggml_tensor * new_source = nullptr;
    if (ggml_cuda_gdn_unwrap_view(concat->src[0]) == gathered) {
        old_source = concat->src[0];
        new_source = concat->src[1];
    } else if (ggml_cuda_gdn_unwrap_view(concat->src[1]) == gathered) {
        old_source = concat->src[1];
        new_source = concat->src[0];
    } else {
        return 0;
    }
    if (old_source->ne[0] != old_steps || old_source->ne[1] != channels ||
            new_source->type != GGML_TYPE_F32 || new_source->ne[0] != n_tokens ||
            new_source->ne[1] != channels || new_source->nb[1] != sizeof(float)) {
        return 0;
    }

    // sized by the cap, not by n_tokens: the count is a runtime dimension now, so these cannot be VLAs
    const ggml_tensor * snapshots[ggml_cuda_gdn_max_tokens] = {};
    int snapshot_start[ggml_cuda_gdn_max_tokens] = {};
    for (int snapshot = 0; snapshot < n_tokens; ++snapshot) {
        index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
        if (index >= cgraph->n_nodes) {
            return 0;
        }
        const ggml_tensor * cpy = cgraph->nodes[index];
        if (cpy->op != GGML_OP_CPY || cpy->type != GGML_TYPE_F32 || cpy->src[0]->type != GGML_TYPE_F32 ||
                ggml_nelements(cpy) != old_steps * channels || !ggml_is_contiguous(cpy) ||
                cpy->src[0]->ne[0] != old_steps || cpy->src[0]->ne[1] != channels ||
                cpy->src[0]->nb[0] != sizeof(float) ||
                cpy->src[0]->nb[1] != (old_steps + n_tokens) * sizeof(float) ||
                ggml_cuda_gdn_unwrap_view(cpy->src[0]) != concat) {
            return 0;
        }
        // from the view, not data pointers: graph_optimize runs this matcher before allocation (plan mode), where data is
        // null; view_offs is relative to view_src (concat, not itself a view), which is what data - concat->data equals
        const ptrdiff_t offset = cpy->src[0]->view_src == concat ? (ptrdiff_t) cpy->src[0]->view_offs : (ptrdiff_t) -1;
        if (offset < 0 || offset % (ptrdiff_t) sizeof(float) != 0 ||
                offset / (ptrdiff_t) sizeof(float) < 1 || offset / (ptrdiff_t) sizeof(float) > n_tokens) {
            return 0;
        }
        snapshots[snapshot] = cpy;
        snapshot_start[snapshot] = (int) (offset / sizeof(float));
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    const ggml_tensor * conv = cgraph->nodes[index];
    if (ggml_cuda_gdn_unwrap_view(conv->src[0]) != concat || conv->src[1]->type != GGML_TYPE_F32 ||
            conv->src[1]->ne[0] != 4 || conv->src[1]->ne[1] != channels ||
            conv->src[1]->nb[0] != sizeof(float) || conv->ne[0] != channels || conv->ne[1] != n_tokens) {
        return 0;
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[index]) != GGML_UNARY_OP_SILU ||
            cgraph->nodes[index]->src[0] != conv) {
        return 0;
    }
    const ggml_tensor * silu = cgraph->nodes[index];
    if (silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu)) {
        return 0;
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_L2_NORM) {
        return 0;
    }
    const ggml_tensor * norm = cgraph->nodes[index];
    if (norm->type != GGML_TYPE_F32 || norm->ne[0] != 128 || norm->ne[1] != 32 ||
            norm->ne[2] != n_tokens || norm->ne[3] != 1 || !ggml_is_contiguous(norm) ||
            ggml_cuda_gdn_unwrap_view(norm->src[0]) != silu) {
        return 0;
    }

    ggml_cuda_gdn_conv_prologue_args args = {};
    args.state = (const char *) gathered->src[0]->data;
    args.state_row = (const int32_t *) gathered->src[1]->data;
    args.state_row_stride = gathered->src[0]->nb[1];
    args.gathered = (float *) gathered->data;
    args.new_steps = (const char *) new_source->data;
    args.new_nb0 = new_source->nb[0];
    args.new_nb1 = new_source->nb[1];
    args.conv_input = (float *) concat->data;
    for (int snapshot = 0; snapshot < n_tokens; ++snapshot) {
        args.snapshots[snapshot] = (float *) snapshots[snapshot]->data;
        args.snapshot_start[snapshot] = snapshot_start[snapshot];
    }
    args.conv_weight = (const char *) conv->src[1]->data;
    args.conv_weight_nb1 = conv->src[1]->nb[1];
    args.conv_silu = (float *) silu->data;
    args.qk_norm = (float *) norm->data;
    memcpy(&args.eps, norm->op_params, sizeof(args.eps));

    // gathered and conv_input are dead after the chain, so ggml-alloc may hand their bytes to silu/norm -- the kernel
    // writes all of them. The state table is read and snapshot-written per channel by the same thread (program order),
    // so it is not a cross-CTA hazard and is left out of the read set (its range covers the snapshot rows).
    // Declare every snapshot the kernel may write. Unused slots are null and skipped.
    static_assert(ggml_cuda_gdn_max_tokens == 16, "declare every snapshot slot below");
    if (!ggml_cuda_fused_ranges_ok({ gathered, concat,
                                     snapshots[0], snapshots[1], snapshots[2],  snapshots[3],  snapshots[4],  snapshots[5],
                                     snapshots[6], snapshots[7], snapshots[8],  snapshots[9],  snapshots[10], snapshots[11],
                                     snapshots[12], snapshots[13], snapshots[14], snapshots[15], silu, norm },
                                   { new_source, conv->src[1] }, "gdn_conv_prologue")) {
        return 0;
    }
    // Dispatch on the actual token count.
    const int grid = channels / 128;
    cudaStream_t st = cuda_ctx->stream();
    switch (n_tokens) {
        case  1: k_gdn_conv_prologue< 1><<<grid, 128, 0, st>>>(args); break;
        case  2: k_gdn_conv_prologue< 2><<<grid, 128, 0, st>>>(args); break;
        case  3: k_gdn_conv_prologue< 3><<<grid, 128, 0, st>>>(args); break;
        case  4: k_gdn_conv_prologue< 4><<<grid, 128, 0, st>>>(args); break;
        case  5: k_gdn_conv_prologue< 5><<<grid, 128, 0, st>>>(args); break;
        case  6: k_gdn_conv_prologue< 6><<<grid, 128, 0, st>>>(args); break;
        case  7: k_gdn_conv_prologue< 7><<<grid, 128, 0, st>>>(args); break;
        case  8: k_gdn_conv_prologue< 8><<<grid, 128, 0, st>>>(args); break;
        case  9: k_gdn_conv_prologue< 9><<<grid, 128, 0, st>>>(args); break;
        case 10: k_gdn_conv_prologue<10><<<grid, 128, 0, st>>>(args); break;
        case 12: k_gdn_conv_prologue<12><<<grid, 128, 0, st>>>(args); break;
        case 16: k_gdn_conv_prologue<16><<<grid, 128, 0, st>>>(args); break;
        default: GGML_ABORT("GDN conv prologue: unhandled n_tokens %d (the matcher should have declined)",
                            n_tokens);
    }
    return index - i;
}

// the rs_replay conv section (k_gdn_conv_replay): GET_ROWS(state, s_copy) GET_ROWS(pack, s_copy) CONCAT(dim 1)
// GET_ROWS(window, conv_idx) CPY(park) CPY(commit) CONCAT(dim 0) SSM_CONV UNARY(SILU) L2_NORM, views skipped. Only
// structure is checked (plan mode runs this before allocation, when data is null); args are read at evaluation.
// Return the next node that does work, skipping views, no-ops and zero-element nodes.
static int ggml_cuda_gdn_next_live(const ggml_cgraph * cgraph, int index) {
    while (index < cgraph->n_nodes &&
           (ggml_cuda_is_view_or_noop(cgraph->nodes[index]) || ggml_nelements(cgraph->nodes[index]) == 0)) {
        ++index;
    }
    return index;
}

static int ggml_cuda_try_fuse_gdn_conv_replay(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    constexpr int channels  = 10240;
    constexpr int old_steps = 3;
#define GDN_REPLAY_DECLINE() return 0

    const ggml_tensor * g_state = cgraph->nodes[i];
    if (g_state->op != GGML_OP_GET_ROWS || g_state->type != GGML_TYPE_F32 || g_state->src[0]->type != GGML_TYPE_F32 ||
            g_state->src[1]->type != GGML_TYPE_I32 || ggml_nelements(g_state->src[1]) != 1 ||
            g_state->ne[0] != old_steps * channels || ggml_nrows(g_state) != 1 ||
            g_state->src[0]->nb[0] != sizeof(float) || !ggml_is_contiguous(g_state)) {
        GDN_REPLAY_DECLINE();
    }
    int index = ggml_cuda_gdn_next_live(cgraph, i + 1);
    if (index >= cgraph->n_nodes) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * g_pack = cgraph->nodes[index];
    if (g_pack->op != GGML_OP_GET_ROWS || g_pack->type != GGML_TYPE_F32 || g_pack->src[0]->type != GGML_TYPE_F32 ||
            g_pack->src[1] != g_state->src[1] || g_pack->ne[0] % channels != 0 || ggml_nrows(g_pack) != 1 ||
            g_pack->src[0]->nb[0] != sizeof(float) || !ggml_is_contiguous(g_pack)) {
        GDN_REPLAY_DECLINE();
    }
    const int pack_tokens = (int) (g_pack->ne[0] / channels);
    if (pack_tokens < 1 || pack_tokens > ggml_cuda_gdn_max_tokens) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * window = cgraph->nodes[index];
    if (window->op != GGML_OP_CONCAT || window->type != GGML_TYPE_F32 || window->op_params[0] != 1 ||
            window->ne[0] != channels || window->ne[1] != old_steps + pack_tokens || window->ne[2] != 1 ||
            ggml_cuda_gdn_unwrap_view(window->src[0]) != g_state || ggml_cuda_gdn_unwrap_view(window->src[1]) != g_pack ||
            !ggml_is_contiguous(window)) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * conv_eff = cgraph->nodes[index];
    if (conv_eff->op != GGML_OP_GET_ROWS || conv_eff->src[0] != window || conv_eff->type != GGML_TYPE_F32 ||
            conv_eff->src[1]->type != GGML_TYPE_I32 || ggml_nelements(conv_eff->src[1]) != old_steps ||
            conv_eff->ne[0] != channels || conv_eff->ne[1] != old_steps || conv_eff->ne[2] != 1 ||
            !ggml_is_contiguous(conv_eff)) {
        GDN_REPLAY_DECLINE();
    }

    // the park and commit copies, in either order
    const ggml_tensor * park   = nullptr;
    const ggml_tensor * commit = nullptr;
    for (int k = 0; k < 2; ++k) {
        index = ggml_cuda_gdn_next_live(cgraph, index + 1);
        if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_CPY) {
            GDN_REPLAY_DECLINE();
        }
        const ggml_tensor * cpy = cgraph->nodes[index];
        if (cpy->type != GGML_TYPE_F32 || cpy->src[0]->type != GGML_TYPE_F32 || cpy->src[1]->type != GGML_TYPE_F32) {
            GDN_REPLAY_DECLINE();
        }
        const ggml_tensor * dst_table = ggml_cuda_gdn_unwrap_view(cpy->src[1]);
        if (ggml_cuda_gdn_unwrap_view(cpy->src[0]) == conv_eff && dst_table == ggml_cuda_gdn_unwrap_view(g_state->src[0]) &&
                commit == nullptr) {
            commit = cpy;
        } else if (dst_table == ggml_cuda_gdn_unwrap_view(g_pack->src[0]) && park == nullptr) {
            park = cpy;
        } else {
            GDN_REPLAY_DECLINE();
        }
    }
    const ggml_tensor * qkv = park->src[0];   // this batch's inputs [C, n], channel-fastest
    const int n_tokens = (int) qkv->ne[1];
    if (qkv->ne[0] != channels || n_tokens < 1 || n_tokens > pack_tokens || qkv->ne[2] != 1 ||
            qkv->nb[0] != sizeof(float) || n_tokens > ggml_cuda_gdn_max_tokens ||
            (n_tokens > 10 && n_tokens != 12 && n_tokens != 16) ||
            !ggml_is_contiguous(commit->src[1]) || ggml_nelements(commit->src[1]) != old_steps * channels ||
            ggml_nelements(park->src[1]) != (int64_t) channels * n_tokens || park->src[1]->nb[0] != sizeof(float)) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * conv_input = cgraph->nodes[index];
    if (conv_input->op != GGML_OP_CONCAT || conv_input->op_params[0] != 0 || conv_input->type != GGML_TYPE_F32 ||
            conv_input->ne[0] != old_steps + n_tokens || conv_input->ne[1] != channels || conv_input->ne[2] != 1 ||
            ggml_cuda_gdn_unwrap_view(conv_input->src[0]) != conv_eff ||
            ggml_cuda_gdn_unwrap_view(conv_input->src[1]) != ggml_cuda_gdn_unwrap_view(qkv)) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_SSM_CONV) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * conv = cgraph->nodes[index];
    if (ggml_cuda_gdn_unwrap_view(conv->src[0]) != conv_input || conv->src[1]->type != GGML_TYPE_F32 ||
            conv->src[1]->ne[0] != 4 || conv->src[1]->ne[1] != channels || conv->src[1]->nb[0] != sizeof(float) ||
            conv->ne[0] != channels || conv->ne[1] != n_tokens) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[index]) != GGML_UNARY_OP_SILU || cgraph->nodes[index]->src[0] != conv) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * silu = cgraph->nodes[index];
    if (silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu)) {
        GDN_REPLAY_DECLINE();
    }

    index = ggml_cuda_gdn_next_live(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_L2_NORM) {
        GDN_REPLAY_DECLINE();
    }
    const ggml_tensor * norm = cgraph->nodes[index];
    if (norm->type != GGML_TYPE_F32 || norm->ne[0] != 128 || norm->ne[1] != 32 || norm->ne[2] != n_tokens ||
            norm->ne[3] != 1 || !ggml_is_contiguous(norm) || ggml_cuda_gdn_unwrap_view(norm->src[0]) != silu) {
        GDN_REPLAY_DECLINE();
    }

    // Declared to the allocation plan: silu and norm are written; the inputs, the two index tensors and the conv weight
    // are read at the LAST node's slot, so they must live until then. The state and pack tables are read and written per
    // channel by the same thread (program order) -- not a cross-CTA hazard -- and are persistent (never allocated here).
    if (!ggml_cuda_fused_ranges_ok({ silu, norm }, { qkv, g_state->src[1], conv_eff->src[1], conv->src[1] },
                                   "gdn_conv_replay")) {
        GDN_REPLAY_DECLINE();
    }

    ggml_cuda_gdn_conv_replay_args args = {};
    args.state            = (const char *) g_state->src[0]->data;
    args.pack             = (const char *) g_pack->src[0]->data;
    args.s_row            = (const int32_t *) g_state->src[1]->data;
    args.state_row_stride = g_state->src[0]->nb[1];
    args.pack_row_stride  = g_pack->src[0]->nb[1];
    args.conv_idx         = (const int32_t *) conv_eff->src[1]->data;
    args.qkv              = (const char *) qkv->data;
    args.qkv_nb1          = qkv->nb[1];
    args.state_dst        = (float *) commit->src[1]->data;
    args.pack_dst         = (float *) park->src[1]->data;
    args.pack_dst_nb1     = park->src[1]->nb[1];
    args.conv_weight      = (const char *) conv->src[1]->data;
    args.conv_weight_nb1  = conv->src[1]->nb[1];
    args.conv_silu        = (float *) silu->data;
    args.qk_norm          = (float *) norm->data;
    memcpy(&args.eps, norm->op_params, sizeof(args.eps));
    args.bias0            = 0.0f;

    const int grid = channels / 128;
    cudaStream_t st = cuda_ctx->stream();
    switch (n_tokens) {
        case  1: k_gdn_conv_replay< 1><<<grid, 128, 0, st>>>(args); break;
        case  2: k_gdn_conv_replay< 2><<<grid, 128, 0, st>>>(args); break;
        case  3: k_gdn_conv_replay< 3><<<grid, 128, 0, st>>>(args); break;
        case  4: k_gdn_conv_replay< 4><<<grid, 128, 0, st>>>(args); break;
        case  5: k_gdn_conv_replay< 5><<<grid, 128, 0, st>>>(args); break;
        case  6: k_gdn_conv_replay< 6><<<grid, 128, 0, st>>>(args); break;
        case  7: k_gdn_conv_replay< 7><<<grid, 128, 0, st>>>(args); break;
        case  8: k_gdn_conv_replay< 8><<<grid, 128, 0, st>>>(args); break;
        case  9: k_gdn_conv_replay< 9><<<grid, 128, 0, st>>>(args); break;
        case 10: k_gdn_conv_replay<10><<<grid, 128, 0, st>>>(args); break;
        case 12: k_gdn_conv_replay<12><<<grid, 128, 0, st>>>(args); break;
        case 16: k_gdn_conv_replay<16><<<grid, 128, 0, st>>>(args); break;
        default: GGML_ABORT("GDN conv replay: unhandled n_tokens %d (the matcher should have declined)", n_tokens);
    }
    return index - i;
#undef GDN_REPLAY_DECLINE
}

static int ggml_cuda_try_fuse_gdn_alpha_beta_prologue(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    constexpr int n_embd = 5120;
    constexpr int rank = 48;
    constexpr int n_tokens = 5;

    const ggml_tensor * alpha_mm = cgraph->nodes[i];
    if (alpha_mm->op != GGML_OP_MUL_MAT || alpha_mm->type != GGML_TYPE_F32 ||
            alpha_mm->src[0]->type != GGML_TYPE_BF16 || alpha_mm->src[1]->type != GGML_TYPE_F32 ||
            alpha_mm->src[0]->ne[0] != n_embd || alpha_mm->src[0]->ne[1] != rank ||
            alpha_mm->src[1]->ne[0] != n_embd || alpha_mm->src[1]->ne[1] != n_tokens ||
            alpha_mm->ne[0] != rank || alpha_mm->ne[1] != n_tokens ||
            alpha_mm->src[0]->nb[0] != sizeof(ggml_bf16_t) ||
            alpha_mm->src[0]->nb[1] != n_embd * sizeof(ggml_bf16_t) ||
            !ggml_is_contiguous(alpha_mm->src[1]) || !ggml_is_contiguous(alpha_mm)) {
        return 0;
    }

    int index = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_ADD) {
        return 0;
    }
    const ggml_tensor * add = cgraph->nodes[index];
    const ggml_tensor * bias = nullptr;
    if (ggml_cuda_gdn_unwrap_view(add->src[0]) == alpha_mm) {
        bias = add->src[1];
    } else if (ggml_cuda_gdn_unwrap_view(add->src[1]) == alpha_mm) {
        bias = add->src[0];
    } else {
        return 0;
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[index]) != GGML_UNARY_OP_SOFTPLUS ||
            cgraph->nodes[index]->src[0] != add) {
        return 0;
    }
    const ggml_tensor * softplus = cgraph->nodes[index];

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_MUL) {
        return 0;
    }
    const ggml_tensor * gate = cgraph->nodes[index];
    const ggml_tensor * a = nullptr;
    if (gate->src[0] == softplus) {
        a = gate->src[1];
    } else if (gate->src[1] == softplus) {
        a = gate->src[0];
    } else {
        return 0;
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_tensor * beta_mm = cgraph->nodes[index];
    if (beta_mm->type != GGML_TYPE_F32 || beta_mm->src[0]->type != GGML_TYPE_BF16 ||
            beta_mm->src[1] != alpha_mm->src[1] ||
            !ggml_are_same_shape(beta_mm->src[0], alpha_mm->src[0]) ||
            !ggml_are_same_shape(beta_mm, alpha_mm) ||
            beta_mm->src[0]->nb[0] != sizeof(ggml_bf16_t) ||
            beta_mm->src[0]->nb[1] != n_embd * sizeof(ggml_bf16_t)) {
        return 0;
    }

    index = ggml_cuda_gdn_next_compute(cgraph, index + 1);
    if (index >= cgraph->n_nodes || cgraph->nodes[index]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[index]) != GGML_UNARY_OP_SIGMOID ||
            ggml_cuda_gdn_unwrap_view(cgraph->nodes[index]->src[0]) != beta_mm) {
        return 0;
    }
    const ggml_tensor * beta = cgraph->nodes[index];
    if (bias->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 ||
            ggml_nelements(bias) != rank || ggml_nelements(a) != rank ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(a) ||
            gate->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(gate) || !ggml_is_contiguous(beta)) {
        return 0;
    }

    if (!ggml_cuda_fused_ranges_ok({ gate, beta },
                                   { alpha_mm->src[1], alpha_mm->src[0], beta_mm->src[0], bias, a }, "gdn_alpha_beta_prologue")) {
        return 0;
    }
    k_gdn_alpha_beta_prologue<<<rank, 256, 0, cuda_ctx->stream()>>>(
        (const nv_bfloat16 *) alpha_mm->src[0]->data,
        (const nv_bfloat16 *) beta_mm->src[0]->data,
        (const float *) alpha_mm->src[1]->data,
        (const float *) bias->data,
        (const float *) a->data,
        (float *) gate->data,
        (float *) beta->data);
    return index - i;
}

// (A) up to 8 consecutive, independent f32 CPY nodes with identical src/dst layouts -> one launch.
//     In the DeltaNet block these are the K = n_rs_seq + 1 conv-state windows written into the rollback slots.
struct ggml_cuda_multi_cpy_args {
    const char * src[8];
    char *       dst[8];
    int64_t ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03;   // src logical shape and byte strides
    int64_t ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13;   // dst logical shape and byte strides
    int64_t ne;                                               // elements per copy
    int     n;                                                // copies
};
static __global__ void k_multi_cpy_f32(const ggml_cuda_multi_cpy_args args) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= args.ne * args.n) {
        return;
    }
    const int     j = (int) (gi / args.ne);
    const int64_t i = gi - (int64_t) j * args.ne;
    // identical index math to cpy_flt in cpy.cu
    const int64_t i03 = i / (args.ne00*args.ne01*args.ne02);
    const int64_t i02 = (i - i03*args.ne00*args.ne01*args.ne02) / (args.ne00*args.ne01);
    const int64_t i01 = (i - i03*args.ne00*args.ne01*args.ne02 - i02*args.ne01*args.ne00) / args.ne00;
    const int64_t i00 = i - i03*args.ne00*args.ne01*args.ne02 - i02*args.ne01*args.ne00 - i01*args.ne00;
    const int64_t x_offset = i00*args.nb00 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03;
    const int64_t i13 = i / (args.ne10*args.ne11*args.ne12);
    const int64_t i12 = (i - i13*args.ne10*args.ne11*args.ne12) / (args.ne10*args.ne11);
    const int64_t i11 = (i - i13*args.ne10*args.ne11*args.ne12 - i12*args.ne10*args.ne11) / args.ne10;
    const int64_t i10 = i - i13*args.ne10*args.ne11*args.ne12 - i12*args.ne10*args.ne11 - i11*args.ne10;
    const int64_t dst_offset = i10*args.nb10 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13;
    *(float *) (args.dst[j] + dst_offset) = *(const float *) (args.src[j] + x_offset);
}
// returns the number of extra CPY nodes fused (0 = no fusion)
static int ggml_cuda_try_fuse_multi_cpy(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_CPY || node->type != GGML_TYPE_F32 || node->src[0]->type != GGML_TYPE_F32) {
        return 0;
    }
    // The copies are interleaved with the VIEW nodes that define the next copy's src/dst; views launch nothing,
    // so they are stepped over and included in the returned skip count.
    int idx[8] = { i };
    int n = 1, last = i, k = i + 1;
    while (n < 8 && k < cgraph->n_nodes) {
        const ggml_tensor * c = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(c)) { k++; continue; }
        if (c->op != GGML_OP_CPY || c->type != GGML_TYPE_F32 || c->src[0]->type != GGML_TYPE_F32) break;
        if (!ggml_are_same_layout(c, node) || !ggml_are_same_layout(c->src[0], node->src[0])) break;
        if (c->src[0]->view_src != nullptr && c->src[0]->view_src == c->view_src) break;   // src aliases dst tensor
        if ((c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) break;
        idx[n++] = k;
        last = k;
        k++;
    }
    if (n < 2) {
        return 0;
    }
    // Separate CPY launches are ordered by the stream, so copy j may legally read what copy j-1 wrote. Merged into
    // one kernel they run concurrently, which would race. Only fuse when every source range is disjoint from every
    // destination range (true for the DeltaNet conv-state slots, but never assumed).
    for (int a = 0; a < n; ++a) {
        const ggml_tensor * sa = cgraph->nodes[idx[a]]->src[0];
        const char * s_lo = (const char *) sa->data;
        const char * s_hi = s_lo + ggml_nbytes(sa);
        for (int b = 0; b < n; ++b) {
            const ggml_tensor * db = cgraph->nodes[idx[b]];
            const char * d_lo = (const char *) db->data;
            const char * d_hi = d_lo + ggml_nbytes(db);
            if (s_lo < d_hi && d_lo < s_hi) {
                return 0;   // overlap: fall back to the individual kernels
            }
        }
    }
    ggml_cuda_multi_cpy_args args = {};
    for (int j = 0; j < n; ++j) {
        args.src[j] = (const char *) cgraph->nodes[idx[j]]->src[0]->data;
        args.dst[j] = (char *) cgraph->nodes[idx[j]]->data;
    }
    const ggml_tensor * s = node->src[0];
    args.ne00 = s->ne[0]; args.ne01 = s->ne[1]; args.ne02 = s->ne[2]; args.ne03 = s->ne[3];
    args.nb00 = s->nb[0]; args.nb01 = s->nb[1]; args.nb02 = s->nb[2]; args.nb03 = s->nb[3];
    args.ne10 = node->ne[0]; args.ne11 = node->ne[1]; args.ne12 = node->ne[2]; args.ne13 = node->ne[3];
    args.nb10 = node->nb[0]; args.nb11 = node->nb[1]; args.nb12 = node->nb[2]; args.nb13 = node->nb[3];
    args.ne = ggml_nelements(node);
    args.n  = n;
    const int64_t total  = args.ne * n;
    const int     blocks = (int) ((total + 255) / 256);
    k_multi_cpy_f32<<<blocks, 256, 0, cuda_ctx->stream()>>>(args);
    return last - i;
}

// (B) ADD(x, bias) -> SOFTPLUS -> MUL(., a) with bias/a broadcast along ne0 (the DeltaNet decay gate) -> one kernel.
static __global__ void k_add_softplus_mul_f32(const float * __restrict__ x, const float * __restrict__ bias,
                                              const float * __restrict__ a, float * __restrict__ dst,
                                              const int64_t ne0, const int64_t ne) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    const int64_t i0 = i % ne0;
    const float s = x[i] + bias[i0];                          // op_add
    const float p = (s > 20.0f) ? s : logf(1.0f + expf(s));   // op_softplus, verbatim from unary.cu
    dst[i] = p * a[i0];                                       // op_mul
}
static bool ggml_cuda_try_fuse_add_softplus_mul(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    // ggml_cuda_can_fuse() only knows its own hard-coded patterns (it ends in `return false`), so use the generic
    // ggml check: consecutive ops, single-use intermediates, each node consuming the previous one, same shapes.
    static const ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
    if (!ggml_can_fuse(cgraph, i, ops, 3)) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * sp  = cgraph->nodes[i + 1];
    const ggml_tensor * mul = cgraph->nodes[i + 2];
    const ggml_tensor * x = add->src[0], * bias = add->src[1], * a = mul->src[1];
    if (sp->src[0] != add || mul->src[0] != sp) return false;
    if (sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) return false;
    for (const ggml_tensor * t : { add, sp, mul, x, bias, a }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    }
    if (!ggml_are_same_shape(add, x) || !ggml_are_same_shape(mul, add) || !ggml_are_same_shape(sp, add)) return false;
    // Every thread reads bias[i%ne0] and a[i%ne0] while other threads write dst, so those two must not overlap dst.
    // (x may alias dst: a thread reads and writes only its own index.) In this graph they are weights, but the
    // separate kernels this replaces were ordered by the stream and would have tolerated the overlap.
    {
        const char * d_lo = (const char *) mul->data;
        const char * d_hi = d_lo + ggml_nbytes(mul);
        for (const ggml_tensor * t : { bias, a }) {
            const char * t_lo = (const char *) t->data;
            const char * t_hi = t_lo + ggml_nbytes(t);
            if (t_lo < d_hi && d_lo < t_hi) return false;
        }
    }
    const int64_t ne0 = add->ne[0];
    if (bias->ne[0] != ne0 || ggml_nelements(bias) != ne0 || a->ne[0] != ne0 || ggml_nelements(a) != ne0) return false;
    const int64_t ne = ggml_nelements(add);
    const int blocks = (int) ((ne + 255) / 256);
    k_add_softplus_mul_f32<<<blocks, 256, 0, cuda_ctx->stream()>>>(
        (const float *) x->data, (const float *) bias->data, (const float *) a->data, (float *) mul->data, ne0, ne);
    return true;
}

// (C) RMS_NORM -> MUL(w) -> [views] -> SILU(z) -> MUL(., silu)  => one launch (Gated-DeltaNet output norm). Requires the
//     z projection to be expanded before the norm in the model graph (src/models/qwen35.cpp does this), so that no
//     compute node sits between the four. Returns the number of nodes to skip (0 = no fusion).
static int ggml_cuda_try_fuse_rms_norm_gated(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || i + 3 >= n) return 0;
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) return 0;
    int k = i + 2;
    while (k < n && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) k++;
    if (k >= n) return 0;
    const ggml_tensor * silu = cgraph->nodes[k];
    if (silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) return 0;
    int m = k + 1;
    while (m < n && ggml_cuda_is_view_or_noop(cgraph->nodes[m])) m++;
    if (m >= n) return 0;
    const ggml_tensor * out = cgraph->nodes[m];
    if (out->op != GGML_OP_MUL) return 0;
    if (!((out->src[0] == mul && out->src[1] == silu) || (out->src[0] == silu && out->src[1] == mul))) return 0;
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1) || !ggml_node_has_n_uses(cgraph, k, 1)) return 0;
    for (int idx : { i, i + 1, k }) {
        if ((cgraph->nodes[idx]->flags & GGML_TENSOR_FLAG_OUTPUT) || (cgraph->nodes[idx]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return 0;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * z = silu->src[0];
    for (const ggml_tensor * t : { norm, mul, silu, out, x, w, z }) {
        if (t->type != GGML_TYPE_F32) return 0;
    }
    if (!ggml_is_contiguous(out) || !ggml_is_contiguous(z) || !ggml_are_same_shape(z, out) ||
        !ggml_are_same_shape(norm, out) || !ggml_are_same_shape(mul, out) || !ggml_are_same_shape(silu, out)) return 0;
    if (!ggml_is_contiguous_rows(x) || w->nb[0] != sizeof(float) || !ggml_can_repeat(w, norm)) return 0;
    if (!ggml_cuda_fused_ranges_ok({ out }, { x, w, z }, "rms_norm_gated")) return 0;
    ggml_cuda_op_rms_norm_fused_gate(*cuda_ctx, cgraph->nodes[i], cgraph->nodes[i + 1], cgraph->nodes[k], cgraph->nodes[m]);
    return m - i;
}

// try and fuse nodes and return the number of nodes to skip
// A q8_1 twin slot for the rotated activation `f32` (rows of ne0 values) that a signed-Hadamard kernel is about to write
// (common.cuh ptq1_q8_twin), or -1 when it is not a verify-width PTQ1_0 input: <= 16 rows (the tensor-core range, which reads
// the ptq1_perm layout) and a row length the matmul's quantizer does not pad. The slot is registered by
// ggml_cuda_ptq1_twin_commit only after the kernel launched.
static int ggml_cuda_ptq1_twin_slot(const ggml_backend_cuda_context & ctx, const ggml_tensor * f32, const int64_t ne0) {
    const int64_t ne = ggml_nelements(f32);
    if (f32->type != GGML_TYPE_F32 || !ggml_is_contiguous(f32) || ne0 <= 0 || ne0 % 512 != 0 || ne % ne0 != 0 ||
            ne / ne0 > 16 || (size_t) (ne / QK8_1) * sizeof(block_q8_1) > ggml_backend_cuda_context::PTQ1_Q8_TWIN_BYTES ||
            ctx.ptq1_q8_twins[ctx.ptq1_q8_twin_next].q8 == nullptr) {
        return -1;
    }
    return ctx.ptq1_q8_twin_next;
}

static void ggml_cuda_ptq1_twin_commit(ggml_backend_cuda_context & ctx, const int slot, const ggml_tensor * f32, const int64_t ne0) {
    auto & tw = ctx.ptq1_q8_twins[slot];
    tw.f32   = f32->data;
    tw.bytes = ggml_nbytes(f32);
    tw.ne0   = ne0;
    tw.nrows = ggml_nelements(f32) / ne0;
    ctx.ptq1_q8_twin_next  = (slot + 1) % ggml_backend_cuda_context::PTQ1_Q8_TWINS;
    ctx.ptq1_q8_twin_fresh = slot;
}

// Whatever nodes i0..i1 write kills every twin it overlaps (the allocator hands a dead activation's bytes to later
// tensors), except the twin those same nodes just produced. Views and no-ops write nothing.
static void ggml_cuda_ptq1_twin_invalidate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i0, const int i1) {
    for (int j = i0; j <= i1; j++) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        const char * t0 = (const char *) t->data;
        const char * t1 = t0 + ggml_nbytes(t);
        for (int s = 0; s < ggml_backend_cuda_context::PTQ1_Q8_TWINS; s++) {
            auto & tw = ctx.ptq1_q8_twins[s];
            if (s != ctx.ptq1_q8_twin_fresh && tw.f32 && t0 < (const char *) tw.f32 + tw.bytes && (const char *) tw.f32 < t1) {
                tw.f32 = nullptr;
            }
        }
    }
}

static int ggml_cuda_try_fuse_gdn_out_chain(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || i + 3 >= n) {
        return 0;
    }
    // Follow the chain across view and no-op nodes.
    const int j1 = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j1 >= n) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[j1];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) {
        return 0;
    }
    const int k = ggml_cuda_gdn_next_compute(cgraph, j1 + 1);
    if (k >= n) {
        return 0;
    }
    const ggml_tensor * silu = cgraph->nodes[k];
    if (silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return 0;
    }
    const int m = ggml_cuda_gdn_next_compute(cgraph, k + 1);
    if (m >= n) {
        return 0;
    }
    const ggml_tensor * gated = cgraph->nodes[m];
    if (gated->op != GGML_OP_MUL ||
            !((gated->src[0] == mul && gated->src[1] == silu) || (gated->src[0] == silu && gated->src[1] == mul))) {
        return 0;
    }

    const auto internal = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    if (!internal(i) || !internal(j1) || !internal(k) || !internal(m)) {
        return 0;
    }

    int p = m + 1;
    const ggml_tensor * prev = gated;
    const ggml_tensor * perm = nullptr;
    while (p < n && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
        const ggml_tensor * view = cgraph->nodes[p];
        if (view->src[0] != prev || !internal(p)) {
            return 0;
        }
        if (view->op == GGML_OP_PERMUTE) {
            perm = view;
        }
        prev = view;
        ++p;
    }
    if (p >= n || cgraph->nodes[p]->op != GGML_OP_CONT || cgraph->nodes[p]->src[0] != prev ||
            perm == nullptr || !internal(p)) {
        return 0;
    }
    const ggml_tensor * cont = cgraph->nodes[p++];

    prev = cont;
    while (p < n && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
        const ggml_tensor * view = cgraph->nodes[p];
        if (view->src[0] != prev || !internal(p)) {
            return 0;
        }
        prev = view;
        ++p;
    }
    if (p >= n || cgraph->nodes[p]->op != GGML_OP_MUL || cgraph->nodes[p]->src[0] != prev || !internal(p)) {
        return 0;
    }
    const ggml_tensor * signed_input = cgraph->nodes[p++];
    const ggml_tensor * signs        = signed_input->src[1];

    prev = signed_input;
    while (p < n && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
        const ggml_tensor * view = cgraph->nodes[p];
        if (view->src[0] != prev || !internal(p)) {
            return 0;
        }
        prev = view;
        ++p;
    }
    if (p >= n) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[p];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != prev ||
            ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }

    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * z = silu->src[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 ||
            signs->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            x->ne[0] != 128 || x->ne[1] != 48 || x->ne[2] < 1 || x->ne[2] > 16 || x->ne[3] != 1 ||
            !ggml_are_same_shape(norm, x) || !ggml_are_same_shape(mul, x) ||
            !ggml_are_same_shape(silu, x) || !ggml_are_same_shape(gated, x) ||
            w->ne[0] != 128 || ggml_nrows(w) != 1 ||
            perm->ne[0] != 128 || perm->ne[1] != 3 || perm->ne[2] != 16 || perm->ne[3] != x->ne[2] ||
            perm->nb[0] != sizeof(float) || perm->nb[1] != 128*16*sizeof(float) ||
            perm->nb[2] != 128*sizeof(float) || perm->nb[3] != 128*48*sizeof(float) ||
            !ggml_is_contiguous(cont) || cont->ne[0] != 128 || cont->ne[1] != 3 || cont->ne[2] != 16 || cont->ne[3] != x->ne[2] ||
            signs->ne[0] != 6144 || ggml_nrows(signs) != 1 ||
            mm->src[0]->ne[0] != 1024 || mm->src[0]->ne[1] != 1024 || ggml_nrows(mm->src[0]) != 1024 ||
            mm->ne[0] != 1024 || ggml_nelements(mm) != ggml_nelements(x)) {
        return 0;
    }

    if (!ggml_cuda_fused_ranges_ok({ mm }, { x, w, z, signs }, "gdn_out_chain")) {
        return 0;
    }

    const int slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, 6144);
    if (!ggml_cuda_op_gdn_out_fwht(*cuda_ctx, norm, mul, silu, signs, mm,
            slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr)) {
        return 0;
    }
    if (slot >= 0) {
        ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, 6144);
    }
    return p - i;
}

// Fuse the ordered-head GDN output chain through its signed Hadamard.
static int ggml_cuda_try_fuse_gdn_out_direct(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || i + 3 >= n) {
        return 0;
    }
    const ggml_tensor * mul   = cgraph->nodes[i + 1];
    const ggml_tensor * silu  = cgraph->nodes[i + 2];
    const ggml_tensor * gated = cgraph->nodes[i + 3];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm) ||
            silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            gated->op != GGML_OP_MUL ||
            !((gated->src[0] == mul && gated->src[1] == silu) || (gated->src[0] == silu && gated->src[1] == mul))) {
        return 0;
    }
    const auto internal = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    if (!internal(i) || !internal(i + 1) || !internal(i + 2) || !internal(i + 3)) {
        return 0;
    }

    int p = i + 4;
    const ggml_tensor * prev = gated;
    while (p < n && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
        const ggml_tensor * view = cgraph->nodes[p];
        if (view->src[0] != prev || !internal(p) || view->op == GGML_OP_PERMUTE || view->op == GGML_OP_TRANSPOSE) {
            return 0;   // heads must stay in order: reshapes only
        }
        prev = view;
        ++p;
    }
    if (p >= n || cgraph->nodes[p]->op != GGML_OP_MUL || cgraph->nodes[p]->src[0] != prev || !internal(p)) {
        return 0;
    }
    const ggml_tensor * signed_input = cgraph->nodes[p++];
    const ggml_tensor * signs        = signed_input->src[1];

    prev = signed_input;
    while (p < n && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
        const ggml_tensor * view = cgraph->nodes[p];
        if (view->src[0] != prev || !internal(p) || view->op == GGML_OP_PERMUTE || view->op == GGML_OP_TRANSPOSE) {
            return 0;
        }
        prev = view;
        ++p;
    }
    if (p >= n) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[p];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != prev || ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }

    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * z = silu->src[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 ||
            signs->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            x->ne[0] != 128 || x->ne[1] != 48 || x->ne[2] < 1 || x->ne[2] > 16 || x->ne[3] != 1 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(z) ||
            !ggml_are_same_shape(norm, x) || !ggml_are_same_shape(mul, x) ||
            !ggml_are_same_shape(silu, x) || !ggml_are_same_shape(gated, x) || !ggml_are_same_shape(z, x) ||
            w->ne[0] != 128 || ggml_nrows(w) != 1 ||
            signs->ne[0] != 6144 || ggml_nrows(signs) != 1 ||
            mm->src[0]->ne[0] != 1024 || mm->src[0]->ne[1] != 1024 || ggml_nrows(mm->src[0]) != 1024 ||
            mm->ne[0] != 1024 || ggml_nelements(mm) != ggml_nelements(x)) {
        return 0;
    }

    if (!ggml_cuda_fused_ranges_ok({ mm }, { x, w, z, signs }, "gdn_out_direct")) {
        return 0;
    }

    const int slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, 6144);
    if (!ggml_cuda_op_gdn_out_fwht(*cuda_ctx, norm, mul, silu, signs, mm,
            slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr, /*permuted=*/ false)) {
        return 0;
    }
    if (slot >= 0) {
        ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, 6144);
    }
    return p - i;
}

// Fuse the attention Q chain RMS_NORM -> MUL(w) -> ROPE -> [RESHAPE] -> MUL_MAT(Hadamard hint) ->
// [RESHAPE] -> XYZKV_WHT as one launch. Only the XYZKV_WHT result is written.
static int ggml_cuda_try_fuse_attn_q_chain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || norm->type != GGML_TYPE_F32 || norm->ne[0] != 256 || norm->ne[2] > 16 ||
            norm->ne[3] != 1) {
        return 0;
    }
    const auto single = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    int j_mul = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j_mul >= n) { return 0; }
    const ggml_tensor * mul = cgraph->nodes[j_mul];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) { return 0; }
    int j_rope = ggml_cuda_gdn_next_compute(cgraph, j_mul + 1);
    if (j_rope >= n) { return 0; }
    const ggml_tensor * rope = cgraph->nodes[j_rope];
    if (rope->op != GGML_OP_ROPE || rope->src[0] != mul) { return 0; }
    int j_had = ggml_cuda_gdn_next_compute(cgraph, j_rope + 1);
    if (j_had >= n) { return 0; }
    const ggml_tensor * had = cgraph->nodes[j_had];
    if (had->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(had, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
            ggml_cuda_gdn_unwrap_view(had->src[1]) != rope) {
        return 0;
    }
    int j_xyzkv = ggml_cuda_gdn_next_compute(cgraph, j_had + 1);
    if (j_xyzkv >= n) { return 0; }
    ggml_tensor * xyzkv = cgraph->nodes[j_xyzkv];
    if (xyzkv->op != GGML_OP_XYZKV_WHT || ggml_cuda_gdn_unwrap_view(xyzkv->src[0]) != had) { return 0; }
    // every node from the norm up to the xyzkv WHT is a link of this chain, used once (views included)
    for (int k = i; k < j_xyzkv; ++k) {
        if (!single(k)) {
            return 0;
        }
    }
    if (!ggml_cuda_fused_ranges_ok({ xyzkv }, { norm->src[0], mul->src[0] == norm ? mul->src[1] : mul->src[0], rope->src[1],
                                              xyzkv->src[1] }, "attn_q_chain")) {
        return 0;
    }
    if (!ggml_cuda_op_attn_q_chain(*cuda_ctx, norm, mul, rope, had, xyzkv)) {
        return 0;
    }
    return j_xyzkv - i;
}

// Fuse the attention output chain XYZKV_WHT(inverse) -> [RESHAPE] -> MUL_MAT(Hadamard-64 hint) ->
// [RESHAPE, VIEW(gate)] -> CONT(gate) -> SIGMOID -> MUL -> MUL(signs) -> [RESHAPE] -> MUL_MAT(1024 Hadamard hint) as one
// launch (fwht.cu ggml_cuda_op_attn_out_chain) that also writes the q8_1 twin. Every link from the xyzkv WHT up to the
// last Hadamard has exactly one consumer and is not an output -- only the last Hadamard is written. Declared to the plan
// together with the signed-Hadamard group inside it. It applies only to batches of at most 16 tokens.
static int ggml_cuda_try_fuse_attn_out_chain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const bool plan = g_ggml_cuda_fusion_plan != nullptr;
    const int n = cgraph->n_nodes;
    const ggml_tensor * xyzkv = cgraph->nodes[i];
    if (xyzkv->op != GGML_OP_XYZKV_WHT || xyzkv->type != GGML_TYPE_F32 || xyzkv->ne[0] != 256 || xyzkv->ne[2] > 16 ||
            xyzkv->ne[3] != 1) {
        return 0;
    }
    const auto single = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    const int j_had = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j_had >= n) { return 0; }
    const ggml_tensor * had = cgraph->nodes[j_had];
    if (had->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(had, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
            ggml_cuda_gdn_unwrap_view(had->src[1]) != xyzkv) {
        return 0;
    }
    const int j_cont = ggml_cuda_gdn_next_compute(cgraph, j_had + 1);
    if (j_cont >= n) { return 0; }
    const ggml_tensor * cont = cgraph->nodes[j_cont];
    if (cont->op != GGML_OP_CONT || cont->src[0] == nullptr || cont->src[0]->op != GGML_OP_VIEW) { return 0; }
    const int j_sig = ggml_cuda_gdn_next_compute(cgraph, j_cont + 1);
    if (j_sig >= n) { return 0; }
    const ggml_tensor * sig = cgraph->nodes[j_sig];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != cont) { return 0; }
    const int j_gated = ggml_cuda_gdn_next_compute(cgraph, j_sig + 1);
    if (j_gated >= n) { return 0; }
    const ggml_tensor * gated = cgraph->nodes[j_gated];
    if (gated->op != GGML_OP_MUL) { return 0; }
    const ggml_tensor * pregate = gated->src[0] == sig ? gated->src[1] : (gated->src[1] == sig ? gated->src[0] : nullptr);
    if (pregate == nullptr || ggml_cuda_gdn_unwrap_view(pregate) != had) { return 0; }
    const int j_sgn = ggml_cuda_gdn_next_compute(cgraph, j_gated + 1);
    if (j_sgn >= n) { return 0; }
    const ggml_tensor * sgn = cgraph->nodes[j_sgn];
    if (sgn->op != GGML_OP_MUL || sgn->src[0] != gated) { return 0; }
    const int j_mm = ggml_cuda_gdn_next_compute(cgraph, j_sgn + 1);
    if (j_mm >= n) { return 0; }
    ggml_tensor * mm = cgraph->nodes[j_mm];
    if (mm->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
            ggml_cuda_gdn_unwrap_view(mm->src[1]) != sgn) {
        return 0;
    }
    // every node from the xyzkv WHT up to the last Hadamard is a link of this chain, used once (views included)
    for (int k = i; k < j_mm; ++k) {
        if (!single(k)) {
            return 0;
        }
    }
    const ggml_tensor * gate  = cont->src[0];
    const ggml_tensor * signs = sgn->src[1];
    if (plan) {
        ggml_cuda_fused_ranges_ok({ mm }, { gated, signs }, "signed_fwht");   // the bit-off arm's group, see above
    }
    if (!ggml_cuda_fused_ranges_ok({ mm }, { xyzkv->src[0], xyzkv->src[1], ggml_cuda_gdn_unwrap_view(gate), signs },
                                   "attn_out_chain")) {
        return 0;
    }
    const int slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, 6144);
    if (!ggml_cuda_op_attn_out_chain(*cuda_ctx, xyzkv, had, gate, signs, mm,
            slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr)) {
        return 0;
    }
    if (slot >= 0) {
        ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, 6144);
    }
    return j_mm - i;
}

// Fuse the device KQ mask chain REPEAT -> SUB -> NEG ->
// STEP -> LOG -> CPY(f16) as one launch (unary.cu ggml_cuda_op_kq_mask_chain). Every link before the CPY has one consumer
// and is not an output. The sliding-window SCALE_BIAS/STEP/MUL form declines.
static int ggml_cuda_try_fuse_kq_mask_chain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * repeat = cgraph->nodes[i];
    if (repeat->op != GGML_OP_REPEAT || repeat->type != GGML_TYPE_F32) {
        return 0;
    }
    const auto single = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    const int j_sub = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j_sub >= n) { return 0; }
    const ggml_tensor * sub = cgraph->nodes[j_sub];
    if (sub->op != GGML_OP_SUB || sub->src[0] != repeat) { return 0; }
    const int j_neg = ggml_cuda_gdn_next_compute(cgraph, j_sub + 1);
    if (j_neg >= n) { return 0; }
    const ggml_tensor * neg = cgraph->nodes[j_neg];
    if (neg->op != GGML_OP_UNARY || ggml_get_unary_op(neg) != GGML_UNARY_OP_NEG || neg->src[0] != sub) { return 0; }
    const int j_step = ggml_cuda_gdn_next_compute(cgraph, j_neg + 1);
    if (j_step >= n) { return 0; }
    const ggml_tensor * step = cgraph->nodes[j_step];
    if (step->op != GGML_OP_UNARY || ggml_get_unary_op(step) != GGML_UNARY_OP_STEP || step->src[0] != neg) { return 0; }
    const int j_log = ggml_cuda_gdn_next_compute(cgraph, j_step + 1);
    if (j_log >= n) { return 0; }
    const ggml_tensor * lg = cgraph->nodes[j_log];
    if (lg->op != GGML_OP_LOG || lg->src[0] != step) { return 0; }
    const int j_cpy = ggml_cuda_gdn_next_compute(cgraph, j_log + 1);
    if (j_cpy >= n) { return 0; }
    ggml_tensor * cpy = cgraph->nodes[j_cpy];
    if (cpy->op != GGML_OP_CPY || cpy->src[0] != lg || cpy->type != GGML_TYPE_F16) { return 0; }
    for (int k = i; k < j_cpy; ++k) {
        if (!single(k)) {
            return 0;
        }
    }
    if (!ggml_cuda_fused_ranges_ok({ cpy }, { repeat->src[0], sub->src[1] }, "kq_mask_chain")) {
        return 0;
    }
    if (!ggml_cuda_op_kq_mask_chain(*cuda_ctx, repeat, sub, cpy)) {
        return 0;
    }
    return j_cpy - i;
}

// Fuse each attention cache write into one launch:
//   K: RMS_NORM -> MUL -> ROPE -> [RESHAPE] -> MUL_MAT(Hadamard-256 hint) -> [RESHAPE, VIEW] -> SET_ROWS(xyzkv2)
//   V: MUL_MAT(Hadamard-64 hint) -> [RESHAPE, VIEW] -> SET_ROWS(xyzkv2)
// Every link before the SET_ROWS has one consumer and is not an output; the SET_ROWS reads the Hadamard's own bytes
// (a reshape/view of it). It applies to batches of at most 16 tokens.
static int ggml_cuda_try_fuse_attn_k_write(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || norm->type != GGML_TYPE_F32 || norm->ne[0] != 256 || norm->ne[2] > 16 ||
            norm->ne[3] != 1) {
        return 0;
    }
    const auto single = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    const int j_mul = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j_mul >= n) { return 0; }
    const ggml_tensor * mul = cgraph->nodes[j_mul];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) { return 0; }
    const int j_rope = ggml_cuda_gdn_next_compute(cgraph, j_mul + 1);
    if (j_rope >= n) { return 0; }
    const ggml_tensor * rope = cgraph->nodes[j_rope];
    if (rope->op != GGML_OP_ROPE || rope->src[0] != mul) { return 0; }
    const int j_had = ggml_cuda_gdn_next_compute(cgraph, j_rope + 1);
    if (j_had >= n) { return 0; }
    const ggml_tensor * had = cgraph->nodes[j_had];
    if (had->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(had, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
            ggml_cuda_gdn_unwrap_view(had->src[1]) != rope) {
        return 0;
    }
    const int j_sr = ggml_cuda_gdn_next_compute(cgraph, j_had + 1);
    if (j_sr >= n) { return 0; }
    ggml_tensor * sr = cgraph->nodes[j_sr];
    if (sr->op != GGML_OP_SET_ROWS || sr->type != GGML_TYPE_XYZKV2_0 || ggml_cuda_gdn_unwrap_view(sr->src[0]) != had ||
            sr->src[0]->data != had->data || !ggml_is_contiguous(sr->src[0]) || !ggml_is_contiguous(had)) {
        return 0;
    }
    for (int k = i; k < j_sr; ++k) {
        if (!single(k)) {
            return 0;
        }
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (!ggml_cuda_fused_ranges_ok({ sr }, { norm->src[0], w, rope->src[1], sr->src[1] }, "attn_k_write")) {
        return 0;
    }
    if (!ggml_cuda_op_attn_k_write(*cuda_ctx, norm, mul, rope, had, sr)) {
        return 0;
    }
    return j_sr - i;
}

static int ggml_cuda_try_fuse_attn_v_write(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * had = cgraph->nodes[i];
    if (had->op != GGML_OP_MUL_MAT || had->type != GGML_TYPE_F32 || had->ne[0] != 64 ||
            ggml_get_op_params_i32(had, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }
    const ggml_tensor * v = ggml_cuda_gdn_unwrap_view(had->src[1]);   // the V projection; the Hadamard reads it as-is
    if (v == nullptr || !ggml_is_contiguous(had->src[1]) || had->src[1]->data != v->data || !ggml_is_contiguous(had)) {
        return 0;
    }
    const auto single = [&](int index) {
        const ggml_tensor * t = cgraph->nodes[index];
        return ggml_node_get_use_count(cgraph, index) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    const int j_sr = ggml_cuda_gdn_next_compute(cgraph, i + 1);
    if (j_sr >= n) { return 0; }
    ggml_tensor * sr = cgraph->nodes[j_sr];
    if (sr->op != GGML_OP_SET_ROWS || sr->type != GGML_TYPE_XYZKV2_0 || ggml_cuda_gdn_unwrap_view(sr->src[0]) != had ||
            sr->src[0]->data != had->data || !ggml_is_contiguous(sr->src[0])) {
        return 0;
    }
    for (int k = i; k < j_sr; ++k) {
        if (!single(k)) {
            return 0;
        }
    }
    if (!ggml_cuda_fused_ranges_ok({ sr }, { v, sr->src[1] }, "attn_v_write")) {
        return 0;
    }
    if (!ggml_cuda_op_attn_v_write(*cuda_ctx, v, had, sr)) {
        return 0;
    }
    return j_sr - i;
}

static int ggml_cuda_try_fuse_ptq1_sibling(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    // Merge three consecutive attention projections that read the same activation.
    if (node->op == GGML_OP_MUL_MAT && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT) {
        ggml_tensor * b = cgraph->nodes[i + 1];
        ggml_tensor * c = cgraph->nodes[i + 2];
        if ((b->flags & GGML_TENSOR_FLAG_COMPUTE) && (c->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_cuda_ptq1_triple_mergeable(node, b, c) &&
                ggml_cuda_fused_ranges_ok({ node, b, c }, { node->src[1], b->src[1], c->src[1] }, "ptq1_triple") &&
                ggml_cuda_mul_mat_vec_q_ptq1_triple(*cuda_ctx, node, b, c)) {
            return 2;
        }
    }
    // Sibling PTQ1_0 matmuls: [A; B] in one launch (mmvq.cu ggml_cuda_mul_mat_vec_q_ptq1_pair). The second output is
    // written a node early, so it must not share bytes with anything the first node reads or writes.
    if (node->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT) {
        ggml_tensor * b = cgraph->nodes[i + 1];
        if ((b->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_ptq1_pair_mergeable(node, b) &&
                ggml_cuda_fused_ranges_ok({ node, b }, { node->src[1], b->src[1] }, "ptq1_sibling") &&
                ggml_cuda_mul_mat_vec_q_ptq1_pair(*cuda_ctx, node, b)) {
            return 1;
        }
    }
    GGML_UNUSED(cuda_ctx);
    return 0;
}

static int ggml_cuda_try_fuse_norm_fwht(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    // Bonsai norm chain: [ADD ->] RMS_NORM -> MUL(w) -> MUL(signs) -> RESHAPE -> MUL_MAT(had_rot, Hadamard hint) in one
    // launch (fwht.cu rms_norm_mul_fwht_f32, bit-identical). 128 chains per PTQ1 verify pass, three launches each before.
    // The norm output is still written when anything besides the sign multiply reads it (the bf16 alpha/beta matvecs).
    {
        const int j = node->op == GGML_OP_ADD ? i + 1 : i;   // index of the RMS_NORM
        if ((node->op == GGML_OP_ADD || node->op == GGML_OP_RMS_NORM) && j + 4 < cgraph->n_nodes) {
            const ggml_tensor * add  = node->op == GGML_OP_ADD ? node : nullptr;
            const ggml_tensor * nrm  = cgraph->nodes[j];
            const ggml_tensor * mw   = cgraph->nodes[j + 1];
            const ggml_tensor * ms   = cgraph->nodes[j + 2];
            const ggml_tensor * rs   = cgraph->nodes[j + 3];
            ggml_tensor *       mm   = cgraph->nodes[j + 4];
            const auto          outp = [](const ggml_tensor * t) { return (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0; };
            if (nrm->op == GGML_OP_RMS_NORM && (!add || nrm->src[0] == add) &&
                    mw->op == GGML_OP_MUL && mw->src[0] == nrm && ms->op == GGML_OP_MUL && ms->src[0] == mw &&
                    rs->op == GGML_OP_RESHAPE && rs->src[0] == ms && mm->op == GGML_OP_MUL_MAT && mm->src[1] == rs &&
                    ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                    ggml_node_get_use_count(cgraph, j) == 1 && !outp(nrm) &&
                    ggml_node_get_use_count(cgraph, j + 2) == 1 && !outp(ms) &&
                    ggml_node_get_use_count(cgraph, j + 3) == 1 && !outp(rs)) {
                const bool need_norm = ggml_node_get_use_count(cgraph, j + 1) > 1 || outp(mw);
                // Fusing the ADD keeps its two inputs alive past the point where the allocator considers them dead:
                // separately, the add consumes them before anything is written, so a later output (the norm, the
                // transform) may be placed over them. Inside one kernel, a row's output landing on ANOTHER row's input
                // is a race. So the ADD is fused only when neither output touches either input; otherwise the ADD runs
                // alone and the chain is fused from the RMS_NORM, whose input (the residual) stays alive.
                // Now through the shared guard (ggml_cuda_fused_ranges_ok), which also covers the no-ADD case and two
                // outputs on each other. The ADD's own output may sit on its inputs (in-place: same element, same
                // thread), so it is only checked against the other outputs.
                const ggml_tensor * mw_out = need_norm ? mw : nullptr;
                const ggml_tensor * a_in   = add ? add->src[0] : nrm->src[0];
                const ggml_tensor * b_in   = add ? add->src[1] : nullptr;
                const bool add_ok = ggml_cuda_fused_ranges_ok({ add, mw_out, mm }, { a_in, b_in, mw->src[1], ms->src[1] },
                                                              "norm_fwht", { { add, a_in }, { add, b_in } });
                if (add_ok &&
                        [&] {
                            const ggml_tensor * xin  = add ? add : nrm->src[0];
                            const int           slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, xin->ne[0]);
                            if (!ggml_cuda_op_rms_norm_fwht(*cuda_ctx, add, nrm, mw->src[1], ms->src[1], need_norm ? (ggml_tensor *) mw : nullptr, mm,
                                    slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr)) {
                                return false;
                            }
                            if (slot >= 0) {
                                ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, xin->ne[0]);
                            }
                            return true;
                        }()) {
                    return (j + 4) - i;
                }
            }
        }
    }
    GGML_UNUSED(cuda_ctx);
    return 0;
}

static int ggml_cuda_try_fuse_glu_fwht(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    // SWIGLU -> MUL(signs) -> RESHAPE -> MUL_MAT(had_rot) (the input of ffn_down, 64 per PTQ1 verify pass): the swiglu is
    // computed in the signed Hadamard's load (fwht.cu ggml_cuda_op_fwht_signed_glu), one launch and one gap less per
    // layer. The Hadamard output may have been placed over the (dead) swiglu input -- reading one while writing the other
    // from different CTAs is a race, so an overlap declines.
    {
        if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                node->src[1] == nullptr && ggml_get_op_params_i32(node, 1) == 0 && i + 3 < cgraph->n_nodes) {
            const ggml_tensor * ml = cgraph->nodes[i + 1];
            const ggml_tensor * rs = cgraph->nodes[i + 2];
            ggml_tensor *       mm = cgraph->nodes[i + 3];
            const ggml_tensor * y  = node->src[0];
            if (ml->op == GGML_OP_MUL && ml->src[0] == node && rs->op == GGML_OP_RESHAPE && rs->src[0] == ml &&
                    mm->op == GGML_OP_MUL_MAT && mm->src[1] == rs && ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                    ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                    ggml_node_get_use_count(cgraph, i + 2) == 1 &&
                    !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(ml->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    !(rs->flags & GGML_TENSOR_FLAG_OUTPUT) && ml->src[1]->type == GGML_TYPE_F32 &&
                    ggml_are_same_shape(ml, node) && ggml_cuda_fused_ranges_ok({ mm }, { y, ml->src[1] }, "glu_fwht") &&
                    [&] {
                        const int slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, node->ne[0]);
                        if (!ggml_cuda_op_fwht_signed_glu(*cuda_ctx, y, ml->src[1], mm, slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr)) {
                            return false;
                        }
                        if (slot >= 0) {
                            ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, node->ne[0]);
                        }
                        return true;
                    }()) {
                return 3;
            }
        }
    }
    GGML_UNUSED(cuda_ctx);
    return 0;
}

// true if `consumer` is the only node that reads t (as a source or through a view of it)
static bool ggml_cuda_tensor_single_consumer(const ggml_cgraph * cgraph, const ggml_tensor * t, const ggml_tensor * consumer) {
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == consumer) {
            continue;
        }
        if (n->view_src == t) {
            return false;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t) {
                return false;
            }
        }
    }
    return true;
}

static int ggml_cuda_try_fuse_signed_fwht(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    // Fold MUL(x, signs) -> RESHAPE -> MUL_MAT(hadamard) into one signed FWHT launch.
    // Bit-exact: the signs are +-1 and the fwht scale 1/sqrt(n) a power of two, so (x*s)*scale == (x*scale)*s.
    // ggml_can_fuse cannot express it because the reshape changes the shape.
    {
        if (node->op == GGML_OP_MUL && i + 2 < cgraph->n_nodes) {
            const ggml_tensor * rs = cgraph->nodes[i + 1];
            ggml_tensor *       mm = cgraph->nodes[i + 2];
            const ggml_tensor * x  = node->src[0];
            const ggml_tensor * s  = node->src[1];
            if (rs->op == GGML_OP_RESHAPE && rs->src[0] == node && mm->op == GGML_OP_MUL_MAT && mm->src[1] == rs &&
                    ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                    ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                    !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(rs->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    x->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(x) && ggml_is_contiguous(s) && ggml_are_same_shape(x, node) &&
                    s->ne[0] == x->ne[0] && ggml_nrows(s) == 1 &&
                    // Declare the in-place MUL output as the group read so the allocator may reuse x.
                    // Runtime still validates x and declines fusion if another tensor reused its storage.
                    // Pin the fused output only when x is not a view and the MUL can reuse it in place.
                    ggml_cuda_fused_ranges_ok({ mm }, { g_ggml_cuda_fusion_plan != nullptr && x->view_src == nullptr &&
                                                        ggml_cuda_tensor_single_consumer(cgraph, x, node) ? node : x, s },
                                              "signed_fwht") &&
                    [&] {
                        const int slot = ggml_cuda_ptq1_twin_slot(*cuda_ctx, mm, x->ne[0]);
                        if (!ggml_cuda_op_fwht_signed(*cuda_ctx, x, s, mm, slot >= 0 ? cuda_ctx->ptq1_q8_twins[slot].q8 : nullptr)) {
                            return false;
                        }
                        if (slot >= 0 && mm->ne[0] >= 512) {   // fwht_dispatch writes the twin from the block kernel only
                            ggml_cuda_ptq1_twin_commit(*cuda_ctx, slot, mm, x->ne[0]);
                        }
                        return true;
                    }()) {
                return 2;
            }
        }
    }
    GGML_UNUSED(cuda_ctx);
    return 0;
}

// Every guarded fusion (ggml_cuda_fused_ranges_ok), in try_fuse's order and under the same switches. graph_optimize
// runs this in plan mode, where the guard records instead of launching. Like try_fuse, the first matcher that claims
// node i wins and the returned count skips the nodes its group spans, so the plan declares exactly the groups the
// evaluation will launch -- a dependency is inserted after its group's LAST node, and one declared for a group that
// does not fire could land inside a pattern that does and break its node adjacency.
// GDN state-direct (GET_ROWS(states, s_copy) -> RESHAPE -> GATED_DELTA_NET(state = it)): returns the GDN node whose
// initial state the gather at node i feeds, when the gather can be skipped and the kernel can read the cache row itself
// (gated_delta_net.cu state_idx); nullptr otherwise. ONE matcher for both passes -- the evaluation loop acts on it, the
// allocation plan declares it (see ggml_cuda_fusion_plan_node).
static const ggml_tensor * ggml_cuda_gdn_state_direct_target(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_GET_ROWS) {
        return nullptr;
    }
    const ggml_tensor * table = node->src[0];
    const ggml_tensor * idx   = node->src[1];
    if (!(node->type == GGML_TYPE_F32 && table->type == GGML_TYPE_F32 && idx->type == GGML_TYPE_I32 &&
            ggml_is_contiguous(idx) && ggml_nrows(idx) == 1 && idx->ne[0] == node->ne[1] &&
            table->nb[0] == sizeof(float) && table->nb[1] == ggml_row_size(GGML_TYPE_F32, table->ne[0]) &&
            node->ne[0] == table->ne[0] && node->ne[2] == 1 && node->ne[3] == 1 &&
            ggml_node_get_use_count(cgraph, i) == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT))) {
        return nullptr;
    }
    for (int r = i + 1; r < cgraph->n_nodes; ++r) {
        const ggml_tensor * rs = cgraph->nodes[r];
        if (rs->op != GGML_OP_RESHAPE || rs->src[0] != node) {
            continue;
        }
        if (ggml_node_get_use_count(cgraph, r) != 1 || (rs->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return nullptr;
        }
        for (int k = r + 1; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * gdn = cgraph->nodes[k];
            if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->src[5] != rs) {
                continue;
            }
            const ggml_tensor * v = gdn->src[2];   // [S_v, H, n_tokens, n_seqs]: rows are D = S_v*S_v*H floats
            return (table->ne[0] == v->ne[0]*v->ne[0]*v->ne[1] && v->ne[3] == idx->ne[0]) ? gdn : nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

// GDN prefix-direct: GET_ROWS(cache_pk, s_copy) -> VIEW -> GATED_DELTA_NET src[6].
// The kernel reads row s_copy[seq] of the table itself. In place is safe: this batch's pack is
// written to the op's own dst and copied into cache_pk by a LATER node (delta-net-base.cpp pack_dst), and the state_zero
// SCALE that build_rs puts in front of the gather is a node of its own, evaluated before the GDN either way. Same floats
// from the same source: bit-identical. Returns the GDN node, or nullptr. ONE matcher for the eval loop and the plan.
static const ggml_tensor * ggml_cuda_gdn_prefix_direct_target(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_GET_ROWS) {
        return nullptr;
    }
    const ggml_tensor * table = node->src[0];
    const ggml_tensor * idx   = node->src[1];
    if (!(node->type == GGML_TYPE_F32 && table->type == GGML_TYPE_F32 && idx->type == GGML_TYPE_I32 &&
            ggml_is_contiguous(idx) && ggml_nrows(idx) == 1 && idx->ne[0] == node->ne[1] &&
            table->nb[0] == sizeof(float) && table->nb[1] == ggml_row_size(GGML_TYPE_F32, table->ne[0]) &&
            node->ne[0] == table->ne[0] && node->ne[2] == 1 && node->ne[3] == 1 && ggml_is_contiguous(node) &&
            ggml_node_get_use_count(cgraph, i) == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT))) {
        return nullptr;
    }
    for (int r = i + 1; r < cgraph->n_nodes; ++r) {
        const ggml_tensor * vw = cgraph->nodes[r];
        if (vw->op != GGML_OP_VIEW || vw->src[0] != node) {
            continue;
        }
        if (ggml_node_get_use_count(cgraph, r) != 1 || (vw->flags & GGML_TENSOR_FLAG_OUTPUT) || vw->view_offs != 0 ||
                vw->nb[2] != node->nb[1]) {
            return nullptr;
        }
        for (int k = r + 1; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * gdn = cgraph->nodes[k];
            if (gdn->op == GGML_OP_GATED_DELTA_NET && gdn->src[6] == vw) {
                return vw->ne[2] == idx->ne[0] ? gdn : nullptr;
            }
        }
        return nullptr;
    }
    return nullptr;
}

static int ggml_cuda_fusion_plan_node(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i,
                                      const std::unordered_map<const ggml_tensor *, int> & index) {
    ggml_cuda_fusion_plan & plan = *g_ggml_cuda_fusion_plan;
    const ggml_tensor *     node = cgraph->nodes[i];
    const auto claimed = [&]() -> int {   // nodes after i that the group just recorded spans, or -1 if none recorded
        if (!plan.matched) {
            return -1;
        }
        plan.matched = false;
        int last_i = i;
        for (const ggml_tensor * w : plan.groups.back().writes) {
            const auto it = index.find(w);
            if (it != index.end() && it->second > last_i) {
                last_i = it->second;
            }
        }
        return last_i - i;
    };
    int skip = -1;
    plan.matched = false;
    // Keep the direct state table and indices alive until the GDN node is allocated.
    if (node->op == GGML_OP_GET_ROWS) {
        const ggml_tensor * gdn = ggml_cuda_gdn_state_direct_target(cgraph, i);
        if (gdn != nullptr) {
            ggml_cuda_fusion_plan::group g;
            g.writes.push_back(gdn);
            g.reads.push_back(node->src[0]);
            g.reads.push_back(node->src[1]);
            plan.groups.push_back(std::move(g));
        }
        // Prefix-direct keeps the pack table alive through the GDN.
        // view and s_copy to the GDN is harmless when the gather does run, and the mask may change after the plan
        const ggml_tensor * gdn_p = ggml_cuda_gdn_prefix_direct_target(cgraph, i);
        if (gdn_p != nullptr) {
            ggml_cuda_fusion_plan::group g;
            g.writes.push_back(gdn_p);
            g.reads.push_back(node->src[0]);
            g.reads.push_back(node->src[1]);
            plan.groups.push_back(std::move(g));
        }
    }
    if (node->op == GGML_OP_GET_ROWS) {
        ggml_cuda_try_fuse_gdn_conv_prologue(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
        ggml_cuda_try_fuse_gdn_conv_replay(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_MUL_MAT) {
        ggml_cuda_try_fuse_gdn_alpha_beta_prologue(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_try_fuse_gdn_out_chain(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
        ggml_cuda_try_fuse_gdn_out_direct(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_try_fuse_attn_q_chain(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
        ggml_cuda_try_fuse_attn_k_write(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
        ggml_cuda_try_fuse_rms_norm_gated(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_MUL_MAT) {
        ggml_cuda_try_fuse_attn_v_write(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_XYZKV_WHT) {
        ggml_cuda_try_fuse_attn_out_chain(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    if (node->op == GGML_OP_REPEAT) {
        ggml_cuda_try_fuse_kq_mask_chain(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    for (int (*matcher)(ggml_backend_cuda_context *, ggml_cgraph *, int) :
            { ggml_cuda_try_fuse_ptq1_sibling, ggml_cuda_try_fuse_norm_fwht, ggml_cuda_try_fuse_glu_fwht,
              ggml_cuda_try_fuse_signed_fwht }) {
        matcher(cuda_ctx, cgraph, i);
        if ((skip = claimed()) >= 0) { return skip; }
    }
    return 0;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_GET_ROWS) {
        const int skip = ggml_cuda_try_fuse_gdn_conv_prologue(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
        const int skip_replay = ggml_cuda_try_fuse_gdn_conv_replay(cuda_ctx, cgraph, i);
        if (skip_replay > 0) {
            return skip_replay;
        }
    }
    if (node->op == GGML_OP_MUL_MAT) {
        const int skip = ggml_cuda_try_fuse_gdn_alpha_beta_prologue(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    if (node->op == GGML_OP_RMS_NORM) {
        const int skip = ggml_cuda_try_fuse_gdn_out_chain(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
        const int skip_direct = ggml_cuda_try_fuse_gdn_out_direct(cuda_ctx, cgraph, i);
        if (skip_direct > 0) {
            return skip_direct;
        }
    }
    if (node->op == GGML_OP_RMS_NORM) {
        const int skip_q = ggml_cuda_try_fuse_attn_q_chain(cuda_ctx, cgraph, i);
        if (skip_q > 0) {
            return skip_q;
        }
        const int skip_k = ggml_cuda_try_fuse_attn_k_write(cuda_ctx, cgraph, i);
        if (skip_k > 0) {
            return skip_k;
        }
        const int skip = ggml_cuda_try_fuse_rms_norm_gated(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    if (node->op == GGML_OP_MUL_MAT) {
        const int skip = ggml_cuda_try_fuse_attn_v_write(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    if (node->op == GGML_OP_XYZKV_WHT) {
        const int skip = ggml_cuda_try_fuse_attn_out_chain(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    if (node->op == GGML_OP_REPEAT) {
        const int skip = ggml_cuda_try_fuse_kq_mask_chain(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    {
        const int skip = ggml_cuda_try_fuse_ptq1_sibling(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_layout(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_layout(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // Read recurrent state directly from its cache table using s_copy indices.
    // In place is safe: each CTA reads its own (head, column) slice before it writes any
    // snapshot, and CTAs own disjoint slices. Returns -1 = this node consumed, nothing launched.
    if (node->op == GGML_OP_GET_ROWS) {
        // the match is shared with the allocation plan (ggml_cuda_fusion_plan_node declares table + s_copy as read by
        // the GDN), so what is skipped here is exactly what the allocator was told to keep alive
        const ggml_tensor * gdn = ggml_cuda_gdn_state_direct_target(cgraph, i);
        if (gdn != nullptr) {
            cuda_ctx->gdn_state_src[gdn] = { (const float *) node->src[0]->data, (const int32_t *) node->src[1]->data };
            return -1;
        }
        const ggml_tensor * gdn_p = ggml_cuda_gdn_prefix_direct_target(cgraph, i);
        if (gdn_p != nullptr) {
            cuda_ctx->gdn_prefix_src[gdn_p] = { (const float *) node->src[0]->data, (const int32_t *) node->src[1]->data,
                                                 node->src[0]->ne[0] };
            return -1;
        }
    }

    {
        const int skip = ggml_cuda_try_fuse_norm_fwht(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    {
        const int skip = ggml_cuda_try_fuse_glu_fwht(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    {
        const int skip = ggml_cuda_try_fuse_signed_fwht(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    if (node->op == GGML_OP_CPY) {
        const int extra = ggml_cuda_try_fuse_multi_cpy(cuda_ctx, cgraph, i);
        if (extra > 0) {
            return extra;
        }
    }
    if (node->op == GGML_OP_ADD && ggml_cuda_try_fuse_add_softplus_mul(cuda_ctx, cgraph, i)) {
        return 2;
    }

    return 0;
}

#ifdef USE_CUDA_GRAPH
// End the current capture chunk, launch it, then resume
// capturing on the same stream. Called only at a clean point of the evaluation (every side stream joined), so a chunk
// holds whole fork/join regions; the chunks run in stream order, exactly as the one graph did.
static void ggml_cuda_graph_chunk_cut(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, int chunk) {
    cudaGraph_t g = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &g));
    if ((int) graph->head_graphs.size() <= chunk) {
        graph->head_graphs.resize(chunk + 1, nullptr);
        graph->head_instances.resize(chunk + 1, nullptr);
    }
    if (graph->head_graphs[chunk] != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(graph->head_graphs[chunk]));
    }
    graph->head_graphs[chunk] = g;
    cudaGraphExec_t & inst = graph->head_instances[chunk];
    if (inst != nullptr) {
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo result_info;
        const cudaError_t stat = cudaGraphExecUpdate(inst, g, &result_info);
#else
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdateResult result_info;
        const cudaError_t stat = cudaGraphExecUpdate(inst, g, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000
        if (stat == cudaErrorGraphExecUpdateFailure) {
            (void) cudaGetLastError();
            CUDA_CHECK(cudaGraphExecDestroy(inst));
            inst = nullptr;
        } else {
            GGML_ASSERT(stat == cudaSuccess);
        }
    }
    if (inst == nullptr) {
        CUDA_CHECK(cudaGraphInstantiate(&inst, g, NULL, NULL, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(inst, cuda_ctx->stream()));
    graph->n_head = chunk + 1;
    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));

}
#endif // USE_CUDA_GRAPH

static void ggml_cuda_graph_evaluate_and_capture(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        const bool use_cuda_graph,
        const bool cuda_graph_update_required,
        const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;
    const bool                   gdn_ab_side         = true;
    bool                         gdn_ab_join_pending = false;
    // Run the attention Q chain, V write and K write on three streams.
    // attn_state 0 idle | 1 the Q chain fired (the V write is next) | 2 the V write ran on stream 1 (the K write is next)
    // | 3 both writes issued (stream 0 joins them before the next node). Anything unexpected joins at once.
    const bool                   attn_par        = true;
    int                          attn_state      = 0;
    bool                         attn_fork_armed = false;
    const auto attn_join = [&](bool j1, bool j2) {
        if (j1) { CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), g_attn_join1[cuda_ctx->device])); }
        if (j2) { CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), g_attn_join2[cuda_ctx->device])); }
    };

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            cuda_ctx->gdn_state_src.clear();   // entries live for one evaluation only (ggml_cuda_try_fuse, GET_ROWS)
            cuda_ctx->gdn_prefix_src.clear();
            for (auto & tw : cuda_ctx->ptq1_q8_twins) {
                tw.f32 = nullptr;              // so do q8_1 twins: inputs are re-uploaded between evaluations
            }
            // Split a large capture at clean points after one eighth, one quarter and one half of its nodes.
#ifdef USE_CUDA_GRAPH
            const bool chunked = use_cuda_graph && cuda_graph_update_required && cgraph->n_nodes >= 2048;
            ggml_cuda_graph * chunk_graph = chunked ? cuda_ctx->cuda_graph(graph_key) : nullptr;
            if (use_cuda_graph && cuda_graph_update_required) {
                cuda_ctx->cuda_graph(graph_key)->n_head = 0;   // this capture decides the chunks the replays launch
            }
#else
            const bool chunked = false;
#endif // USE_CUDA_GRAPH
            const int chunk_cuts[3] = { cgraph->n_nodes / 8, cgraph->n_nodes / 4, cgraph->n_nodes / 2 };
            int n_cut = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
#ifdef USE_CUDA_GRAPH
                if (chunked && n_cut < 3 && i >= chunk_cuts[n_cut] && !is_concurrent_event_active &&
                        cuda_ctx->curr_stream_no == 0 && !gdn_ab_join_pending && attn_state == 0 && !attn_fork_armed) {
                    ggml_cuda_graph_chunk_cut(cuda_ctx, chunk_graph, n_cut);
                    n_cut++;
                }
#endif // USE_CUDA_GRAPH
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                cuda_ctx->ptq1_q8_twin_fresh = -1;
                // Fork the GDN alpha/beta prologue onto stream 1 and join after the qkvz matmul.
                const bool gdn_ab_fork = gdn_ab_side && !is_concurrent_event_active && cuda_ctx->curr_stream_no == 0 &&
                                         g_gdn_ab_fork[cuda_ctx->device] != nullptr && ggml_cuda_is_gdn_alpha_mm(node, nullptr);
                if (gdn_ab_fork) {
                    CUDA_CHECK(cudaEventRecord(g_gdn_ab_fork[cuda_ctx->device], cuda_ctx->stream()));
                    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(cuda_ctx->device, 1), g_gdn_ab_fork[cuda_ctx->device]));
                    cuda_ctx->curr_stream_no = 1;
                }
                // Route V and K writes to streams 1 and 2.
                const bool attn_qk_head = node->op == GGML_OP_RMS_NORM && node->type == GGML_TYPE_F32 && node->ne[0] == 256;
                const bool attn_v_head  = node->op == GGML_OP_MUL_MAT && node->type == GGML_TYPE_F32 && node->ne[0] == 64 &&
                                          ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD;
                int attn_side = 0;
                if (attn_state == 3 || (attn_state == 1 && !attn_v_head) || (attn_state == 2 && !attn_qk_head)) {
                    attn_join(attn_state >= 2, attn_state == 3);
                    attn_state = 0;
                }
                if (attn_par && !gdn_ab_fork && !is_concurrent_event_active) {
                    if (attn_state == 0 && attn_qk_head) {
                        CUDA_CHECK(cudaEventRecord(g_attn_fork[cuda_ctx->device], cuda_ctx->stream()));   // before the Q chain
                        attn_fork_armed = true;
                    } else if (attn_state == 1 || attn_state == 2) {
                        attn_side = attn_state;   // 1: the V write, 2: the K write
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(cuda_ctx->device, attn_side), g_attn_fork[cuda_ctx->device]));
                        cuda_ctx->curr_stream_no = attn_side;
                    }
                }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                {
                    const ggml_tensor * claimed_last = nodes_to_skip > 0 ? cgraph->nodes[i + nodes_to_skip] : nullptr;
                    if (attn_side != 0) {
                        CUDA_CHECK(cudaEventRecord(attn_side == 1 ? g_attn_join1[cuda_ctx->device] : g_attn_join2[cuda_ctx->device],
                                                   cuda_ctx->stream()));
                        cuda_ctx->curr_stream_no = 0;
                        if (claimed_last != nullptr && claimed_last->op == GGML_OP_SET_ROWS) {
                            attn_state = attn_side == 1 ? 2 : 3;
                        } else {   // not the expected cache write: rejoin now, before anything else runs on stream 0
                            attn_join(true, attn_side == 2);
                            attn_state = 0;
                        }
                    } else if (attn_fork_armed) {
                        attn_fork_armed = false;
                        attn_state = claimed_last != nullptr && claimed_last->op == GGML_OP_XYZKV_WHT ? 1 : 0;
                    }
                }
                if (gdn_ab_fork) {
                    CUDA_CHECK(cudaEventRecord(g_gdn_ab_join[cuda_ctx->device], cuda_ctx->stream()));   // on stream 1
                    cuda_ctx->curr_stream_no = 0;
                    gdn_ab_join_pending = true;
                }
                const auto gdn_ab_join_after_launch = [&]() {
                    if (gdn_ab_join_pending && !gdn_ab_fork) {
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), g_gdn_ab_join[cuda_ctx->device]));
                        gdn_ab_join_pending = false;
                    }
                };

                if (nodes_to_skip < 0) {   // this node was consumed by a later one (GET_ROWS -> GDN state): nothing to launch
                    continue;
                }
                ggml_cuda_ptq1_twin_invalidate(*cuda_ctx, cgraph, i, i + nodes_to_skip);   // nodes about to be written

                if (nodes_to_skip != 0) {
                    gdn_ab_join_after_launch();
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                gdn_ab_join_after_launch();

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
            if (gdn_ab_join_pending) {   // never leave stream 1 unjoined (a capture must end joined)
                CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), g_gdn_ab_join[cuda_ctx->device]));
                gdn_ab_join_pending = false;
            }
            if (attn_state != 0) {
                attn_join(attn_state >= 2, attn_state == 3);
                attn_state = 0;
            }
            attn_fork_armed = false;
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (!cuda_graph_update_required) {
            // Replay leading graph chunks in capture order.
            for (int c = 0; c < graph->n_head; ++c) {
                CUDA_CHECK(cudaGraphLaunch(graph->head_instances[c], cuda_ctx->stream()));
            }
        }
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    const bool had_graph_wait = cuda_ctx->graph_wait_ev != nullptr;
    if (had_graph_wait) {
        ggml_cuda_upload_flush(cuda_ctx);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), cuda_ctx->graph_wait_ev, 0));
        cuda_ctx->graph_wait_ev = nullptr;
    }
    ggml_cuda_upload_flush(cuda_ctx);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

// Make the next graph compute on this backend wait for `event` (nullptr clears). The wait is
// queued on the stream only there -- after the scheduler has handled the graph's inputs -- so nothing the caller issues
// on this stream before the graph (uploads, the scheduler's synchronous input paths) waits behind the event. Reached
// through the registry: ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_wait_event").
static void ggml_backend_cuda_graph_wait_event(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    cuda_ctx->graph_wait_ev = event != nullptr ? (cudaEvent_t) event->context : nullptr;
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_upload_flush(cuda_ctx);

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

// The GDN alpha/beta prologue depends only on the layer's attn_norm, not its qkv/z matmul.
// During planning, each layer's 9-node block moves to just before
// the layer's attn_qkv MUL_MAT, and attn_norm is kept alive until the qkv and z outputs are allocated -- so the evaluation
// can run the block on stream 1 while stream 0 runs the qkvz matmul, and no qkv/z output can land on the attn_norm the
// side kernel still reads. The move is independent of launch scheduling.
static void ggml_cuda_gdn_ab_hoist(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
                                   ggml_backend_graph_optimize_params * params) {
    constexpr int NB = 9;   // alpha MUL_MAT, RESHAPE, ADD, SOFTPLUS, MUL, RESHAPE, beta MUL_MAT, RESHAPE, SIGMOID
    const int n = cgraph->n_nodes;
    int moved = 0;
    for (int ia = 0; ia + NB <= n; ++ia) {
        int layer = -1;
        if (!ggml_cuda_is_gdn_alpha_mm(cgraph->nodes[ia], &layer)) {
            continue;
        }
        ggml_tensor ** blk = cgraph->nodes + ia;
        char beta_w[64];
        snprintf(beta_w, sizeof(beta_w), "blk.%d.ssm_beta.weight", layer);
        if (blk[1]->op != GGML_OP_RESHAPE || blk[2]->op != GGML_OP_ADD ||
                blk[3]->op != GGML_OP_UNARY || ggml_get_unary_op(blk[3]) != GGML_UNARY_OP_SOFTPLUS ||
                blk[4]->op != GGML_OP_MUL || blk[5]->op != GGML_OP_RESHAPE || blk[6]->op != GGML_OP_MUL_MAT ||
                blk[6]->src[0] == nullptr || strcmp(blk[6]->src[0]->name, beta_w) != 0 ||
                blk[7]->op != GGML_OP_RESHAPE || blk[8]->op != GGML_OP_UNARY ||
                ggml_get_unary_op(blk[8]) != GGML_UNARY_OP_SIGMOID) {
            continue;
        }
        // the insertion point: this layer's attn_qkv MUL_MAT (followed by its attn_gate sibling), before the block
        char qkv_w[64], gate_w[64];
        snprintf(qkv_w,  sizeof(qkv_w),  "blk.%d.attn_qkv.weight",  layer);
        snprintf(gate_w, sizeof(gate_w), "blk.%d.attn_gate.weight", layer);
        int iq = -1;
        for (int k = ia - 1; k >= 0; --k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr && strcmp(t->src[0]->name, qkv_w) == 0) {
                iq = k;
                break;
            }
        }
        if (iq < 0 || iq + 1 >= ia) {
            continue;
        }
        ggml_tensor * qkv = cgraph->nodes[iq];
        ggml_tensor * z   = cgraph->nodes[iq + 1];
        if (z->op != GGML_OP_MUL_MAT || z->src[0] == nullptr || strcmp(z->src[0]->name, gate_w) != 0) {
            z = nullptr;
        }
        // legal: every node source of the block lies before iq or inside the block, and nothing in [iq, ia) reads the block
        const auto in_block = [&](const ggml_tensor * t) {
            for (int b = 0; b < NB; ++b) {
                if (blk[b] == t) {
                    return true;
                }
            }
            return false;
        };
        bool ok = true;
        for (int k = iq; k < ia && ok; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                ok = t->src[s] == nullptr || !in_block(t->src[s]);
            }
            for (int b = 0; b < NB && ok; ++b) {   // and the block reads none of them
                for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                    ok = blk[b]->src[s] != t;
                }
            }
        }
        if (!ok) {
            continue;
        }
        ggml_tensor * block[NB];
        memcpy(block, blk, sizeof(block));
        memmove(cgraph->nodes + iq + NB, cgraph->nodes + iq, (size_t) (ia - iq) * sizeof(ggml_tensor *));
        memcpy(cgraph->nodes + iq, block, sizeof(block));
        ggml_tensor * x = block[0]->src[1];   // attn_norm
        params->add_alloc_dep(params->user_data, x, qkv);
        if (z != nullptr) {
            params->add_alloc_dep(params->user_data, x, z);
        }
        moved++;
        ia += NB - 1;   // the node after the block's old end is where the scan resumes
    }
    if (moved > 0) {
        const int dev = cuda_ctx->device;
        cuda_ctx->stream(dev, 1);   // create the side stream outside any capture
        if (g_gdn_ab_fork[dev] == nullptr) {
            ggml_cuda_set_device(dev);
            CUDA_CHECK(cudaEventCreateWithFlags(&g_gdn_ab_fork[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_gdn_ab_join[dev], cudaEventDisableTiming));
        }
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_gdn_ab_hoist(cuda_ctx, cgraph, params);
    {   // Create side-stream events outside stream capture.
        const int dev = cuda_ctx->device;
        if (g_attn_fork[dev] == nullptr) {
            ggml_cuda_set_device(dev);
            cuda_ctx->stream(dev, 1);
            cuda_ctx->stream(dev, 2);
            CUDA_CHECK(cudaEventCreateWithFlags(&g_attn_fork[dev],  cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_attn_join1[dev], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_attn_join2[dev], cudaEventDisableTiming));
        }
    }

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

    // Keep fusion inputs alive until the fused output is allocated.
    if (!disable_fusion) {
        std::unordered_map<const ggml_tensor *, int> index;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            index[cgraph->nodes[i]] = i;
        }
        ggml_cuda_fusion_plan plan;
        g_ggml_cuda_fusion_plan = &plan;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (!ggml_cuda_is_view_or_noop(cgraph->nodes[i])) {   // the evaluation loop steps over these too
                i += ggml_cuda_fusion_plan_node(cuda_ctx, cgraph, i, index);
            }
        }
        g_ggml_cuda_fusion_plan = nullptr;
        for (const auto & g : plan.groups) {
            const ggml_tensor * last   = nullptr;   // the write allocated last: every other tensor lives until it exists
            int                 last_i = -1;
            for (const ggml_tensor * w : g.writes) {
                const auto it = index.find(w);
                if (it != index.end() && it->second > last_i) {
                    last_i = it->second;
                    last   = w;
                }
            }
            if (last == nullptr) {
                continue;
            }
            for (const auto * list : { &g.reads, &g.writes }) {
                for (const ggml_tensor * t : *list) {
                    if (t != last) {
                        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(t), const_cast<ggml_tensor *>(last));
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
        return;
    }

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
                // the expert path reads PTQ1_0 through the row-major dp4a kernels
                if (a->type == GGML_TYPE_PTQ1_0 && (op->op == GGML_OP_MUL_MAT_ID || !ggml_cuda_ptq1_ilv_operand_ok(a))) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_PTQ1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                if (op->src[0]->type == GGML_TYPE_PTQ1_0 && !ggml_cuda_ptq1_ilv_operand_ok(op->src[0])) {
                    return false;
                }
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_PTQ1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                if (op->type == GGML_TYPE_XYZKV2_0 && op->src[0]->ne[0] % 64 != 0) {
                    return false;
                }
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL ||
                               op->type == GGML_TYPE_XYZKV2_0) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_XYZKV_WHT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] % 32 == 0;  // supports 32, 64, and 128 WHT groups
        case GGML_OP_DRAFT_SAMPLE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_I32 &&
                   op->src[2]->type == GGML_TYPE_I32 && op->type == GGML_TYPE_I32;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_graph_wait_event") == 0) {
        return (void *)ggml_backend_cuda_graph_wait_event;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    // q8_1 twins (common.cuh ptq1_q8_twin): allocated here, never inside a stream capture
    {
        int prev_device = 0;
        CUDA_CHECK(cudaGetDevice(&prev_device));
        ggml_cuda_set_device(device);
        for (auto & tw : ctx->ptq1_q8_twins) {
            CUDA_CHECK(cudaMalloc(&tw.q8, ggml_backend_cuda_context::PTQ1_Q8_TWIN_BYTES));
        }
        {
            constexpr size_t stack_size = 256;
            const cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, stack_size);
            if (err != cudaSuccess) {
                GGML_LOG_WARN("%s: device %d: stack limit %zu B not set: %s\n", __func__, device, stack_size, cudaGetErrorString(err));
                (void) cudaGetLastError();
            }
        }
        ggml_cuda_set_device(prev_device);
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
