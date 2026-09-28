#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

// ---------------------------------------------------------------------------------------------------------------------
void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// GGML_OP_DRAFT_SAMPLE runs the speculative drafter's coupled draw for one row, so the draft
// chain can run step after step on the GPU with no host round trip in between (common/speculative.cpp). The reference
// is ggml-cpu's ggml_compute_forward_draft_sample, itself the CPU chain step for step:
//   1 the exact top-k over the row, ties to the smaller id    (common_topk_scan::run_mapped)
//   2 top-p: softmax over the k sorted, the cumulative cut    (llama_sampler_top_p_apply)
//   3 coupled dist: float exps, a double sum, the argmax of logit + Gumbel(key, id), p /= sum (llama_sampler_dist_apply)
// A candidate's key orders by (logit desc, id asc): the orderable logit bits above the inverted id, so the plain 64-bit
// maximum is exactly the CPU's order and every key is unique. The host re-runs step 2-3 on the returned candidates and
// truncates the draft at any disagreement (a last-ulp exp can only move a top-p cut that sits on 0.95).

// Each thread keeps its top-K in registers, merges by warp shuffles, and
// spreads the tail's exps and Gumbel draws over the lanes; only the float sums the CPU does in order stay sequential.

#define DRAFT_SAMPLE_THREADS 256

static __device__ __forceinline__ uint32_t ds_orderable(float v) {
    // -0 keys as +0: the CPU compares floats, so the two tie and the smaller id wins (common_topk_scan::run_mapped)
    const uint32_t u = __float_as_uint(v) == 0x80000000u ? 0u : __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static __device__ __forceinline__ float ds_from_orderable(uint32_t o) {
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
}

static __device__ __forceinline__ uint64_t ds_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static __device__ float ds_gumbel(uint64_t key, int32_t id) {   // llama_coupled_gumbel: the same bits, in double
    const uint64_t h = ds_splitmix64(key ^ ds_splitmix64((uint64_t) (uint32_t) id));
    const double   u = ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return (float) -log(-log(u));
}

static __device__ __forceinline__ float ds_expf(float x) {   // correctly rounded in all but double-rounding corners
    return (float) exp((double) x);
}

// a[] strongest first; k enters where it belongs and the weakest falls off (registers only: fully unrolled)
template <int K>
static __device__ __forceinline__ void ds_insert(uint64_t (&a)[K], uint64_t k) {
    if (k <= a[K - 1]) {
        return;
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
        if (k > a[i]) {
            const uint64_t t = a[i];
            a[i] = k;
            k = t;
        }
    }
}

// the warp's top-K of its lanes' lists, into out[] on every lane; a lane's list is consumed (keys are unique, so each
// maximum has exactly one owner; 0 = nothing left)
template <int K>
static __device__ __forceinline__ void ds_warp_topk(uint64_t (&a)[K], uint64_t (&out)[K]) {
#pragma unroll
    for (int r = 0; r < K; ++r) {
        uint64_t m = a[0];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const uint64_t y = __shfl_xor_sync(0xffffffffu, m, o);
            m = y > m ? y : m;
        }
        out[r] = m;
        if (m != 0 && a[0] == m) {
#pragma unroll
            for (int i = 0; i < K - 1; ++i) {
                a[i] = a[i + 1];
            }
            a[K - 1] = 0;
        }
    }
}

// 3: the chain as the CPU runs it, on one full warp. Lane i < top_k holds the i-th strongest candidate's key (0 = none);
//    the sums the CPU does in order are gathered and done in order, so they round the same way. Writes the record's
//    candidates and out[0..2] (out[3] = 0); returns the selected candidate's index and id, -1 when fewer than top_k
//    finite logits exist. Every value it returns is warp-uniform.
static __device__ int ds_tail(const uint64_t mine, const int lane, const int top_k, const float top_p,
                              const uint32_t * __restrict__ key2, int32_t * __restrict__ out, int32_t & sel_tok) {
    constexpr int KMAX = GGML_DRAFT_SAMPLE_MAX_K;
    const bool    have = lane < top_k && mine != 0;
    const int32_t id   = have ? (int32_t) (0xFFFFFFFFu - (uint32_t) (mine & 0xFFFFFFFFull)) : -1;
    const float   lg   = have ? ds_from_orderable((uint32_t) (mine >> 32)) : -INFINITY;
    const float   lg0  = __shfl_sync(0xffffffffu, lg, 0);
    const bool    full = __all_sync(0xffffffffu, lane >= top_k || have);   // top_k finite candidates exist

    const float e = have ? ds_expf(lg - lg0) : 0.0f;   // softmax numerators: top-p's and the dist's are equal
    int kept = top_k;
    float cum_sum = 0.0f;
    if (top_p < 1.0f) {
        for (int i = 0; i < top_k; ++i) {
            cum_sum += __shfl_sync(0xffffffffu, e, i);
        }
        const float pn = have ? e / cum_sum : 0.0f;
        float cum = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cum += __shfl_sync(0xffffffffu, pn, i);
            if (cum >= top_p) {
                kept = i + 1;
                break;   // every lane walks the same values: the loop exits uniformly
            }
        }
    }

    int   sel = 0;
    float pr  = 0.0f;
    if (kept == 1) {
        pr = lane == 0 ? 1.0f : 0.0f;
    } else {
        double sum_cum = 0.0;
        for (int i = 0; i < kept; ++i) {
            sum_cum += (double) __shfl_sync(0xffffffffu, e, i);
        }
        const uint64_t key = (uint64_t) key2[0] | ((uint64_t) key2[1] << 32);
        const float    s   = lane < kept ? lg + ds_gumbel(key, id) : -INFINITY;
        float bestv = -INFINITY;
        for (int i = 0; i < kept; ++i) {   // strict >: the first maximum wins, as on the CPU
            const float si = __shfl_sync(0xffffffffu, s, i);
            if (si > bestv) {
                bestv = si;
                sel   = i;
            }
        }
        pr = lane < kept ? (float) ((double) e / sum_cum) : 0.0f;
    }

    if (lane < top_k) {
        out[4 + lane]            = full ? id : 0;
        out[4 + KMAX + lane]     = full ? __float_as_int(lg) : 0;
        out[4 + 2*KMAX + lane]   = full ? __float_as_int(pr) : 0;
    }
    sel_tok = __shfl_sync(0xffffffffu, id, sel);
    const int32_t id1 = __shfl_sync(0xffffffffu, id, 1);
    if (lane == 0) {
        out[0] = full ? sel_tok : -1;
        out[1] = full ? kept    : -1;
        out[2] = full && kept > 1 ? id1 : -1;
        out[3] = 0;
    }
    return full ? sel : -1;
}

template <int K>
static __global__ void k_draft_sample(const float * __restrict__ x, const int32_t * __restrict__ col_ids,
                                      const uint32_t * __restrict__ key2, int32_t * __restrict__ out,
                                      const int32_t * __restrict__ step, int32_t * __restrict__ rec,
                                      const int64_t rec_nb1, int32_t * __restrict__ col_out,
                                      const int n, const int top_k, const float top_p) {
    constexpr int NWARPS = DRAFT_SAMPLE_THREADS / 32;
    __shared__ uint64_t warp_lists[NWARPS][K];
    __shared__ int32_t  sel_id;
    __shared__ int32_t  sel_col;

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    // 1: each thread's exact top-K over its strided columns, key = (orderable logit, inverted id): the CPU's order
    uint64_t a[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        a[i] = 0;   // below every real key: finite logits order above 0x00800000
    }
    for (int c = t; c < n; c += DRAFT_SAMPLE_THREADS) {
        const float v = x[c];
        if (v > -INFINITY) {   // NaN and -inf are never kept
            ds_insert<K>(a, ((uint64_t) ds_orderable(v) << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) col_ids[c]));
        }
    }

    // 2: warp merge, then the warps' lists merged by warp 0
    uint64_t w[K];
    ds_warp_topk<K>(a, w);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < K; ++i) {
            warp_lists[warp][i] = w[i];
        }
    }
    if (t == 0) {
        sel_id  = -1;
        sel_col = 0;
    }
    __syncthreads();

    if (warp == 0) {
        uint64_t b[K];
#pragma unroll
        for (int i = 0; i < K; ++i) {
            b[i] = lane < NWARPS ? warp_lists[lane][i] : 0;
        }
        uint64_t best[K];
        ds_warp_topk<K>(b, best);   // every lane of warp 0 holds the block's top-K, strongest first

        uint64_t mine = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (i == lane) {
                mine = best[i];
            }
        }
        int32_t sel_tok;
        const int sel = ds_tail(mine, lane, top_k, top_p, key2, out, sel_tok);
        if (lane == 0) {
            sel_id = sel >= 0 ? sel_tok : -1;
        }
    }
    __syncthreads();

    // 4: the selected id's column (ids are unique over the columns), for the next step's embedding lookup
    if (sel_id >= 0) {
        for (int c = t; c < n; c += DRAFT_SAMPLE_THREADS) {
            if (col_ids[c] == sel_id) {
                sel_col = c;
            }
        }
    }
    __syncthreads();

    if (t == 0) {
        out[3] = sel_col;
        if (rec != nullptr) {
            int32_t * row = (int32_t *) ((char *) rec + (int64_t) step[0]*rec_nb1);
            for (int i = 0; i < GGML_DRAFT_SAMPLE_OUT; ++i) {
                row[i] = out[i];
            }
        }
        if (col_out != nullptr) {
            col_out[0] = sel_col;
        }
    }
}

// The wide-row kernel holds the whole row in registers and does not sort it. The top_k-th largest thread maximum is a value that at least top_k columns reach, so every column of the
// exact top-k lies at or above it: a bisection over its bits with block-wide counts finds it, and stops as soon as
// exactly top_k maxima clear the bar. What clears it (top_k plus the odd neighbour) is ranked by counting under the
// CPU's key and carries its column, so the column search is gone. Rows longer than 32768 keep v2.

#define DS3_THREADS 1024
#define DS3_PER     32            // columns per thread: rows up to 32768, the compact draft head
#define DS3_CMAX    DS3_THREADS   // top_k distinct thread maxima admit at most 32*top_k <= 1024 candidates

static __global__ void __launch_bounds__(DS3_THREADS, 1)
k_draft_sample_v3(const float * __restrict__ x, const int32_t * __restrict__ col_ids,
                  const uint32_t * __restrict__ key2, int32_t * __restrict__ out,
                  const int32_t * __restrict__ step, int32_t * __restrict__ rec,
                  const int64_t rec_nb1, int32_t * __restrict__ col_out,
                  const int n, const int top_k, const float top_p) {
    __shared__ uint64_t cand_key[DS3_CMAX];
    __shared__ int32_t  cand_col[DS3_CMAX];
    __shared__ uint64_t top_key[GGML_DRAFT_SAMPLE_MAX_K];
    __shared__ int32_t  top_col[GGML_DRAFT_SAMPLE_MAX_K];
    __shared__ int      n_cand;

    const int t    = threadIdx.x;
    const int lane = t & 31;

    // 1: column t + j*DS3_THREADS as an orderable key in v[j]; NaN and -inf become 0 and are never kept
    uint32_t v[DS3_PER];
    uint32_t vmax = 0;
#pragma unroll
    for (int j = 0; j < DS3_PER; ++j) {
        const int   c = t + j*DS3_THREADS;
        const float f = c < n ? x[c] : -INFINITY;
        v[j] = f > -INFINITY ? ds_orderable(f) : 0u;
        vmax = max(vmax, v[j]);
    }
    if (t == 0) {
        n_cand = 0;
    }
    if (t < GGML_DRAFT_SAMPLE_MAX_K) {
        top_key[t] = 0;
        top_col[t] = 0;
    }

    // 2: the bar, bit by bit: keep a bit while at least top_k thread maxima reach it (the count is block-uniform)
    uint32_t bar = 0;
    for (int b = 31; b >= 0; --b) {
        const uint32_t probe = bar | (1u << b);
        const int      cnt   = __syncthreads_count(vmax >= probe);
        if (cnt >= top_k) {
            bar = probe;
            if (cnt == top_k) {
                break;
            }
        }
    }

    // 3: the candidates: every finite column at or above the bar
    const uint32_t lo = bar > 0 ? bar : 1u;
#pragma unroll
    for (int j = 0; j < DS3_PER; ++j) {
        if (v[j] >= lo) {
            const int slot = atomicAdd(&n_cand, 1);
            if (slot < DS3_CMAX) {
                const int c = t + j*DS3_THREADS;
                cand_key[slot] = ((uint64_t) v[j] << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) col_ids[c]);
                cand_col[slot] = c;
            }
        }
    }
    __syncthreads();

    // 4: rank by counting (keys are unique, as the ids are): the top_k strongest land in order. Past DS3_CMAX (only
    //    thousands of tied maxima get there) nothing is ranked and the record reads "not full": the host stops the draft.
    const int m = n_cand;
    if (m <= DS3_CMAX && t < m) {
        const uint64_t mine = cand_key[t];
        int rank = 0;
        for (int i = 0; i < m; ++i) {
            rank += cand_key[i] > mine;
        }
        if (rank < top_k) {
            top_key[rank] = mine;
            top_col[rank] = cand_col[t];
        }
    }
    __syncthreads();

    // 5: the tail on warp 0; the selected candidate's column is at hand
    if (t < 32) {
        int32_t sel_tok;
        const int sel = ds_tail(lane < top_k ? top_key[lane] : 0, lane, top_k, top_p, key2, out, sel_tok);
        const int32_t c = sel >= 0 ? top_col[sel] : 0;
        if (lane == 0) {
            out[3] = c;
            if (col_out != nullptr) {
                col_out[0] = c;
            }
        }
        __syncwarp();   // the record row copies what every lane wrote
        if (rec != nullptr) {
            int32_t * row = (int32_t *) ((char *) rec + (int64_t) step[0]*rec_nb1);
            for (int i = lane; i < GGML_DRAFT_SAMPLE_OUT; i += 32) {
                row[i] = out[i];
            }
        }
    }
}

void ggml_cuda_op_draft_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits  = dst->src[0];
    const ggml_tensor * col_ids = dst->src[1];
    const ggml_tensor * key     = dst->src[2];
    GGML_ASSERT(logits->type == GGML_TYPE_F32 && col_ids->type == GGML_TYPE_I32 && key->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(logits) && ggml_is_contiguous(col_ids) && dst->type == GGML_TYPE_I32);
    const int   top_k = ggml_get_op_params_i32(dst, 0);
    const float top_p = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(top_k >= 1 && top_k <= GGML_DRAFT_SAMPLE_MAX_K);

    const ggml_tensor * step = dst->src[3];
    const ggml_tensor * rec  = dst->src[4];
    const ggml_tensor * col  = dst->src[5];
    const float   * x   = (const float *) logits->data;
    const int32_t * ids = (const int32_t *) col_ids->data;
    const uint32_t * k2 = (const uint32_t *) key->data;
    int32_t       * o   = (int32_t *) dst->data;
    const int32_t * sp  = step ? (const int32_t *) step->data : nullptr;
    int32_t       * rp  = rec  ? (int32_t *) rec->data : nullptr;
    const int64_t   rnb = rec  ? (int64_t) rec->nb[1] : 0;
    int32_t       * cp  = col  ? (int32_t *) col->data : nullptr;
    const int       n   = (int) ggml_nelements(logits);
    cudaStream_t    st  = ctx.stream();
    if (n <= DS3_THREADS*DS3_PER) {
        k_draft_sample_v3<<<1, DS3_THREADS, 0, st>>>(x, ids, k2, o, sp, rp, rnb, cp, n, top_k, top_p);
        return;
    }
    // v2 for longer rows: the smallest register list that holds top_k (the merged K contain the top_k)
    if (top_k <= 8) {
        k_draft_sample< 8><<<1, DRAFT_SAMPLE_THREADS, 0, st>>>(x, ids, k2, o, sp, rp, rnb, cp, n, top_k, top_p);
    } else if (top_k <= 16) {
        k_draft_sample<16><<<1, DRAFT_SAMPLE_THREADS, 0, st>>>(x, ids, k2, o, sp, rp, rnb, cp, n, top_k, top_p);
    } else if (top_k <= 20) {
        k_draft_sample<20><<<1, DRAFT_SAMPLE_THREADS, 0, st>>>(x, ids, k2, o, sp, rp, rnb, cp, n, top_k, top_p);
    } else {
        k_draft_sample<32><<<1, DRAFT_SAMPLE_THREADS, 0, st>>>(x, ids, k2, o, sp, rp, rnb, cp, n, top_k, top_p);
    }
}
