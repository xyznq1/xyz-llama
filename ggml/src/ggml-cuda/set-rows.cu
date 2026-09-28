#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "xyzkv-quant.cuh"
#include "attn-chain.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

// ---- xyzkv2 set_rows: GROUP_SIZE-element groups with WHT rotation + norm correction ----
//
// Same structure as xyzkv3 but 2-bit quantization only (no signs byte).

template <typename idx_t, int GROUP_SIZE>
__launch_bounds__(128)
static __global__ void k_set_rows_xyzkv2(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_xyzkv2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    static_assert(GROUP_SIZE == 128 || GROUP_SIZE == 64, "GROUP_SIZE must be 128 or 64");

    const int j = threadIdx.x;

    constexpr int blocks_per_group = GROUP_SIZE / QK_XYZKV2;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_xyzkv2_0 * dst_row_ptr = (block_xyzkv2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_xyzkv2_0 * blk_base    = dst_row_ptr + i_grp * blocks_per_group;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = GROUP_SIZE / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT ----
    if (GROUP_SIZE == 128) {
        x[j] *= XYZKV_WHT_SIGNS1[j];
    } else {
        x[j] *= XYZKV_WHT_SIGNS1_64[j];
    }
    __syncthreads();

#define WHT_STAGE_SHARED_T2(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T2(1)
    WHT_STAGE_SHARED_T2(2)
    WHT_STAGE_SHARED_T2(4)
    WHT_STAGE_SHARED_T2(8)
    WHT_STAGE_SHARED_T2(16)
    WHT_STAGE_SHARED_T2(32)
    if (GROUP_SIZE == 128) { WHT_STAGE_SHARED_T2(64) }
#undef WHT_STAGE_SHARED_T2

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) {
        x[j] = x[j] * inv_sqrt_group * XYZKV_WHT_SIGNS2[j];
    } else {
        x[j] = x[j] * inv_sqrt_group * XYZKV_WHT_SIGNS2_64[j];
    }
    __syncthreads();

    // ---- Step 5: Quantize element j to 2-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = xyzkv_nearest_centroid_2bit(rv);

    // ---- Step 6: Pack qs (warp-cooperative, no atomics) ----
    // Each warp handles 32 elements. With QK_XYZKV2 > WARP_SIZE, multiple warps
    // share one block and write to different byte offsets within it.
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;
    const int elem_in_block = j % QK_XYZKV2;
    block_xyzkv2_0 * blk = blk_base + (j / QK_XYZKV2);

    // Pack qs: 4 elements per byte, 2 bits each.
    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[elem_in_block / 4] = qs_byte;

    // No signs packing needed for xyzkv2

    // ---- Step 7: Reconstruction norm ----
    const float c = XYZKV_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm (one per xyzkv2 block) ----
    if (elem_in_block == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// ---- xyzkv2 tail kernel: straight 2-bit quantize without WHT rotation ----

template <typename idx_t>
static __global__ void k_set_rows_xyzkv2_tail(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_xyzkv2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int tail_size) {

    const int j = threadIdx.x;

    int64_t tmp = blockIdx.x;
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_xyzkv2_0 * dst_row_ptr = (block_xyzkv2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    const int64_t n_full = ne00 / QK_XYZKV2_GROUP;
    const int64_t tail_start = n_full * QK_XYZKV2_GROUP;
    block_xyzkv2_0 * blk_base = dst_row_ptr + n_full * (QK_XYZKV2_GROUP / QK_XYZKV2);

    // ---- Load ----
    const float val = src_row[tail_start + j];

    // ---- L2 norm ----
    const int n_warps = tail_size / WARP_SIZE;
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;

    __shared__ float warp_accum[4];
    float v2 = val * val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (lane == 0) warp_accum[warp_id] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Normalize (no WHT!) ----
    const float rv = val * inv_norm;

    // ---- Quantize ----
    const uint8_t idx = xyzkv_nearest_centroid_2bit(rv);

    // ---- Pack qs ----
    block_xyzkv2_0 * blk = blk_base + warp_id;

    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[lane / 4] = qs_byte;

    // ---- Reconstruction norm ----
    const float c = XYZKV_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (lane == 0) warp_accum[warp_id] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne00);
}

template<typename idx_t>
static void set_rows_cuda_xyzkv2(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_XYZKV2 == 0);

    cudaStream_t stream = ctx.stream();

    int group_size = 128;
    memcpy(&group_size, dst->op_params, sizeof(int));
    if (group_size != 64 && group_size != 128) group_size = 128;
    GGML_ASSERT(ne00 % group_size == 0);

    const int64_t n_full_groups   = ne00 / group_size;
    const int     tail_size       = (int)(ne00 % group_size);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (n_full_groups > 0) {
        const int64_t ne_total = n_full_groups * ne01 * ne02 * ne03;
        if (group_size == 128) {
            k_set_rows_xyzkv2<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
                src0_d, src1_d, (block_xyzkv2_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        } else {
            k_set_rows_xyzkv2<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
                src0_d, src1_d, (block_xyzkv2_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        }
    }

    if (tail_size > 0) {
        GGML_ASSERT(tail_size % QK_XYZKV2 == 0);
        const int64_t n_rows = ne01 * ne02 * ne03;
        k_set_rows_xyzkv2_tail<idx_t><<<(int)n_rows, tail_size, 0, stream>>>(
            src0_d, src1_d, (block_xyzkv2_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3, tail_size);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_XYZKV2_0) {
        set_rows_cuda_xyzkv2<idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

// Attention K and V cache writes use one launch each.
//
// K: RMS_NORM + MUL + ROPE (M-RoPE) + Hadamard-256 + SET_ROWS(xyzkv2) was four launches per attention layer (the
// norm+mul, rope_multi, fwht_cuda<256>, k_set_rows_xyzkv2<idx_t, 128>); V: Hadamard-64 + SET_ROWS(xyzkv2) two. One CTA of
// 256 threads per (token, kv head) row = two 128-groups of the token's K (V) row, the row in shared memory. Bit-identical
// by construction: the prologue is attn-chain.cuh's (the Q chain's, stage by stage), the V Hadamard fwht_cuda<64>'s
// (x*scale first, stages h = 1..32 inside each 64-chunk), and each 128-group is stored with k_set_rows_xyzkv2<idx_t, 128>'s
// own steps below -- the same warp-shuffle norm and in-order warp sum, the same WHT
// stages, centroid, packing and corrected norm.

// k_set_rows_xyzkv2<idx_t, 128> for the 128-group in x[0..127] (shared, loaded and synced):
// thread j (0..127) of the group; blk is the group's block. Both groups of the CTA call it together (CTA-wide syncs).
static __device__ __forceinline__ void attn_xyzkv2_store_group128(float * x, const int j, block_xyzkv2_0 * blk,
                                                                  float * warp_accum, float * s_norm_sq, float * s_recon_sq) {
    constexpr int GROUP_SIZE = 128;
    constexpr int n_warps    = GROUP_SIZE / WARP_SIZE;
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        *s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(*s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT ----
    x[j] *= XYZKV_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T2(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T2(1)
    WHT_STAGE_SHARED_T2(2)
    WHT_STAGE_SHARED_T2(4)
    WHT_STAGE_SHARED_T2(8)
    WHT_STAGE_SHARED_T2(16)
    WHT_STAGE_SHARED_T2(32)
    WHT_STAGE_SHARED_T2(64)
#undef WHT_STAGE_SHARED_T2

    constexpr float inv_sqrt_group = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_group * XYZKV_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Quantize element j to 2-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = xyzkv_nearest_centroid_2bit(rv);

    // ---- Step 6: Pack qs (one block per 128-group: QK_XYZKV2 == 128) ----
    const int lane = j % WARP_SIZE;
    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[j / 4] = qs_byte;

    // ---- Step 7: Reconstruction norm ----
    const float c = XYZKV_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        *s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(*s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm ----
    if (j == 0) blk->norm = __float2half(corrected_norm);
}

template <typename idx_t>
__launch_bounds__(256, 1)
static __global__ void k_attn_k_write_xyzkv2(
        const float * __restrict__ x, const int64_t sx1, const int64_t sx2, const float * __restrict__ w, const float eps,
        const int32_t * __restrict__ pos, const int ne02, const attn_mrope_params rp, const float fwht_scale,
        const idx_t * __restrict__ idx, const int64_t s10, char * __restrict__ dst, const int64_t nb1, const int n_heads) {
    static_assert(QK_XYZKV2 == 128, "one xyzkv2 block per 128-group");
    __shared__ float s[256];
    __shared__ float s_sum[32];
    __shared__ float warp_accum[2][4];
    __shared__ float s_norm_sq[2];
    __shared__ float s_recon_sq[2];

    const int row  = blockIdx.x;   // head + token*n_heads
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int half = threadIdx.x / 128;

    ggml_cuda_pdl_sync();
    attn_head_prologue_256(s, s_sum, x + tok*sx2 + head*sx1, w, eps, pos, tok, ne02, rp, fwht_scale);

    // groups head*2 + {0, 1} of the token's K row, in the cache row the token's index names
    const int64_t dst_row = idx[tok*s10];
    block_xyzkv2_0 * blk = (block_xyzkv2_0 *) (dst + dst_row*nb1) + head*2 + half;
    attn_xyzkv2_store_group128(s + half*128, threadIdx.x % 128, blk, warp_accum[half], &s_norm_sq[half], &s_recon_sq[half]);
}

template <typename idx_t>
__launch_bounds__(256, 1)
static __global__ void k_attn_v_write_xyzkv2(
        const float * __restrict__ v, const int64_t sv2, const float fwht_scale,
        const idx_t * __restrict__ idx, const int64_t s10, char * __restrict__ dst, const int64_t nb1, const int n_heads) {
    static_assert(QK_XYZKV2 == 128, "one xyzkv2 block per 128-group");
    __shared__ float s[256];
    __shared__ float warp_accum[2][4];
    __shared__ float s_norm_sq[2];
    __shared__ float s_recon_sq[2];

    const int row  = blockIdx.x;   // head + token*n_heads
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int tid  = threadIdx.x;
    const int half = tid / 128;

    // the V rotation's 64-point Hadamard on the head's four 64-chunks (fwht_cuda<64, false>)
    ggml_cuda_pdl_sync();
    s[tid] = v[tok*sv2 + head*256 + tid] * fwht_scale;
    __syncthreads();
#pragma unroll
    for (int h = 1; h < 64; h *= 2) {
        if (tid < 128) {
            const int e = (tid / h) * 2*h + (tid % h);
            const float a = s[e];
            const float b = s[e + h];
            s[e]     = a + b;
            s[e + h] = a - b;
        }
        __syncthreads();
    }

    const int64_t dst_row = idx[tok*s10];
    block_xyzkv2_0 * blk = (block_xyzkv2_0 *) (dst + dst_row*nb1) + head*2 + half;
    attn_xyzkv2_store_group128(s + half*128, tid % 128, blk, warp_accum[half], &s_norm_sq[half], &s_recon_sq[half]);
}

// the SET_ROWS node's xyzkv2 contract for a [n_heads*256, tokens] row set: group size 128, 1-D contiguous indices
static bool attn_xyzkv2_set_rows_ok(const ggml_tensor * set_rows, const int64_t n_cols, const int64_t n_tok) {
    const ggml_tensor * src0 = set_rows->src[0];
    const ggml_tensor * idx  = set_rows->src[1];
    int group_size = 128;
    memcpy(&group_size, set_rows->op_params, sizeof(int));
    if (group_size != 64 && group_size != 128) group_size = 128;   // set_rows_cuda_xyzkv2's reading
    return set_rows->type == GGML_TYPE_XYZKV2_0 && group_size == 128 && src0->type == GGML_TYPE_F32 &&
           src0->ne[0] == n_cols && src0->ne[1] == n_tok && src0->ne[2] == 1 && src0->ne[3] == 1 &&
           set_rows->ne[0] == n_cols && set_rows->ne[2] == 1 && set_rows->ne[3] == 1 &&
           (idx->type == GGML_TYPE_I64 || idx->type == GGML_TYPE_I32) && idx->ne[0] == n_tok &&
           idx->ne[1] == 1 && idx->ne[2] == 1 && idx->ne[3] == 1;
}

bool ggml_cuda_op_attn_k_write(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                               const ggml_tensor * rope, const ggml_tensor * had, ggml_tensor * set_rows) {
    const ggml_tensor * x   = rms_norm->src[0];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * idx = set_rows->src[1];
    constexpr int N = 256;
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 ||
            x->ne[0] != N || x->nb[0] != sizeof(float) || x->ne[2] < 1 || x->ne[2] > 16 || x->ne[3] != 1 ||
            !ggml_are_same_shape(rms_norm, x) || !ggml_are_same_shape(mul, x) || !ggml_are_same_shape(rope, x) ||
            w->ne[0] != N || ggml_nrows(w) != 1 || !ggml_is_contiguous(w) ||
            had->ne[0] != N || had->src[0]->ne[0] != N || had->src[0]->ne[1] != N ||
            ggml_nelements(had) != ggml_nelements(x) ||
            !attn_xyzkv2_set_rows_ok(set_rows, x->ne[0]*x->ne[1], x->ne[2])) {
        return false;
    }
    attn_mrope_params rp;
    if (!attn_mrope_params_get(rope, rp)) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const float fwht_scale = 1 / sqrtf(N);   // fwht_dispatch's

    const int n_heads = (int) x->ne[1];
    const int ne02    = (int) x->ne[2];
    const int64_t s10 = idx->nb[0] / ggml_type_size(idx->type);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_heads * ne02), 1, 1), dim3(N, 1, 1), 0, ctx.stream());
    if (idx->type == GGML_TYPE_I64) {
        ggml_cuda_kernel_launch(k_attn_k_write_xyzkv2<int64_t>, lp, (const float *) x->data,
            (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)), (const float *) w->data, eps,
            (const int32_t *) rope->src[1]->data, ne02, rp, fwht_scale, (const int64_t *) idx->data, s10,
            (char *) set_rows->data, (int64_t) set_rows->nb[1], n_heads);
    } else {
        ggml_cuda_kernel_launch(k_attn_k_write_xyzkv2<int32_t>, lp, (const float *) x->data,
            (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)), (const float *) w->data, eps,
            (const int32_t *) rope->src[1]->data, ne02, rp, fwht_scale, (const int32_t *) idx->data, s10,
            (char *) set_rows->data, (int64_t) set_rows->nb[1], n_heads);
    }
    return true;
}

bool ggml_cuda_op_attn_v_write(ggml_backend_cuda_context & ctx, const ggml_tensor * v, const ggml_tensor * had,
                               ggml_tensor * set_rows) {
    const ggml_tensor * idx = set_rows->src[1];
    if (v->type != GGML_TYPE_F32 || had->type != GGML_TYPE_F32 || !ggml_is_contiguous(v) ||
            v->ne[0] % 256 != 0 || v->ne[1] < 1 || v->ne[1] > 16 || v->ne[2] != 1 || v->ne[3] != 1 ||
            had->ne[0] != 64 || had->src[0]->ne[0] != 64 || had->src[0]->ne[1] != 64 ||
            ggml_nelements(had) != ggml_nelements(v) ||
            !attn_xyzkv2_set_rows_ok(set_rows, v->ne[0], v->ne[1])) {
        return false;
    }
    const float fwht_scale = 1 / sqrtf(64);   // fwht_dispatch's, n = 64

    const int n_heads = (int) (v->ne[0] / 256);
    const int n_tok   = (int) v->ne[1];
    const int64_t s10 = idx->nb[0] / ggml_type_size(idx->type);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_heads * n_tok), 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    if (idx->type == GGML_TYPE_I64) {
        ggml_cuda_kernel_launch(k_attn_v_write_xyzkv2<int64_t>, lp, (const float *) v->data,
            (int64_t) (v->nb[1] / sizeof(float)), fwht_scale, (const int64_t *) idx->data, s10,
            (char *) set_rows->data, (int64_t) set_rows->nb[1], n_heads);
    } else {
        ggml_cuda_kernel_launch(k_attn_v_write_xyzkv2<int32_t>, lp, (const float *) v->data,
            (int64_t) (v->nb[1] / sizeof(float)), fwht_scale, (const int32_t *) idx->data, s10,
            (char *) set_rows->data, (int64_t) set_rows->nb[1], n_heads);
    }
    return true;
}
