#pragma once

// Fused W13 epilogue for the warprole kernel: SwiGLU with clamp + per-K128
// FP8 quantization, computed straight from the gate and up tiles a task just
// produced.  The arithmetic is the production sequence (bf16 round -> bf16
// clamp -> fp32 silu*up -> group absmax/448 -> e4m3 rn.satfinite), copied from
// fp8_block_pipeline::activate_quant_worker, which reproduces SGLang's
// silu_and_mul_contig_post_quant bit for bit.
//
// Weight layout: the plain [E, 2I, K] gate/up weight as the model loads it
// (gate rows first, then up rows).  Intermediate tile j is N tile j (gate) and
// N tile I/128 + j (up) of that tensor; nothing is copied or reordered.
#if defined(KITTENS_SM90)
#include <ATen/ATen.h>

#include "sm90_fp8_block_warprole_gemm.cuh"

namespace mok_sm90::warprole::epilogue {
using namespace kittens;
namespace pipeline = mok_sm90::fp8_block_pipeline;
using gemm::d_st;
using gemm::acc_rt;
using gemm::smem_layout;

constexpr int INTER_GROUPS = INTER / K_TILE;   // 16 activation scale groups per row
constexpr int UP_TILE_OFFSET = INTER / N_TILE;  // up tile j is N tile 16 + j of w13

// Barrier over the consumer threads only (thread ids [0, 128*NC)).
template <int NC> __device__ __forceinline__ void consumers_sync() {
    if constexpr (NC == 1) warpgroup::sync(1);
    else asm volatile("bar.sync 3, 256;" ::: "memory");
}

// EPI_THREADS threads (t = 0..EPI_THREADS-1, whole warps) call this after the
// bf16 gate tile sits in `gate_tile`, the bf16 up tile in `up_tile`, and the
// caller's barrier passed.  Thread t handles 8 consecutive columns of one row
// per pass; 16 threads share a row so the K128 absmax is a 16-lane shuffle
// reduction.  EPI_THREADS need not divide the tile: the last pass is guarded
// (the guard is uniform over each 16-lane group, so the shuffle stays whole).
template <int EPI_THREADS>
__device__ __forceinline__ void swiglu_quant_tile(
        int t, d_st &gate_tile, d_st &up_tile, uint8_t *hidden, float *hidden_scale,
        int m_tile, int i_tile, float limit) {
    static_assert(EPI_THREADS % 32 == 0);
    constexpr int THREADS_PER_ROW = N_TILE / 8;                    // 16
    constexpr int ROWS_PER_PASS = EPI_THREADS / THREADS_PER_ROW;   // 6, 8 or 16
    constexpr int PASSES = (M_TILE + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    const int sub = t % THREADS_PER_ROW;
    const int col0 = sub * 8;
    const __nv_bfloat162 limit2 = __floats2bfloat162_rn(limit, limit);
    const __nv_bfloat162 neg_limit2 = __floats2bfloat162_rn(-limit, -limit);
#pragma unroll 1
    for (int pass = 0; pass < PASSES; ++pass) {
        const int local_row = pass * ROWS_PER_PASS + t / THREADS_PER_ROW;
        const bool valid = local_row < M_TILE;
        const int load_row = valid ? local_row : 0;
        const uint4 gate_raw = *reinterpret_cast<const uint4 *>(
            d_st::idx(gate_tile.data, {load_row, col0}));
        const uint4 up_raw = *reinterpret_cast<const uint4 *>(
            d_st::idx(up_tile.data, {load_row, col0}));
        const __nv_bfloat162 *gate_pairs = reinterpret_cast<const __nv_bfloat162 *>(&gate_raw);
        const __nv_bfloat162 *up_pairs = reinterpret_cast<const __nv_bfloat162 *>(&up_raw);
        float values[8];
        float local_max = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __nv_bfloat162 gate = __hmin2(gate_pairs[i], limit2);
            __nv_bfloat162 up = __hmax2(up_pairs[i], neg_limit2);
            up = __hmin2(up, limit2);
            const float2 gate_f = __bfloat1622float2(gate);
            const float2 up_f = __bfloat1622float2(up);
            const float x = gate_f.x / (1.0f + __expf(-gate_f.x)) * up_f.x;
            const float y = gate_f.y / (1.0f + __expf(-gate_f.y)) * up_f.y;
            values[2 * i] = x;
            values[2 * i + 1] = y;
            local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
        }
        const float absmax = fmaxf(pipeline::subgroup_max_16(local_max), 1e-10f);
        const float scale = absmax / pipeline::V4_FP8_MAX;
        const float inv_scale = 1.0f / scale;
        uint32_t packed[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t lo = pipeline::pack_v4_fp8x2(values[4 * i] * inv_scale, values[4 * i + 1] * inv_scale);
            const uint32_t hi = pipeline::pack_v4_fp8x2(values[4 * i + 2] * inv_scale, values[4 * i + 3] * inv_scale);
            packed[i] = lo | (hi << 16);
        }
        if (!valid) continue;
        const size_t row = static_cast<size_t>(m_tile) * M_TILE + local_row;
        *reinterpret_cast<uint2 *>(hidden + row * INTER + i_tile * N_TILE + col0) =
            make_uint2(packed[0], packed[1]);
        if (sub == 0) hidden_scale[row * INTER_GROUPS + i_tile] = scale;
    }
}

// Consumer-side W13 task: gate and up M64 x N128 tiles for intermediate tile
// `i_tile`, then the fused epilogue.  NC=2 computes gate on consumer 0 and up on
// consumer 1 in parallel; NC=1 computes them back to back.
template <int NC, int STAGES>
__device__ __forceinline__ void consumer_w13_task(
        smem_layout<NC, STAGES> &smem, semaphore (&full)[STAGES], semaphore (&empty)[STAGES],
        int64_t &stage_counter, int role, const float *B_scale, int expert, int m_tile, int i_tile,
        uint8_t *hidden, float *hidden_scale, float limit) {
    constexpr int N_TILES_128 = 2 * INTER / N_TILE;   // 32 N128 tiles: 16 gate, then 16 up
    constexpr int K_BLOCKS = W13_K_BLOCKS;
    if constexpr (NC == 2) {
        const int n_tile = i_tile + role * UP_TILE_OFFSET;
        gemm::stage_b_scale_row<NC, STAGES>(smem, role, B_scale, expert, N_TILES_128, n_tile, K_BLOCKS);
        warpgroup::sync(role + 1);
        acc_rt acc;
        gemm::consumer_task<NC, STAGES>(smem, full, empty, stage_counter, role, smem.b_scale[role], K_BLOCKS, acc);
        rt_bf<16, N_TILE> out;
        warp::copy(out, acc);
        warpgroup::store(smem.d[role], out);
    } else {
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
            const int n_tile = i_tile + half * UP_TILE_OFFSET;
            gemm::stage_b_scale_row<NC, STAGES>(smem, half, B_scale, expert, N_TILES_128, n_tile, K_BLOCKS);
            warpgroup::sync(1);
            acc_rt acc;
            gemm::consumer_task<NC, STAGES>(smem, full, empty, stage_counter, 0, smem.b_scale[half], K_BLOCKS, acc);
            rt_bf<16, N_TILE> out;
            warp::copy(out, acc);
            warpgroup::store(smem.d[half], out);
        }
    }
    consumers_sync<NC>();   // gate and up tiles visible to every consumer thread
    swiglu_quant_tile<128 * NC>(threadIdx.x, smem.d[0], smem.d[1], hidden, hidden_scale, m_tile, i_tile, limit);
    consumers_sync<NC>();   // nobody still reads d[] when the next task overwrites it
}

// Fused-kernel split of the W13 task: the consumers only run the mainloop(s) and
// park the bf16 gate/up tiles in smem.d[]; the epilogue runs on other warps.
// `wait_d_empty` is called once, right before the first store into d[].
template <int NC, int STAGES, typename WaitEmpty>
__device__ __forceinline__ void consumer_w13_gemm_to_d(
        smem_layout<NC, STAGES> &smem, semaphore (&full)[STAGES], semaphore (&empty)[STAGES],
        int64_t &stage_counter, int role, const float *B_scale, int expert, int i_tile, WaitEmpty wait_d_empty) {
    constexpr int N_TILES_128 = 2 * INTER / N_TILE;
    constexpr int K_BLOCKS = W13_K_BLOCKS;
    if constexpr (NC == 2) {
        const int n_tile = i_tile + role * UP_TILE_OFFSET;
        gemm::stage_b_scale_row<NC, STAGES>(smem, role, B_scale, expert, N_TILES_128, n_tile, K_BLOCKS);
        warpgroup::sync(role + 1);
        acc_rt acc;
        gemm::consumer_task<NC, STAGES>(smem, full, empty, stage_counter, role, smem.b_scale[role], K_BLOCKS, acc);
        rt_bf<16, N_TILE> out;
        warp::copy(out, acc);
        wait_d_empty();
        warpgroup::store(smem.d[role], out);
    } else {
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
            const int n_tile = i_tile + half * UP_TILE_OFFSET;
            gemm::stage_b_scale_row<NC, STAGES>(smem, half, B_scale, expert, N_TILES_128, n_tile, K_BLOCKS);
            warpgroup::sync(1);
            acc_rt acc;
            gemm::consumer_task<NC, STAGES>(smem, full, empty, stage_counter, 0, smem.b_scale[half], K_BLOCKS, acc);
            rt_bf<16, N_TILE> out;
            warp::copy(out, acc);
            if (half == 0) wait_d_empty();
            warpgroup::store(smem.d[half], out);
        }
    }
}

// Producer-side W13 task: the same stage sequence the consumers consume.
template <int NC, int STAGES, typename A_GL, typename B_GL>
__device__ __forceinline__ void producer_w13_task(
        const A_GL &A, const float *A_scale, const B_GL &B, smem_layout<NC, STAGES> &smem,
        semaphore (&full)[STAGES], semaphore (&empty)[STAGES], int64_t &stage_counter,
        int m_tile, int expert, int i_tile) {
    if constexpr (NC == 2) {
        gemm::producer_task<NC, STAGES>(A, A_scale, B, smem, full, empty, stage_counter,
                                        m_tile, expert, i_tile, W13_K_BLOCKS, UP_TILE_OFFSET);
    } else {
        gemm::producer_task<NC, STAGES>(A, A_scale, B, smem, full, empty, stage_counter,
                                        m_tile, expert, i_tile, W13_K_BLOCKS);
        gemm::producer_task<NC, STAGES>(A, A_scale, B, smem, full, empty, stage_counter,
                                        m_tile, expert, UP_TILE_OFFSET + i_tile, W13_K_BLOCKS);
    }
}

// ---------------------------------------------------------------------------
// Standalone fused W13 kernel: tasks are (m_tile, i_tile) over contiguous rows.
// ---------------------------------------------------------------------------
namespace standalone {

struct globals {
    gemm::a_gl A;        // [M, 4096] fp8
    gemm::b_gl B;        // [E, 4096, 4096] fp8, gate rows then up rows
    const float *A_scale;
    const float *B_scale;
    const int *m_indices;
    const int *num_tokens;
    uint8_t *hidden;      // [M, 2048] fp8 out
    float *hidden_scale;  // [M, 16] out
    float limit;
    int m_tiles;
};

template <int NC, int STAGES, int CTAS_PER_SM>
__global__ __launch_bounds__(gemm::num_threads<NC>(), CTAS_PER_SM)
void w13_kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    auto &smem = *reinterpret_cast<smem_layout<NC, STAGES> *>(
        ((reinterpret_cast<uint64_t>(&__shm[0])) + 1023) & ~static_cast<uint64_t>(1023));
    __shared__ semaphore full[STAGES];
    __shared__ semaphore empty[STAGES];
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_semaphore(full[s], 1, 1);
            init_semaphore(empty[s], NC * 4, 0);
        }
    }
    __syncthreads();

    constexpr int TASKS_PER_M = INTER / N_TILE;   // 16
    const int role = warpgroup::groupid();
    const int64_t total = static_cast<int64_t>(g.m_tiles) * TASKS_PER_M;
    int64_t stage_counter = 0;

    if (role == NC + 1) {
        warpgroup::decrease_registers<gemm::comm_regs<CTAS_PER_SM>()>();
        return;
    }
    if (role == NC) {
        warpgroup::decrease_registers<gemm::producer_regs<CTAS_PER_SM>()>();
        if (warpgroup::warpid() != 0) return;
        for (int64_t t = blockIdx.x; t < total; t += gridDim.x) {
            const int m_tile = static_cast<int>(t / TASKS_PER_M);
            if (g.num_tokens != nullptr && m_tile * M_TILE >= g.num_tokens[0]) break;
            const int i_tile = static_cast<int>(t % TASKS_PER_M);
            const int expert = g.m_indices[m_tile * M_TILE];
            producer_w13_task<NC, STAGES>(g.A, g.A_scale, g.B, smem, full, empty, stage_counter,
                                          m_tile, expert, i_tile);
        }
        return;
    }

    warpgroup::increase_registers<gemm::consumer_regs<NC, CTAS_PER_SM>()>();
    for (int64_t t = blockIdx.x; t < total; t += gridDim.x) {
        const int m_tile = static_cast<int>(t / TASKS_PER_M);
        if (g.num_tokens != nullptr && m_tile * M_TILE >= g.num_tokens[0]) break;
        const int i_tile = static_cast<int>(t % TASKS_PER_M);
        const int expert = g.m_indices[m_tile * M_TILE];
        consumer_w13_task<NC, STAGES>(smem, full, empty, stage_counter, role, g.B_scale, expert,
                                      m_tile, i_tile, g.hidden, g.hidden_scale, g.limit);
    }
}

template <int NC, int STAGES, int CTAS_PER_SM>
inline void entry_w13_out(at::Tensor A, at::Tensor A_scale, at::Tensor W13, at::Tensor W13_scale,
                          at::Tensor m_indices, at::Tensor num_tokens, at::Tensor hidden,
                          at::Tensor hidden_scale, double swiglu_limit) {
    TORCH_CHECK(A.dim() == 2 && W13.dim() == 3, "A must be [M,K] and W13 [E,2I,K]");
    kittens::py::tensor_check<gemm::a_gl>(A);
    kittens::py::tensor_check<gemm::b_gl>(W13);
    const int total_m = (int)A.size(0);
    const int experts = (int)W13.size(0);
    TORCH_CHECK(A.size(1) == HIDDEN && W13.size(2) == HIDDEN, "K must be 4096");
    TORCH_CHECK(W13.size(1) == 2 * INTER, "W13 must have 2*2048 rows per expert");
    TORCH_CHECK(total_m >= 64 && total_m % 64 == 0, "M must be positive and divisible by 64");
    TORCH_CHECK(A_scale.is_cuda() && W13_scale.is_cuda() && m_indices.is_cuda(),
                "scales and m_indices must be CUDA tensors");
    TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Float
                    && W13_scale.scalar_type() == at::ScalarType::Float,
                "scales must be float32");
    TORCH_CHECK(m_indices.scalar_type() == at::ScalarType::Int, "m_indices must be int32");
    TORCH_CHECK(A_scale.is_contiguous() && W13_scale.is_contiguous() && m_indices.is_contiguous(),
                "scales and m_indices must be contiguous");
    TORCH_CHECK(A_scale.dim() == 2 && A_scale.size(0) == total_m && A_scale.size(1) == W13_K_BLOCKS,
                "A_scale must have shape [M,32]");
    TORCH_CHECK(W13_scale.dim() == 3 && W13_scale.size(0) == experts
                    && W13_scale.size(1) == 2 * INTER / 128 && W13_scale.size(2) == W13_K_BLOCKS,
                "W13_scale must have shape [E,32,32]");
    TORCH_CHECK(m_indices.dim() == 1 && m_indices.size(0) == total_m, "m_indices must have shape [M]");
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::Float8_e4m3fn
                    && hidden.is_contiguous() && hidden.dim() == 2 && hidden.size(0) == total_m
                    && hidden.size(1) == INTER,
                "hidden must be a contiguous CUDA float8_e4m3fn [M,2048] tensor");
    TORCH_CHECK(hidden_scale.is_cuda() && hidden_scale.scalar_type() == at::ScalarType::Float
                    && hidden_scale.is_contiguous() && hidden_scale.dim() == 2
                    && hidden_scale.size(0) == total_m && hidden_scale.size(1) == INTER_GROUPS,
                "hidden_scale must be a contiguous CUDA float32 [M,16] tensor");
    TORCH_CHECK(num_tokens.is_cuda() && num_tokens.scalar_type() == at::ScalarType::Int
                    && num_tokens.is_contiguous() && num_tokens.dim() == 1 && num_tokens.numel() == 1,
                "num_tokens must be contiguous CUDA int32 [1]");
    kittens::py::device_check(A, W13, A_scale, W13_scale, m_indices);
    kittens::py::device_check(A, hidden, hidden_scale, num_tokens);

    c10::cuda::CUDAGuard device_guard(A.device());
    const int m_tiles = total_m / 64;
    globals g{
        kittens::py::tensor_to_gl<gemm::a_gl>(A),
        kittens::py::tensor_to_gl<gemm::b_gl>(W13),
        A_scale.data_ptr<float>(),
        W13_scale.data_ptr<float>(),
        m_indices.data_ptr<int>(),
        num_tokens.data_ptr<int>(),
        static_cast<uint8_t *>(hidden.data_ptr()),
        hidden_scale.data_ptr<float>(),
        static_cast<float>(swiglu_limit),
        m_tiles,
    };
    constexpr int SMEM = sizeof(smem_layout<NC, STAGES>) + 1024;
    constexpr int THREADS = gemm::num_threads<NC>();
    auto *kernel_ptr = w13_kernel<NC, STAGES, CTAS_PER_SM>;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    CUDACHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int num_sms = 0;
    CUDACHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, A.get_device()));
    int blocks_per_sm = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr, THREADS, SMEM));
    TORCH_CHECK(blocks_per_sm >= CTAS_PER_SM, "warprole w13 occupancy below target: ", blocks_per_sm,
                " < ", CTAS_PER_SM, " CTAs per SM (smem ", SMEM, " bytes)");
    const int64_t total_tasks = static_cast<int64_t>(m_tiles) * (INTER / N_TILE);
    const int grid = static_cast<int>(std::min<int64_t>(total_tasks, static_cast<int64_t>(num_sms) * CTAS_PER_SM));
    kernel_ptr<<<grid, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace standalone
}  // namespace mok_sm90::warprole::epilogue
#endif  // KITTENS_SM90
