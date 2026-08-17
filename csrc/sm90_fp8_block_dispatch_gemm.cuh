#pragma once

// MoK-form producer/consumer fusion, first cut: pull dispatch and the gate/up
// contiguous grouped GEMM run inside ONE persistent kernel.  The leading
// clusters are communication CTAs (grid-stride pull of routed rows plus the
// input-publish barrier that used to be a separate kernel); the remaining
// clusters are the unchanged cluster-2 TMA GEMM, except each M64 tile first
// spins on a per-tile ready counter and therefore starts as soon as its own
// 64 rows have landed -- data flows, no dispatch/GEMM kernel boundary, no
// global wait.  GPU0's clock skew then delays only the tiles that contain its
// rows instead of the whole layer.
//
// Deadlock safety: communication clusters occupy the front of the grid so the
// first wave resident on the SMs always contains every producer; consumer
// spins use nanosleep backoff.  All ready/expected slots are zeroed by the
// dispatch-preparation phase of the same iteration, which keeps CUDA graph
// replay valid with no host-side counters.
#if defined(KITTENS_SM90)

#include "sm90_fp8_block_routed.cuh"
#include "sm90_fp8_block_worker_test.cuh"

namespace mok_sm90::fp8_block_dispatch_gemm {

using fp8_block_routed::MAX_EP_SIZE;
using fp8_block_test::a_st;
using fp8_block_test::b_st;
using fp8_block_test::d_st;
using fp8_block_test::acc_rt;

constexpr int THREADS = 128;
constexpr int MAX_LOCAL_EXPERTS = 256;  // smem expert-segment cache bound

struct globals {
    // --- GEMM (consumer) side, contiguous contract.  The gl layouts have no
    // default constructor, so they lead the struct and are the only members
    // provided at aggregate initialization; everything after them is
    // value-initialized there and assigned afterwards. ---
    fp8_block_test::contiguous::a_gl A;    // aliases routed_x
    fp8_block_test::contiguous::b_gl B;
    fp8_block_test::contiguous::d_gl D;
    // --- dispatch (producer) side ---
    const uint8_t *x_peer[MAX_EP_SIZE];
    const float *x_scale_peer[MAX_EP_SIZE];
    uint8_t *routed_x;
    float *routed_x_scale;
    int *m_indices;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    const int *tokens_per_expert;
    int ep_size;
    int num_local_tokens;
    int hidden_size;
    int scale_columns;
    int topk;
    int num_local_experts;
    int schedule_capacity;
    int copy_clusters;  // communication clusters at the front of the grid
    // input-publish barrier (absorbs the pre-dispatch barrier kernel)
    unsigned int *barrier_flag;
    unsigned int *barrier_target;
    unsigned int *barrier_multicast_ptr;
    unsigned int *input_expected_scratch;  // zeroed each iteration
    // producer->consumer handoff: per-M64-tile completed-row counters
    unsigned int *tile_ready;              // [capacity/64], zeroed each iter
    const float *A_scale;                  // aliases routed_x_scale
    const float *B_scale;
    int n;
    int k_blocks;
    int n_tiles;
};

__device__ __forceinline__ void copy_role(const globals &g) {
    const int copy_cta_count = g.copy_clusters * 2;
    const int copy_cta_idx = blockIdx.x;

    // Absorbed input barrier: the first CTA arrives for this rank (its
    // x_buffer copy precedes this kernel on the stream) and publishes the
    // expected flag value; every communication CTA then waits for all ranks.
    if (copy_cta_idx == 0 && threadIdx.x == 0) {
        const unsigned int expected =
            atomicAdd(g.barrier_target, static_cast<unsigned int>(g.ep_size))
            + static_cast<unsigned int>(g.ep_size);
        asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                     "l"(g.input_expected_scratch), "r"(expected) : "memory");
        asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" ::
                     "l"(g.barrier_multicast_ptr) : "memory");
        asm volatile("{fence.proxy.alias;}" ::: "memory");
    }
    if (threadIdx.x == 0) {
        unsigned int expected;
        do {
            asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                         : "=r"(expected)
                         : "l"(g.input_expected_scratch) : "memory");
            if (expected == 0u) __nanosleep(128);
        } while (expected == 0u);
        unsigned int value;
        do {
            asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}"
                         : "=r"(value) : "l"(g.barrier_flag) : "memory");
            if (value < expected) __nanosleep(128);
        } while (value < expected);
        asm volatile("{fence.acquire.sys;}" ::: "memory");
    }
    __syncthreads();

    // Cache the per-expert segment ends once per CTA: the per-row expert
    // lookup below then scans shared memory instead of issuing up to
    // num_local_experts uncached global loads for every row, which was the
    // dominant serial cost of the first version.
    __shared__ int expert_row_end[MAX_LOCAL_EXPERTS];
    if (threadIdx.x == 0) {
        int offset = 0;
        for (int e = 0; e < g.num_local_experts; ++e) {
            offset += g.tokens_per_expert[e];
            expert_row_end[e] = offset;
        }
    }
    __syncthreads();

    const int device_rows = g.num_tokens[0];
    const int valid_rows = device_rows < g.schedule_capacity
                               ? device_rows
                               : g.schedule_capacity;
    const int fp8_vectors = g.hidden_size / static_cast<int>(sizeof(uint4));
    const uint4 zero{0, 0, 0, 0};

    // One row per warp: rows fly concurrently with no block-wide barrier in
    // the loop.  __syncwarp orders the lanes' row stores before lane 0's
    // release-increment (gpu scope suffices -- the consumer GEMM CTAs are on
    // this device), so each tile_ready add still carries its full row.
    constexpr int WARPS = THREADS / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int row = copy_cta_idx * WARPS + warp; row < valid_rows;
         row += copy_cta_count * WARPS) {
        const int peer_rank = g.schedule_peer_rank[row];
        const int peer_token_idx = g.schedule_peer_token_idx[row];
        const bool valid = peer_rank >= 0 && peer_rank < g.ep_size
                           && peer_token_idx >= 0
                           && peer_token_idx < g.num_local_tokens * g.topk;
        auto *dst_vectors = reinterpret_cast<uint4 *>(g.routed_x)
                            + static_cast<size_t>(row) * fp8_vectors;
        float *dst_scale = g.routed_x_scale
                           + static_cast<size_t>(row) * g.scale_columns;
        if (valid) {
            const int source_row = peer_token_idx / g.topk;
            const auto *src_vectors =
                reinterpret_cast<const uint4 *>(g.x_peer[peer_rank])
                + static_cast<size_t>(source_row) * fp8_vectors;
            const float *src_scale =
                g.x_scale_peer[peer_rank]
                + static_cast<size_t>(source_row) * g.scale_columns;
            #pragma unroll 4
            for (int i = lane; i < fp8_vectors; i += 32)
                dst_vectors[i] = src_vectors[i];
            for (int i = lane; i < g.scale_columns; i += 32)
                dst_scale[i] = src_scale[i];
        } else {
            #pragma unroll 4
            for (int i = lane; i < fp8_vectors; i += 32)
                dst_vectors[i] = zero;
            for (int i = lane; i < g.scale_columns; i += 32)
                dst_scale[i] = 0.0f;
        }
        if (lane == 0) {
            int expert = 0;
            while (expert < g.num_local_experts - 1
                   && row >= expert_row_end[expert])
                ++expert;
            g.m_indices[row] = expert;
        }
        __syncwarp();
        if (lane == 0) {
            asm volatile("{red.release.gpu.global.add.u32 [%0], 1;}" ::
                         "l"(g.tile_ready + (row >> 6)) : "memory");
        }
    }
}

__device__ __forceinline__ void gemm_role(const globals &g,
                                          int gemm_cluster_idx) {
    const int cta_rank = cluster_ctarank();
    const int n_pairs = g.n_tiles / 2;
    const int n_tile_base = 2 * (gemm_cluster_idx % n_pairs);
    const int n_tile = n_tile_base + cta_rank;
    const int m_tile = gemm_cluster_idx / n_pairs;
    const int global_row_base = m_tile * 64;
    if (global_row_base >= g.num_tokens[0])
        return;

    // Consume as soon as this tile's own rows have been dispatched.  The
    // producer publishes m_indices before its release-increment, so the
    // expert lookup below is ordered by the acquire spin.
    if (threadIdx.x == 0) {
        unsigned int done;
        do {
            asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                         : "=r"(done)
                         : "l"(g.tile_ready + m_tile) : "memory");
            if (done < 64u) __nanosleep(256);
        } while (done < 64u);
    }
    __syncthreads();

    const int expert = g.m_indices[global_row_base];

    extern __shared__ int __shm[];
    shared_allocator al((int *)&__shm[0]);
    constexpr int PIPE_DEPTH = 2;
    auto &a_smem = al.allocate<a_st, PIPE_DEPTH>();
    auto &b_smem = al.allocate<b_st, PIPE_DEPTH>();
    d_st &d_smem = al.allocate<d_st>();
    __shared__ semaphore inputs_arrived[PIPE_DEPTH];
    __shared__ semaphore inputs_finished[PIPE_DEPTH];
    __shared__ semaphore inputs_ready[PIPE_DEPTH];

    if (threadIdx.x < PIPE_DEPTH) {
        init_semaphore(inputs_arrived[threadIdx.x], 0, 1);
        init_semaphore(inputs_finished[threadIdx.x], 0, 1);
        init_semaphore(inputs_ready[threadIdx.x], 0, 2);
    }
    everyone::tma::cluster::sync();

    acc_rt total;
    uint32_t phasebits = 0xFFFF0000;
    uint32_t ready_phase = 0;
    if (threadIdx.x == 0) {
        wait(inputs_finished[0], get_phasebit<1>(phasebits, 0));
        update_phasebit<1>(phasebits, 0);
        tma::cluster::expect_bytes(
            inputs_arrived[0], sizeof(a_st) + sizeof(b_st));
        tma::cluster::load_async(
            b_smem[0], g.B, {expert, n_tile, 0}, inputs_arrived[0],
            static_cast<uint16_t>(1 << cta_rank));
        tma::cluster::arrive(inputs_ready[0], 0);
        if (cta_rank == 0) {
            wait(inputs_ready[0], get_phasebit<0>(ready_phase, 0));
            update_phasebit<0>(ready_phase, 0);
            tma::cluster::load_async(
                a_smem[0], g.A, {m_tile, 0}, inputs_arrived[0], 0b11);
        }
    }
    for (int kb = 0; kb < g.k_blocks; ++kb) {
        const int stage = kb % PIPE_DEPTH;
        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
        update_phasebit<0>(phasebits, stage);

        acc_rt partial;
        warpgroup::mm_ABt(partial, a_smem[stage], b_smem[stage]);
        if (kb + 1 < g.k_blocks) {
            const int next_stage = (kb + 1) % PIPE_DEPTH;
            if (threadIdx.x == 0) {
                wait(inputs_finished[next_stage],
                     get_phasebit<1>(phasebits, next_stage));
                update_phasebit<1>(phasebits, next_stage);
                tma::cluster::expect_bytes(
                    inputs_arrived[next_stage],
                    sizeof(a_st) + sizeof(b_st));
                tma::cluster::load_async(
                    b_smem[next_stage], g.B,
                    {expert, n_tile, kb + 1}, inputs_arrived[next_stage],
                    static_cast<uint16_t>(1 << cta_rank));
                tma::cluster::arrive(inputs_ready[next_stage], 0);
                if (cta_rank == 0) {
                    wait(inputs_ready[next_stage],
                         get_phasebit<0>(ready_phase, next_stage));
                    update_phasebit<0>(ready_phase, next_stage);
                    tma::cluster::load_async(
                        a_smem[next_stage], g.A, {m_tile, kb + 1},
                        inputs_arrived[next_stage], 0b11);
                }
            }
        }
        warpgroup::mma_async_wait<0>();

        typename acc_rt::col_vec row_scale;
        const int local_row = warpid() * 16 + laneid() / 4;
        const int global_row = global_row_base + local_row;
        const float b_scale =
            g.B_scale[(expert * (g.n / 128) + n_tile / 2) * g.k_blocks + kb];
        row_scale[0][0].x =
            g.A_scale[global_row * g.k_blocks + kb] * b_scale;
        row_scale[0][0].y =
            g.A_scale[(global_row + 8) * g.k_blocks + kb] * b_scale;
        warpgroup::mul_row(partial, partial, row_scale);

        if (kb == 0)
            warp::copy(total, partial);
        else
            warpgroup::add(total, total, partial);
        warpgroup::sync(0);
        if (threadIdx.x == 0)
            tma::cluster::arrive(inputs_finished[stage], cta_rank);
    }

    rt_bf<16, 64> out;
    warp::copy(out, total);
    warpgroup::store(d_smem, out);
    warpgroup::sync(0);
    warpgroup::store(g.D, d_smem, {m_tile, n_tile});
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void kernel(const __grid_constant__ globals g) {
    const int cluster_idx = clusterIdx().x;
    if (cluster_idx < g.copy_clusters)
        copy_role(g);
    else
        gemm_role(g, cluster_idx - g.copy_clusters);
}


inline void entry_out(
    const at::Tensor &x_buffer, const std::vector<int64_t> &x_ptrs,
    const at::Tensor &x_scale_buffer,
    const std::vector<int64_t> &x_scale_ptrs,
    const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
    const at::Tensor &m_indices, const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx, const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert, int64_t topk,
    const at::Tensor &barrier_buffer, int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &input_expected_scratch, const at::Tensor &tile_ready,
    const at::Tensor &B, const at::Tensor &B_scale, const at::Tensor &D,
    int64_t copy_clusters) {
    // Dispatch-side contracts are identical to the split path; reuse them.
    fp8_block_routed::check_pointer_list(x_ptrs, "x_ptrs");
    fp8_block_routed::check_pointer_list(x_scale_ptrs, "x_scale_ptrs");
    TORCH_CHECK(x_ptrs.size() == x_scale_ptrs.size(),
                "x and scale pointer lists must have equal length");
    const int64_t schedule_capacity = routed_x.size(0);
    fp8_block_routed::check_schedule(
        schedule_peer_rank, schedule_peer_token_idx, num_tokens,
        tokens_per_expert, schedule_capacity);
    TORCH_CHECK(routed_x.dim() == 2 && routed_x.is_cuda()
                    && routed_x.scalar_type() == at::kFloat8_e4m3fn
                    && routed_x.is_contiguous(),
                "routed_x must be contiguous CUDA float8_e4m3fn");
    const int64_t hidden_size = routed_x.size(1);
    TORCH_CHECK(schedule_capacity % 64 == 0 && hidden_size % 128 == 0
                    && hidden_size >= 128,
                "routed_x must be M64 x K128 aligned");
    TORCH_CHECK(x_buffer.dim() == 2 && x_buffer.size(1) == hidden_size
                    && x_buffer.scalar_type() == at::kFloat8_e4m3fn
                    && x_buffer.is_contiguous() && x_buffer.is_cuda(),
                "x_buffer must be contiguous CUDA FP8 [T,H]");
    TORCH_CHECK(x_scale_buffer.dim() == 2
                    && x_scale_buffer.size(0) == x_buffer.size(0)
                    && x_scale_buffer.size(1) == hidden_size / 128
                    && x_scale_buffer.scalar_type() == at::kFloat
                    && x_scale_buffer.is_contiguous(),
                "x_scale_buffer must be contiguous float32 [T,H/128]");
    TORCH_CHECK(m_indices.numel() == schedule_capacity
                    && m_indices.scalar_type() == at::kInt,
                "m_indices must be int32 [capacity]");
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    TORCH_CHECK(routed_x_scale.dim() == 2
                    && routed_x_scale.size(0) == schedule_capacity
                    && routed_x_scale.size(1) == hidden_size / 128
                    && routed_x_scale.scalar_type() == at::kFloat
                    && routed_x_scale.is_contiguous(),
                "routed_x_scale must be contiguous float32 [capacity,H/128]");
    for (const at::Tensor *t :
         {&barrier_buffer, &barrier_target, &input_expected_scratch}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt
                        && t->is_contiguous() && t->numel() == 1,
                    "barrier state tensors must be int32 [1]");
    }
    TORCH_CHECK(barrier_buffer_multicast_ptr > 0,
                "barrier multicast pointer must be positive");
    TORCH_CHECK(tile_ready.is_cuda() && tile_ready.scalar_type() == at::kInt
                    && tile_ready.is_contiguous()
                    && tile_ready.numel() == schedule_capacity / 64,
                "tile_ready must be int32 [capacity/64]");
    // GEMM-side contracts mirror the contiguous entry.
    TORCH_CHECK(B.dim() == 3 && B.scalar_type() == at::kFloat8_e4m3fn
                    && B.is_contiguous() && B.size(2) == hidden_size,
                "B must be contiguous FP8 [E,N,K]");
    const int experts = (int)B.size(0);
    const int n = (int)B.size(1);
    const int k_blocks = (int)(hidden_size / 128);
    TORCH_CHECK(n >= 128 && n % 128 == 0, "N must be a positive K128 multiple");
    TORCH_CHECK(B_scale.dim() == 3 && B_scale.size(0) == experts
                    && B_scale.size(1) == n / 128
                    && B_scale.size(2) == k_blocks
                    && B_scale.scalar_type() == at::kFloat
                    && B_scale.is_contiguous(),
                "B_scale must be contiguous float32 [E,N/128,K/128]");
    TORCH_CHECK(D.dim() == 2 && D.size(0) == schedule_capacity
                    && D.size(1) == n
                    && D.scalar_type() == at::kBFloat16
                    && D.is_contiguous() && D.is_cuda(),
                "D must be contiguous BF16 [capacity,N]");
    TORCH_CHECK(copy_clusters > 0 && copy_clusters <= 32,
                "copy_clusters must be in [1,32]");
    TORCH_CHECK(tokens_per_expert.numel() <= MAX_LOCAL_EXPERTS,
                "local expert count exceeds the smem segment cache");
    kittens::py::device_check(routed_x, routed_x_scale, m_indices, B, B_scale);
    kittens::py::device_check(routed_x, D);

    c10::cuda::CUDAGuard device_guard(routed_x.device());
    globals g{
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::a_gl>(
            const_cast<at::Tensor &>(routed_x)),
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::b_gl>(
            const_cast<at::Tensor &>(B)),
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::d_gl>(
            const_cast<at::Tensor &>(D)),
    };
    for (size_t rank = 0; rank < x_ptrs.size(); ++rank) {
        g.x_peer[rank] = reinterpret_cast<const uint8_t *>(x_ptrs[rank]);
        g.x_scale_peer[rank] =
            reinterpret_cast<const float *>(x_scale_ptrs[rank]);
    }
    g.routed_x = reinterpret_cast<uint8_t *>(routed_x.data_ptr());
    g.routed_x_scale = routed_x_scale.data_ptr<float>();
    g.m_indices = m_indices.data_ptr<int>();
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    g.ep_size = static_cast<int>(x_ptrs.size());
    g.num_local_tokens = static_cast<int>(x_buffer.size(0));
    g.hidden_size = static_cast<int>(hidden_size);
    g.scale_columns = static_cast<int>(hidden_size / 128);
    g.topk = static_cast<int>(topk);
    g.num_local_experts = static_cast<int>(tokens_per_expert.numel());
    g.schedule_capacity = static_cast<int>(schedule_capacity);
    g.copy_clusters = static_cast<int>(copy_clusters);
    g.barrier_flag =
        reinterpret_cast<unsigned int *>(barrier_buffer.data_ptr<int>());
    g.barrier_target =
        reinterpret_cast<unsigned int *>(barrier_target.data_ptr<int>());
    g.barrier_multicast_ptr =
        reinterpret_cast<unsigned int *>(barrier_buffer_multicast_ptr);
    g.input_expected_scratch = reinterpret_cast<unsigned int *>(
        input_expected_scratch.data_ptr<int>());
    g.tile_ready =
        reinterpret_cast<unsigned int *>(tile_ready.data_ptr<int>());
    g.A_scale = routed_x_scale.data_ptr<float>();
    g.B_scale = B_scale.data_ptr<float>();
    g.n = n;
    g.k_blocks = k_blocks;
    g.n_tiles = n / 64;

    const int m_tiles = static_cast<int>(schedule_capacity / 64);
    const int total_ctas =
        static_cast<int>(copy_clusters) * 2 + m_tiles * g.n_tiles;
    constexpr int PIPE_DEPTH = 2;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(routed_x.get_device());
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    kernel<<<total_ctas, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::fp8_block_dispatch_gemm

#endif
