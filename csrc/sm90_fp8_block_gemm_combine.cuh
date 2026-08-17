#pragma once

// MoK-form producer/consumer fusion, second symmetric cut: the down (w2)
// contiguous grouped GEMM and the combine push run inside ONE persistent
// kernel.  The leading clusters are communication CTAs that spin on a
// per-M64-tile completion counter and push each finished 64-row block to its
// peers' combine buffers (one warp per row stream, stores are fire-and-forget
// so no register staging is needed on this side); the remaining clusters are
// the unchanged cluster-2 TMA GEMM, each publishing tile completion with a
// gpu-scope release-add.  Combine transport therefore overlaps the tail of
// the down GEMM instead of waiting for the whole layer.
//
// The post-combine barrier is fused exactly like the 19c3cee combine kernel:
// every push CTA joins a completion count (release fence first), the last
// one acquires, publishes the monotonic expected value, and issues the
// multicast arrive that the epilogue's fused wait consumes.  No input
// barrier is needed: this kernel reads only rank-local tensors, and the
// peers' combine-buffer clears were already proven complete by the dispatch
// kernel's input barrier earlier in the same iteration.
#if defined(KITTENS_SM90)

#include "sm90_fp8_block_routed.cuh"
#include "sm90_fp8_block_worker_test.cuh"

namespace mok_sm90::fp8_block_gemm_combine {

using fp8_block_routed::MAX_EP_SIZE;
using fp8_block_test::a_st;
using fp8_block_test::b_st;
using fp8_block_test::d_st;
using fp8_block_test::acc_rt;

constexpr int THREADS = 128;

struct globals {
    // --- GEMM (producer) side, contiguous contract; gl layouts lead the
    // struct for aggregate initialization (no default ctor). ---
    fp8_block_test::contiguous::a_gl A;    // down_input [capacity, I]
    fp8_block_test::contiguous::b_gl B;    // w2 [E, H, I]
    fp8_block_test::contiguous::d_gl D;    // routed_y [capacity, H]
    // --- combine push (consumer) side ---
    uint8_t *combine_peer[MAX_EP_SIZE];    // peers' BF16 combine buffers
    const uint8_t *routed_y;               // same storage as D
    const int *m_indices;                  // produced by the dispatch cut
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    int ep_size;
    int num_local_tokens;
    int hidden_size;
    int topk;
    int schedule_capacity;
    // producer->consumer handoff: per-M64-tile completed-column counters
    unsigned int *down_ready;              // [capacity/64], zeroed each iter
    // fused post-combine barrier state (19c3cee protocol)
    unsigned int *completion_counter;      // [1], self-resetting
    unsigned int *barrier_target;          // [1], monotonic
    unsigned int *barrier_expected_scratch;  // [1], zeroed each iter
    unsigned int *barrier_multicast_ptr;
    const float *A_scale;
    const float *B_scale;
    int n;
    int k_blocks;
    int n_tiles;
};

// Executed by the GEMM CTA that completes its M64 block last (the
// last-arriver): push all 64 rows to their peers' combine buffers, join the
// block-completion count, and let the very last pusher issue the fused
// arrive.  Push is store-direction and fire-and-forget, so riding it on the
// finishing CTA costs no resident communication CTAs at all -- the first
// version's dedicated push clusters occupied SM slots for the whole kernel
// and taxed the GEMM ~11%.
__device__ __forceinline__ void push_block(const globals &g, int m_tile) {
    const int device_rows = g.num_tokens[0];
    const int valid_rows = device_rows < g.schedule_capacity
                               ? device_rows
                               : g.schedule_capacity;
    const int block_base = m_tile * 64;
    const int block_rows =
        valid_rows - block_base < 64 ? valid_rows - block_base : 64;
    const int row_vectors =
        g.hidden_size * 2 / static_cast<int>(sizeof(uint4));  // BF16 rows

    constexpr int WARPS = THREADS / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int r = warp; r < block_rows; r += WARPS) {
        const int row = block_base + r;
        const int peer_rank = g.schedule_peer_rank[row];
        const int peer_token_idx = g.schedule_peer_token_idx[row];
        if (peer_rank < 0 || peer_rank >= g.ep_size || peer_token_idx < 0
            || peer_token_idx >= g.num_local_tokens * g.topk)
            continue;
        const auto *src = reinterpret_cast<const uint4 *>(g.routed_y)
                          + static_cast<size_t>(row) * row_vectors;
        auto *dst = reinterpret_cast<uint4 *>(g.combine_peer[peer_rank])
                    + static_cast<size_t>(peer_token_idx) * row_vectors;
        #pragma unroll 4
        for (int i = lane; i < row_vectors; i += 32)
            dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("{fence.release.sys;}" ::: "memory");
        const unsigned int active_blocks = static_cast<unsigned int>(
            (valid_rows + 63) / 64);
        const unsigned int finished =
            atomicAdd(g.completion_counter, 1u) + 1u;
        if (finished == active_blocks) {
            asm volatile("{fence.acquire.sys;}" ::: "memory");
            *g.completion_counter = 0u;  // reset for the next graph replay
            const unsigned int expected =
                atomicAdd(g.barrier_target,
                          static_cast<unsigned int>(g.ep_size))
                + static_cast<unsigned int>(g.ep_size);
            asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                         "l"(g.barrier_expected_scratch), "r"(expected)
                         : "memory");
            asm volatile(
                "{multimem.red.release.sys.global.add.u32 [%0], 1;}" ::
                "l"(g.barrier_multicast_ptr) : "memory");
            asm volatile("{fence.proxy.alias;}" ::: "memory");
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
    // Join this M64 block's completion count.  The syncthreads gathers every
    // thread's D stores to thread 0, whose gpu-scope release-add makes them
    // visible through the counter's atomic chain; the CTA that observes the
    // final count acquires that chain (so all n_tiles worth of D rows are
    // readable) and pushes the whole block to the peers.
    __syncthreads();
    __shared__ unsigned int block_done;
    if (threadIdx.x == 0) {
        unsigned int prior;
        asm volatile("{atom.add.release.gpu.global.u32 %0, [%1], 1;}"
                     : "=r"(prior)
                     : "l"(g.down_ready + m_tile) : "memory");
        block_done = prior + 1u;
    }
    __syncthreads();
    if (block_done == static_cast<unsigned int>(g.n_tiles)) {
        if (threadIdx.x == 0)
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        __syncthreads();
        push_block(g, m_tile);
    }
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void kernel(const __grid_constant__ globals g) {
    gemm_role(g, clusterIdx().x);
}

inline void entry_out(
    const at::Tensor &down_input, const at::Tensor &down_input_scale,
    const at::Tensor &weight, const at::Tensor &weight_scale,
    const at::Tensor &m_indices, const at::Tensor &num_tokens,
    const at::Tensor &routed_y, const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs, int64_t topk,
    const at::Tensor &down_ready, const at::Tensor &combine_completion,
    const at::Tensor &barrier_target,
    const at::Tensor &barrier_expected_scratch,
    int64_t barrier_buffer_multicast_ptr) {
    const int64_t schedule_capacity = down_input.size(0);
    TORCH_CHECK(down_input.dim() == 2 && down_input.is_cuda()
                    && down_input.scalar_type() == at::kFloat8_e4m3fn
                    && down_input.is_contiguous()
                    && schedule_capacity % 64 == 0,
                "down_input must be contiguous CUDA FP8 [capacity(M64), I]");
    const int64_t inter_size = down_input.size(1);
    TORCH_CHECK(inter_size >= 128 && inter_size % 128 == 0,
                "down_input K must be a positive K128 multiple");
    const int k_blocks = static_cast<int>(inter_size / 128);
    TORCH_CHECK(down_input_scale.dim() == 2
                    && down_input_scale.size(0) == schedule_capacity
                    && down_input_scale.size(1) == k_blocks
                    && down_input_scale.scalar_type() == at::kFloat
                    && down_input_scale.is_contiguous(),
                "down_input_scale must be contiguous float32 [capacity,I/128]");
    TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == at::kFloat8_e4m3fn
                    && weight.is_contiguous() && weight.size(2) == inter_size,
                "weight must be contiguous FP8 [E,H,I]");
    const int experts = static_cast<int>(weight.size(0));
    const int n = static_cast<int>(weight.size(1));
    TORCH_CHECK(n >= 128 && n % 128 == 0, "H must be a positive K128 multiple");
    TORCH_CHECK(weight_scale.dim() == 3 && weight_scale.size(0) == experts
                    && weight_scale.size(1) == n / 128
                    && weight_scale.size(2) == k_blocks
                    && weight_scale.scalar_type() == at::kFloat
                    && weight_scale.is_contiguous(),
                "weight_scale must be contiguous float32 [E,H/128,I/128]");
    TORCH_CHECK(m_indices.numel() == schedule_capacity
                    && m_indices.scalar_type() == at::kInt,
                "m_indices must be int32 [capacity]");
    TORCH_CHECK(routed_y.dim() == 2 && routed_y.size(0) == schedule_capacity
                    && routed_y.size(1) == n
                    && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous() && routed_y.is_cuda(),
                "routed_y must be contiguous BF16 [capacity,H]");
    fp8_block_routed::check_combine_schedule(
        schedule_peer_rank, schedule_peer_token_idx, num_tokens,
        schedule_capacity);
    fp8_block_routed::check_pointer_list(
        combine_buffer_ptrs, "combine_buffer_ptrs");
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    TORCH_CHECK(combine_buffer.dim() == 2 && combine_buffer.is_cuda()
                    && combine_buffer.scalar_type() == at::kBFloat16
                    && combine_buffer.is_contiguous()
                    && combine_buffer.size(1) == n
                    && combine_buffer.size(0) % topk == 0,
                "combine_buffer must be contiguous BF16 [T*topk,H]");
    const int num_local_tokens =
        static_cast<int>(combine_buffer.size(0) / topk);
    TORCH_CHECK(down_ready.is_cuda() && down_ready.scalar_type() == at::kInt
                    && down_ready.is_contiguous()
                    && down_ready.numel() == schedule_capacity / 64,
                "down_ready must be int32 [capacity/64]");
    for (const at::Tensor *t :
         {&combine_completion, &barrier_target, &barrier_expected_scratch}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt
                        && t->is_contiguous() && t->numel() == 1,
                    "barrier state tensors must be int32 [1]");
    }
    TORCH_CHECK(barrier_buffer_multicast_ptr > 0,
                "barrier multicast pointer must be positive");
    kittens::py::device_check(down_input, down_input_scale, m_indices,
                              weight, weight_scale);
    kittens::py::device_check(down_input, routed_y);

    c10::cuda::CUDAGuard device_guard(down_input.device());
    globals g{
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::a_gl>(
            const_cast<at::Tensor &>(down_input)),
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::b_gl>(
            const_cast<at::Tensor &>(weight)),
        kittens::py::tensor_to_gl<fp8_block_test::contiguous::d_gl>(
            const_cast<at::Tensor &>(routed_y)),
    };
    for (size_t rank = 0; rank < combine_buffer_ptrs.size(); ++rank)
        g.combine_peer[rank] =
            reinterpret_cast<uint8_t *>(combine_buffer_ptrs[rank]);
    g.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    g.m_indices = m_indices.data_ptr<int>();
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.ep_size = static_cast<int>(combine_buffer_ptrs.size());
    g.num_local_tokens = num_local_tokens;
    g.hidden_size = n;
    g.topk = static_cast<int>(topk);
    g.schedule_capacity = static_cast<int>(schedule_capacity);
    g.down_ready =
        reinterpret_cast<unsigned int *>(down_ready.data_ptr<int>());
    g.completion_counter = reinterpret_cast<unsigned int *>(
        combine_completion.data_ptr<int>());
    g.barrier_target =
        reinterpret_cast<unsigned int *>(barrier_target.data_ptr<int>());
    g.barrier_expected_scratch = reinterpret_cast<unsigned int *>(
        barrier_expected_scratch.data_ptr<int>());
    g.barrier_multicast_ptr =
        reinterpret_cast<unsigned int *>(barrier_buffer_multicast_ptr);
    g.A_scale = down_input_scale.data_ptr<float>();
    g.B_scale = weight_scale.data_ptr<float>();
    g.n = n;
    g.k_blocks = k_blocks;
    g.n_tiles = n / 64;

    const int m_tiles = static_cast<int>(schedule_capacity / 64);
    const int total_ctas = m_tiles * g.n_tiles;
    constexpr int PIPE_DEPTH = 2;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(down_input.get_device());
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    kernel<<<total_ctas, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::fp8_block_gemm_combine

#endif
