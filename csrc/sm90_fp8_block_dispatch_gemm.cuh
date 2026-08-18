#pragma once

// MoK-form producer/consumer fusion, first cut: pull dispatch and the gate/up
// contiguous grouped GEMM run inside ONE kernel, fed by a device-side ticket
// queue over a fixed resident-worker-cluster grid.
//
// Scheduling safety (structural, does not depend on block scheduling order):
// every worker cluster draws strictly increasing tickets from a global
// counter, and the first `copy_clusters` tickets are copy tasks by
// construction.  Whatever subset of workers the scheduler makes resident
// first, the earliest-drawn tickets are copy work, so any GEMM ticket can
// only be drawn after every copy ticket has been claimed by an already
// resident, progressing cluster.  A GEMM tile spin therefore never waits on
// a producer that is not resident.  Cross-rank liveness (the input-publish
// barrier at the head of every copy task) is an SPMD precondition: rank
// launch failure is terminated by the timeout traps below, not reasoned
// away.  See k1-resident-worker-redesign-20260817.md for the full proof.
//
// Ticket protocol per cluster (lockstep): CTA rank 0 thread 0 draws the
// ticket and publishes it to a per-cluster global slot; a cluster barrier
// (release/acquire) makes it visible; both CTAs then enter the same task
// kind, finish, and hit the boundary barrier before the next draw.  GEMM
// pipeline phase bits persist across tasks (megakernel pattern) so the
// mbarrier semaphores are initialized exactly once.
#if defined(KITTENS_SM90)

#include <array>

#include "sm90_fp8_block_gemm_core.cuh"
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

// Trap protocol (host-mapped pinned record, 8 x u64):
//   [0] owner/error_code (CAS from 0)  [1] site_id  [2] slot or m_tile
//   [3] expected  [4] observed  [5] ep_rank  [6] ticket  [7] iter_count
// Only the CAS winner writes [1..7], fences to the host, and traps; losers
// park forever without touching the workspace and die with the kernel.
constexpr unsigned long long ERR_TIMEOUT = 1;
constexpr unsigned long long ERR_CONTRACT = 2;
constexpr unsigned long long SITE_K1_INPUT_SCRATCH = 1;
constexpr unsigned long long SITE_K1_BARRIER_FLAG = 2;
constexpr unsigned long long SITE_K1_TILE_READY = 3;
constexpr unsigned long long SITE_K1_CONTRACT = 6;
constexpr unsigned long long SPIN_TRAP_ITERS = 1ull << 28;

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
    int copy_clusters;  // C_cluster: number of copy tickets
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
    // --- resident-worker ticket queue ---
    unsigned int *ticket_counter;   // [1], zeroed each iteration
    unsigned int *worker_ticket;    // [>= worker clusters], publish slots
    unsigned long long *trap_record;  // host-mapped pinned, never zeroed here
    int total_tickets;              // copy_clusters + m_tiles * (n_tiles/2)
    int ep_rank;                    // for trap records only
    // Stage-C debug knobs (0 = off / default in production):
    unsigned long long delay_ticket0_cycles;  // busy-wait after ticket-0 draw
    unsigned long long spin_trap_iters;       // spin timeout override
    unsigned int *ticket_visit;    // [>= total_tickets] when record_visits
    int record_visits;             // exactly-once debug counting
};

__device__ __forceinline__ void park_forever() {
    while (true) __nanosleep(1u << 20);
}

// Two-phase publication so a CPU watchdog can never observe a half-written
// record: the winner claims slot[0] with a sentinel, writes the payload,
// fences to the system scope, and only then release-stores the final error
// code.  Host readers treat the sentinel as "not committed yet".  Losers
// (sentinel or final code) park so the winner cannot be cut short.
constexpr unsigned long long TRAP_CLAIMED = ~0ull;

__device__ __noinline__ void trap_commit(
    const globals &g, unsigned long long code, unsigned long long site,
    unsigned long long slot, unsigned long long expected,
    unsigned long long observed, unsigned long long ticket,
    unsigned long long iters) {
    const unsigned long long prev =
        atomicCAS(g.trap_record, 0ull, TRAP_CLAIMED);
    if (prev != 0ull) park_forever();
    g.trap_record[1] = site;
    g.trap_record[2] = slot;
    g.trap_record[3] = expected;
    g.trap_record[4] = observed;
    g.trap_record[5] = static_cast<unsigned long long>(g.ep_rank);
    g.trap_record[6] = ticket;
    g.trap_record[7] = iters;
    __threadfence_system();
    asm volatile("{st.release.sys.global.u64 [%0], %1;}" ::
                 "l"(g.trap_record), "l"(code) : "memory");
    __trap();
}

__device__ __forceinline__ void copy_task(const globals &g, unsigned int ticket,
                                          int cta_rank) {
    const int copy_cta_count = g.copy_clusters * 2;
    const int copy_cta_idx = static_cast<int>(ticket) * 2 + cta_rank;

    // Absorbed input barrier: ticket 0's rank-0 CTA arrives for this rank
    // BEFORE entering any wait (its x_buffer copy precedes this kernel on
    // the stream) and publishes the expected flag value; every copy task
    // then waits for all ranks.  The arrive-before-wait order is what the
    // cross-rank liveness precondition in the header rests on.
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
        unsigned long long iters = 0;
        do {
            asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                         : "=r"(expected)
                         : "l"(g.input_expected_scratch) : "memory");
            if (expected != 0u) break;
            __nanosleep(128);
            if (++iters >= g.spin_trap_iters)
                trap_commit(g, ERR_TIMEOUT, SITE_K1_INPUT_SCRATCH,
                            copy_cta_idx, 1, 0, ticket, iters);
        } while (true);
        unsigned int value;
        iters = 0;
        do {
            asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}"
                         : "=r"(value) : "l"(g.barrier_flag) : "memory");
            if (value >= expected) break;
            __nanosleep(128);
            if (++iters >= g.spin_trap_iters)
                trap_commit(g, ERR_TIMEOUT, SITE_K1_BARRIER_FLAG,
                            copy_cta_idx, expected, value, ticket, iters);
        } while (true);
        asm volatile("{fence.acquire.sys;}" ::: "memory");
    }
    __syncthreads();

    // Cache the per-expert segment ends once per task: the per-row expert
    // lookup below then scans shared memory instead of issuing up to
    // num_local_experts uncached global loads for every row.  The leading
    // __syncthreads above also orders any previous task's readers before
    // this rebuild.
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
            // Stage through registers: all of a chunk's remote loads issue
            // before any store, so their latencies overlap.  The interleaved
            // load/store form serialized on the possible dst/src alias.
            constexpr int VEC_CHUNK = 8;  // 32 lanes x 8 x 16B = 4KB per pass
            uint4 buffer[VEC_CHUNK];
            for (int base = 0; base < fp8_vectors; base += 32 * VEC_CHUNK) {
                #pragma unroll
                for (int j = 0; j < VEC_CHUNK; ++j) {
                    const int i = base + lane + j * 32;
                    if (i < fp8_vectors) buffer[j] = src_vectors[i];
                }
                #pragma unroll
                for (int j = 0; j < VEC_CHUNK; ++j) {
                    const int i = base + lane + j * 32;
                    if (i < fp8_vectors) dst_vectors[i] = buffer[j];
                }
            }
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

__device__ __forceinline__ void gemm_task(
    const globals &g, int gemm_cluster_idx, int cta_rank,
    unsigned int ticket, uint32_t &phasebits, uint32_t &ready_phase,
    a_st (&a_smem)[2], b_st (&b_smem)[2], d_st &d_smem,
    semaphore (&inputs_arrived)[2], semaphore (&inputs_finished)[2],
    semaphore (&inputs_ready)[2]) {
    const auto coord = fp8_block_gemm_core::decode_tile(
        g, gemm_cluster_idx, cta_rank);
    if (coord.global_row_base >= g.num_tokens[0])
        return;

    // Consume as soon as this tile's own rows have been dispatched.  The
    // producer publishes m_indices before its release-increment, so the
    // expert lookup below is ordered by the acquire spin.
    if (threadIdx.x == 0) {
        unsigned int done;
        unsigned long long iters = 0;
        do {
            asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                         : "=r"(done)
                         : "l"(g.tile_ready + coord.m_tile) : "memory");
            if (done >= 64u) break;
            __nanosleep(256);
            if (++iters >= g.spin_trap_iters)
                trap_commit(g, ERR_TIMEOUT, SITE_K1_TILE_READY,
                            coord.m_tile, 64, done, ticket, iters);
        } while (true);
    }
    __syncthreads();

    fp8_block_gemm_core::run_tile(
        g, coord, cta_rank, phasebits, ready_phase,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);
    // Task-boundary drain: the direct global store above is synchronous per
    // thread; this sync keeps slow storers ahead of the next task's d_smem
    // writes.  The boundary cluster barrier in the worker loop covers the
    // cross-CTA side.
    warpgroup::sync(0);
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void kernel(const __grid_constant__ globals g) {
    const int cta_rank = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    // Device-side contract check before the first task: num_tokens is a
    // device-resident dynamic scalar (graph replay varies it), so the host
    // binding cannot validate it without breaking capture.  Fail closed.
    const int nt = g.num_tokens[0];
    if (nt < 0 || nt > g.schedule_capacity || (nt & 63) != 0) {
        if (threadIdx.x == 0)
            trap_commit(g, ERR_CONTRACT, SITE_K1_CONTRACT, 0,
                        g.schedule_capacity, nt, 0, 0);
        park_forever();
    }

    // GEMM pipeline state lives at kernel scope and persists across tasks:
    // semaphores are initialized exactly once, phase bits carry forward
    // (megakernel pattern), so per-task reinitialization hazards never
    // arise.
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
    uint32_t phasebits = 0xFFFF0000;
    uint32_t ready_phase = 0;
    everyone::tma::cluster::sync();

    // Worker loop: rank-0 thread 0 draws a ticket and publishes it to this
    // cluster's global slot; the cluster barrier (release/acquire at cluster
    // scope) makes it visible to the peer CTA; both CTAs execute the same
    // task and meet at the boundary barrier before the next draw.
    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int drawn = atomicAdd(g.ticket_counter, 1u);
            // Explicit release store with a memory clobber: the cluster
            // barrier's hardware semantics would suffice, but the inline-PTX
            // barrier wrapper carries no clobber, so the compiler must be
            // told not to sink this store past it.
            asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                         "l"(g.worker_ticket + cluster_id), "r"(drawn)
                         : "memory");
        }
        everyone::tma::cluster::sync();
        unsigned int ticket;
        asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                     : "=r"(ticket)
                     : "l"(g.worker_ticket + cluster_id) : "memory");
        if (ticket >= static_cast<unsigned int>(g.total_tickets))
            break;
        if (g.record_visits && cta_rank == 0 && threadIdx.x == 0)
            atomicAdd(g.ticket_visit + ticket, 1u);
        // Stage-C injection: hold the resident ticket-0 producer for a
        // bounded busy-wait AFTER claiming its ticket and BEFORE the
        // arrive/copy, exercising the resident-but-slow-producer path.
        if (ticket == 0u && g.delay_ticket0_cycles != 0ull) {
            if (threadIdx.x == 0) {
                const unsigned long long start = clock64();
                while (static_cast<unsigned long long>(clock64()) - start
                       < g.delay_ticket0_cycles) { }
            }
            __syncthreads();
        }
        if (ticket < static_cast<unsigned int>(g.copy_clusters))
            copy_task(g, ticket, cta_rank);
        else
            gemm_task(g, static_cast<int>(ticket) - g.copy_clusters,
                      cta_rank, ticket, phasebits, ready_phase,
                      a_smem, b_smem, d_smem,
                      inputs_arrived, inputs_finished, inputs_ready);
        everyone::tma::cluster::sync();
    }
}


// Per-device cluster-occupancy cache.  prewarm() is called at workspace
// creation (never inside a capture); entry_out fails closed if the cache is
// cold under an active capture.
inline int &occupancy_slot(int device_index) {
    static std::array<int, 64> cache = [] {
        std::array<int, 64> c{};
        c.fill(-1);
        return c;
    }();
    return cache.at(static_cast<size_t>(device_index));
}

inline int occupancy_cache(int device_index) {
    return occupancy_slot(device_index);
}

inline int prewarm(int device_index) {
    constexpr int PIPE_DEPTH = 2;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;
    c10::cuda::CUDAGuard guard(device_index);
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(THREADS, 1, 1);
    cfg.dynamicSmemBytes = SMEM;
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int max_clusters = 0;
    CUDACHECK(cudaOccupancyMaxActiveClusters(&max_clusters, kernel, &cfg));
    TORCH_CHECK(max_clusters >= 1,
                "K1 worker kernel has zero cluster occupancy");
    occupancy_slot(device_index) = max_clusters;
    return max_clusters;
}

inline int64_t entry_prewarm(int64_t device_index) {
    return static_cast<int64_t>(prewarm(static_cast<int>(device_index)));
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
    int64_t copy_clusters, int64_t ep_rank,
    const at::Tensor &ticket_counter, const at::Tensor &worker_ticket,
    int64_t trap_record_ptr, int64_t forced_worker_clusters,
    int64_t delay_ticket0_cycles, int64_t spin_trap_iters,
    const at::Tensor &ticket_visit, int64_t record_visits) {
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
         {&barrier_buffer, &barrier_target, &input_expected_scratch,
          &ticket_counter}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt
                        && t->is_contiguous() && t->numel() == 1,
                    "barrier/queue state tensors must be int32 [1]");
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
    TORCH_CHECK(ep_rank >= 0 && ep_rank < (int64_t)x_ptrs.size(),
                "ep_rank must index the peer list");
    TORCH_CHECK(trap_record_ptr > 0,
                "trap_record_ptr must be a mapped pinned address");
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
    const int n_pairs = g.n_tiles / 2;
    const int total_tickets =
        static_cast<int>(copy_clusters) + m_tiles * n_pairs;
    g.ticket_counter =
        reinterpret_cast<unsigned int *>(ticket_counter.data_ptr<int>());
    g.worker_ticket =
        reinterpret_cast<unsigned int *>(worker_ticket.data_ptr<int>());
    {
        // Host-mapped pinned record: resolve the actual device-usable
        // address (UVA usually aliases the host pointer, but a
        // host-register pinned backend need not).
        void *dev = nullptr;
        CUDACHECK(cudaHostGetDevicePointer(
            &dev, reinterpret_cast<void *>(trap_record_ptr), 0));
        TORCH_CHECK(dev != nullptr, "trap record host memory is not mapped");
        g.trap_record = reinterpret_cast<unsigned long long *>(dev);
    }
    g.total_tickets = total_tickets;
    g.ep_rank = static_cast<int>(ep_rank);
    g.delay_ticket0_cycles =
        static_cast<unsigned long long>(delay_ticket0_cycles);
    g.spin_trap_iters = spin_trap_iters > 0
                            ? static_cast<unsigned long long>(spin_trap_iters)
                            : SPIN_TRAP_ITERS;
    TORCH_CHECK(record_visits == 0
                    || (ticket_visit.is_cuda()
                        && ticket_visit.scalar_type() == at::kInt
                        && ticket_visit.numel() >= total_tickets),
                "ticket_visit must be int32 with one slot per ticket");
    g.ticket_visit =
        reinterpret_cast<unsigned int *>(ticket_visit.data_ptr<int>());
    g.record_visits = static_cast<int>(record_visits);

    constexpr int PIPE_DEPTH = 2;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;

    // Per-device cluster occupancy cache, warmed by prewarm() at workspace
    // creation (before any CUDA graph capture).  Inside a capture the query
    // itself is illegal, so an un-warmed cache fails closed.
    const int device_index = routed_x.get_device();
    int max_clusters = occupancy_cache(device_index);
    if (max_clusters < 0) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDACHECK(cudaStreamIsCapturing(
            at::cuda::getCurrentCUDAStream(device_index), &cap));
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "K1 occupancy cache not warmed before graph capture; "
                    "call fp8_block_dispatch_gemm_prewarm at workspace "
                    "creation");
        max_clusters = prewarm(device_index);
    }
    int workers = forced_worker_clusters > 0
                      ? static_cast<int>(forced_worker_clusters)
                      : std::min(max_clusters, total_tickets);
    if (workers < 1) workers = 1;
    TORCH_CHECK(worker_ticket.is_cuda()
                    && worker_ticket.scalar_type() == at::kInt
                    && worker_ticket.is_contiguous()
                    && worker_ticket.numel() >= workers,
                "worker_ticket must be int32 with one slot per worker");

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(routed_x.get_device());
    kernel<<<workers * 2, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::fp8_block_dispatch_gemm

#endif
