#pragma once

// The warprole megakernel: one launch per layer, one CTA per SM, three warp
// roles.  Phases (per rank):
//   0  rank barrier                       comm warpgroup
//   1  dispatch minibatch q -> x_ready[q] comm warpgroup, 8-row tickets
//   2  W13 tasks -> hidden, hidden_ready  producer + consumers (fused SwiGLU/quant)
//   3  W2 tasks  -> routed_y, y_ready     producer + consumers (TMA store)
//   4  combine tile m -> peers, push_done comm warpgroup
//   5  weighted reduce of local tokens    consumers, after push_done == ep_size
// The task stream order is static (config.cuh decode_task) but the assignment
// is dynamic: each CTA's producer claims the next index from a global cursor
// (input_expected_scratch[1]) and hands it to its consumers through a 2-entry
// shared-memory mailbox, so a CTA that finished early takes the next task instead
// of idling on a fixed stride.  The last CTA to finish rewinds the cursor.
// Dependencies are the three counters plus the ring barriers inside the CTA.
// Every wait is bounded by spin_limit and traps through the shared trap record.
#if defined(KITTENS_SM90)
#include <cstdlib>
#include <ATen/ATen.h>

#include <vector>

#include "sm90_fp8_block_terminal_route_flags.cuh"
#include "sm90_fp8_block_warprole_comm.cuh"
#include "sm90_fp8_block_warprole_epilogue.cuh"
#include "utils.cuh"

namespace mok_sm90::warprole::fused {
using namespace kittens;
namespace route_flags = mok_sm90::fp8_block_terminal_route_flags;
namespace tc = mok_sm90::fp8_block_terminal_comm;

constexpr unsigned long long ERR_TIMEOUT = 1ull;
constexpr unsigned long long ERR_CONTRACT = 2ull;
constexpr unsigned long long SITE_WARPROLE_BARRIER = 40ull;
constexpr unsigned long long SITE_WARPROLE_X_READY = 41ull;
constexpr unsigned long long SITE_WARPROLE_HIDDEN_READY = 42ull;
constexpr unsigned long long SITE_WARPROLE_Y_READY = 43ull;
constexpr unsigned long long SITE_WARPROLE_PUSH_DONE = 44ull;
constexpr unsigned long long SITE_WARPROLE_CONTRACT = 45ull;
constexpr unsigned long long SITE_WARPROLE_TASK_BOX = 46ull;

// Named barriers of the W13 hand-off between the consumers and the epilogue warps
// (warps 1-3 of the producer warpgroup): ids 1-2 are the consumer warpgroups,
// 3 the standalone consumers_sync, 4 the phase-5 consumer barrier.
constexpr int BAR_D_FULL = 5;    // consumers arrive after parking gate/up in smem.d, epilogue warps sync
constexpr int BAR_D_EMPTY = 6;   // epilogue warps arrive when done reading smem.d, consumers sync
constexpr int BAR_EPI = 7;       // the 96 epilogue threads alone
constexpr int EPI_WARPS = 3;
constexpr int EPI_THREADS = 32 * EPI_WARPS;
template <int NC> constexpr int handoff_threads() { return 128 * NC + EPI_THREADS; }

constexpr int W2_N_TILES_128 = HIDDEN / N_TILE;     // 32
constexpr int W13_N_TILES_128 = 2 * INTER / N_TILE; // 32 (interleaved gate/up)

struct globals {
    comm::globals c{};
    gemm::a_gl routed_x_gl;    // [capacity, 4096] fp8
    gemm::a_gl hidden_gl;      // [capacity, 2048] fp8
    gemm::b_gl w13i;           // [E, 4096, 4096] fp8, interleaved gate/up
    gemm::b_gl w2;             // [E, 4096, 2048] fp8
    gemm::d_gl routed_y_gl;    // [capacity, 4096] bf16
    const float *w13i_scale = nullptr;   // [E, 32, 32]
    const float *w2_scale = nullptr;     // [E, 32, 16]
    uint8_t *hidden = nullptr;           // [capacity, 2048] fp8
    float *hidden_scale = nullptr;       // [capacity, 16]
    unsigned int *hidden_ready = nullptr;   // [m_tiles]
    float limit = 0.0f;
    const __nv_bfloat16 *combine_local = nullptr;   // [num_local_tokens * topk, 4096]
    const float *weights = nullptr;                 // [num_local_tokens * topk]
    const int *topk_ids = nullptr;                  // [num_local_tokens * topk]
    __nv_bfloat16 *output = nullptr;                // [num_local_tokens, 4096]
    unsigned int *barrier_flag = nullptr;
    unsigned int *barrier_target = nullptr;
    unsigned int *barrier_multicast_ptr = nullptr;
    unsigned int *input_expected_scratch = nullptr;
    unsigned int *task_cursor = nullptr;   // input_expected_scratch + 1: next unclaimed task index
    unsigned int *cta_done = nullptr;      // input_expected_scratch + 2: CTAs finished; the last rewinds both
    unsigned long long *trap_record = nullptr;
    unsigned long long spin_limit = 0;
    // Benchmark-only knobs (MOK_WARPROLE_NO_DEPS / MOK_WARPROLE_COMM_OFF /
    // MOK_WARPROLE_REDUCE_OFF): skip every dependency wait, make the comm warpgroup
    // idle, skip the phase-5 weighted reduce.  Output is garbage when any is set.
    int skip_waits = 0;
    int comm_off = 0;
    int reduce_off = 0;
    // Benchmark-only timeline probe (MOK_WARPROLE_PROBE): per CTA, PROBE_SLOTS
    // globaltimer stamps at the phase boundaries; nullptr in production.
    unsigned long long *probe = nullptr;

    // kittens::gl has no default constructor: the five TMA-described tensors come first.
    __host__ globals(const gemm::a_gl &rx, const gemm::a_gl &h, const gemm::b_gl &w13, const gemm::b_gl &w2_,
                     const gemm::d_gl &ry)
        : routed_x_gl(rx), hidden_gl(h), w13i(w13), w2(w2_), routed_y_gl(ry) {}
};

__device__ __forceinline__ void trap(const globals &g, unsigned long long code, unsigned long long site,
                                     unsigned long long slot, unsigned long long expected,
                                     unsigned long long observed, unsigned long long iters) {
    utils::mok_trap_commit(g.trap_record, code, site, slot, expected, observed,
                           static_cast<unsigned long long>(g.c.ep_rank), 0ull, iters);
}

constexpr int PROBE_SLOTS = 8;
// 0 entry  1 rank barrier done  2 all minibatches dispatched  3 first W13 task starts
// 4 last GEMM task done  5 combine_finish done  6 push_done observed  7 reduce done
__device__ __forceinline__ void stamp(const globals &g, int slot) {
    if (g.probe != nullptr) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        g.probe[blockIdx.x * PROBE_SLOTS + slot] = t;
    }
}

template <bool SYS>
__device__ __forceinline__ void wait_geq_or_trap(const globals &g, const unsigned int *ctr, unsigned int target,
                                                 unsigned long long site, unsigned long long slot) {
    if (g.skip_waits) return;   // MOK_WARPROLE_NO_DEPS: benchmark-only, no ordering guarantee
    unsigned long long iters = 0;
    while (true) {
        const unsigned int v = SYS ? comm::load_acquire_sys(ctr) : comm::load_acquire_gpu(ctr);
        if (v >= target) return;
        __nanosleep(256);
        if (++iters >= g.spin_limit) trap(g, ERR_TIMEOUT, site, slot, target, v, iters);
    }
}

// Phase 0.  Executed by the whole comm warpgroup of every CTA; CTA 0 arrives on
// behalf of this rank.  Same protocol as terminal_full::production_input_barrier.
// Phase-5 reduce, one warp per token.  Per element this is exactly
// route_flags::weighted_reduce_masked_value (invalid routes skipped, the first
// valid route a rounded multiply, later ones rounded FMAs, one BF16 rounding),
// so the output stays bitwise equal to the split path.  It replaces a
// CTA-per-token loop over single BF16 loads that re-read topk_ids and weights
// for every column and kept about one load latency in flight per thread:
// 0.8 ms for 2048 tokens (~100 GB/s) on H20, measured with the step-3 knobs.
// The pointers are restrict-qualified on purpose: without it the compiler has to
// keep every 16-byte load behind the previous iteration's store to `output`, so
// each 512-byte step of a row paid one full memory latency (149 / 248 us for
// 2048 / 3888 tokens on the probe, ~0.8 TB/s).
// Weighted reduce of one 8-column chunk: raw[r] holds 8 bf16 of route r, routes with
// ids < 0 are padding.  Same arithmetic order as split's combine_reduce: the first valid
// route by __fmul_rn, the rest by __fmaf_rn in route order, one bf16 rounding at the end.
__device__ __forceinline__ uint4 reduce_chunk(const uint4 (&raw)[TOPK], const int (&ids)[TOPK],
                                              const float (&w)[TOPK]) {
    __align__(16) __nv_bfloat16 out8[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        float acc = 0.0f;
        bool initialized = false;
#pragma unroll
        for (int r = 0; r < TOPK; ++r) {
            if (ids[r] < 0) continue;
            const float v = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&raw[r])[c]);
            if (!initialized) {
                acc = __fmul_rn(v, w[r]);
                initialized = true;
            } else {
                acc = __fmaf_rn(v, w[r], acc);
            }
        }
        out8[c] = initialized ? __float2bfloat16_rn(acc) : __float2bfloat16_rn(0.0f);
    }
    return *reinterpret_cast<const uint4 *>(out8);
}

// One-dimensional TMA bulk copy global -> this CTA's shared memory; completion is counted
// as transaction bytes on `bar` (16-byte aligned addresses, size a multiple of 16).
__device__ __forceinline__ void bulk_load(void *smem_dst, const void *gmem_src, uint32_t bytes, semaphore &bar) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(gmem_src), "r"(bytes), "r"(mbar) : "memory");
}

// Phase-5 reduce staging: the TOPK combine rows of one token are 48 KB; as many tokens as
// the (idle by then) GEMM ring holds are kept in flight per CTA, at most four.
constexpr int REDUCE_ROW_BYTES = HIDDEN * 2;
constexpr int REDUCE_TOKEN_BYTES = TOPK * REDUCE_ROW_BYTES;
template <int NC, int STAGES> constexpr int reduce_stages() {
    constexpr int fit = static_cast<int>(sizeof(gemm::smem_layout<NC, STAGES>) / REDUCE_TOKEN_BYTES);
    static_assert(fit >= 2, "the GEMM ring must hold at least two tokens of combine rows");
    return fit > 4 ? 4 : fit;
}

__device__ __forceinline__ void rank_barrier(const globals &g) {
    if (warpgroup::laneid() == 0) {
        if (blockIdx.x == 0) {
            const unsigned int expected =
                atomicAdd(g.barrier_target, static_cast<unsigned int>(g.c.ep_size))
                + static_cast<unsigned int>(g.c.ep_size);
            asm volatile("{st.release.gpu.global.u32 [%0], %1;}" :: "l"(g.input_expected_scratch), "r"(expected) : "memory");
            asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" :: "l"(g.barrier_multicast_ptr) : "memory");
            asm volatile("{fence.proxy.alias;}" ::: "memory");
        }
        unsigned int expected = 0;
        unsigned long long iters = 0;
        while ((expected = comm::load_acquire_gpu(g.input_expected_scratch)) == 0u) {
            __nanosleep(128);
            if (++iters >= g.spin_limit) trap(g, ERR_TIMEOUT, SITE_WARPROLE_BARRIER, 1, 1, 0, iters);
        }
        unsigned int observed = 0;
        iters = 0;
        while (true) {
            asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}" : "=r"(observed) : "l"(g.barrier_flag) : "memory");
            if (observed >= expected) break;
            __nanosleep(128);
            if (++iters >= g.spin_limit) trap(g, ERR_TIMEOUT, SITE_WARPROLE_BARRIER, 2, expected, observed, iters);
        }
        asm volatile("{fence.acquire.sys;}" ::: "memory");
    }
    warpgroup::sync(7);
}

template <int NC, int STAGES>
__global__ __launch_bounds__(gemm::num_threads<NC>(), 1)
void kernel(const __grid_constant__ globals g) {
    using smem_t = gemm::smem_layout<NC, STAGES>;
    extern __shared__ int __shm[];
    auto &smem = *reinterpret_cast<smem_t *>(
        ((reinterpret_cast<uint64_t>(&__shm[0])) + 1023) & ~static_cast<uint64_t>(1023));
    __shared__ semaphore full[STAGES];
    __shared__ semaphore empty[STAGES];
    __shared__ semaphore reduce_full[reduce_stages<NC, STAGES>()];
    __shared__ int expert_row_end[256];
    __shared__ int task_box[2];          // claimed task index by parity, -1 = no more tasks
    __shared__ unsigned int task_seq;    // entries the producer has published so far
    __shared__ int epi_task[2];          // (m_tile, i_tile) of the gate/up pair parked in smem.d; m_tile -1 = end
    if (threadIdx.x == 0) {
        task_seq = 0u;
        // arrival count 1: the expect_tx arrive of issue(); the copies only move the tx-count
        for (int r = 0; r < reduce_stages<NC, STAGES>(); ++r) init_semaphore(reduce_full[r], 1, 0);
        tc::build_expert_row_ends(g.c, expert_row_end);
    }
    gemm::standalone::init_ring<NC, STAGES>(full, empty);   // ends with __syncthreads

    const shape s = comm::active_shape(g.c);
    if (blockIdx.x == 0 && threadIdx.x == 0 && (s.num_rows % M_TILE != 0 || s.num_rows < 0))
        trap(g, ERR_CONTRACT, SITE_WARPROLE_CONTRACT, 0, 0, static_cast<unsigned long long>(s.num_rows), 0);
    const int role = warpgroup::groupid();   // 0..NC-1 consumers, NC producer, NC+1 comm
    const int m_tiles = s.num_rows / M_TILE;
    if (threadIdx.x == 0) stamp(g, 0);

    if (role == NC + 1) {
        // ------------------------------------------------ comm warpgroup
        warpgroup::decrease_registers<gemm::comm_regs<1>()>();
        if (!g.comm_off) {   // MOK_WARPROLE_COMM_OFF: benchmark-only, the warpgroup idles
            rank_barrier(g);
            if (warpgroup::laneid() == 0) stamp(g, 1);
            for (int q = 0; q < minibatches(s); ++q) comm::dispatch_minibatch(g.c, s, q, expert_row_end);
            if (warpgroup::laneid() == 0) stamp(g, 2);
            // Combine is striped by row over every comm warp of the grid rather than
            // one tile per CTA: the tiles of the last minibatch all become ready at
            // about the same time, and pushing a 512 KB tile from a single SM left a
            // 64 us tail after the last GEMM task (probe, 2048 and 3888 tokens).
            // With 4 * gridDim.x stripes > 64 a warp owns at most one row per tile.
            {
                const int stripes = static_cast<int>(gridDim.x) * 4;
                const int stripe = static_cast<int>(blockIdx.x) * 4 + warpgroup::warpid();
                const int lane = laneid();
                for (int m = 0; m < m_tiles; ++m) {
                    const int base = m * M_TILE;
                    int r = stripe - base % stripes;
                    if (r < 0) r += stripes;
                    if (r >= M_TILE) continue;
                    if (lane == 0)
                        wait_geq_or_trap<false>(g, g.c.y_ready + m, static_cast<unsigned int>(y_ready_target<NC>()),
                                                SITE_WARPROLE_Y_READY, m);
                    __syncwarp();
                    for (; r < M_TILE; r += stripes) tc::push_routed_row(g.c, base + r, lane);
                }
            }
            comm::combine_finish(g.c);
            if (warpgroup::laneid() == 0) stamp(g, 5);
        }
    } else if (role == NC) {
        // ------------------------------------------------ producer warpgroup
        warpgroup::decrease_registers<gemm::producer_regs<1>()>();
        if (warpgroup::warpid() != 0) {
            // Warps 1-3: fused SwiGLU/quant epilogue of every W13 tile the consumers park in
            // smem.d, so the tensor cores never wait for it.  Hand-off through two named
            // barriers; this side also publishes hidden_ready for the tile.
            const int t = static_cast<int>(threadIdx.x) - (NC * 128 + 32);
            asm volatile("bar.arrive %0, %1;" :: "n"(BAR_D_EMPTY), "n"(handoff_threads<NC>()) : "memory");
            for (;;) {
                asm volatile("bar.sync %0, %1;" :: "n"(BAR_D_FULL), "n"(handoff_threads<NC>()) : "memory");
                const int m_tile = *reinterpret_cast<volatile int *>(&epi_task[0]);
                const int i_tile = *reinterpret_cast<volatile int *>(&epi_task[1]);
                if (m_tile < 0) break;
                epilogue::swiglu_quant_tile<EPI_THREADS>(t, smem.d[0], smem.d[1], g.hidden, g.hidden_scale,
                                                         m_tile, i_tile, g.limit);
                asm volatile("bar.arrive %0, %1;" :: "n"(BAR_D_EMPTY), "n"(handoff_threads<NC>()) : "memory");
                asm volatile("bar.sync %0, %1;" :: "n"(BAR_EPI), "n"(EPI_THREADS) : "memory");
                if (t == 0) {   // barrier cumulativity + gpu-scope fence publish the whole tile
                    asm volatile("{fence.acq_rel.gpu;}" ::: "memory");
                    comm::add_release_gpu(g.hidden_ready + m_tile, 1u);
                }
            }
        } else {
            int64_t stage_counter = 0;
            // Lane 0 claims one task ahead: the atomic's round trip then hides behind the
            // loads of the task in flight instead of stalling the producer at every task start.
            int next = 0;
            if (laneid() == 0) next = static_cast<int>(atomicAdd(g.task_cursor, 1u));
            for (unsigned int k = 0;; ++k) {
                // Publish the claimed index to the consumers.  The mailbox entry for task k
                // is not rewritten before task k + 2 starts, and the producer cannot get
                // there until every consumer has drained task k + 1 down to STAGES stages
                // (STAGES < K blocks of any task), so entry k was read long before its reuse.
                const int t = __shfl_sync(0xffffffffu, next, 0);
                const task tk = decode_task<NC>(static_cast<int64_t>(t), s);
                if (laneid() == 0) {
                    *reinterpret_cast<volatile int *>(&task_box[k & 1u]) = tk.kind == task_kind::none ? -1 : t;
                    __threadfence_block();
                    *reinterpret_cast<volatile unsigned int *>(&task_seq) = k + 1u;
                }
                if (tk.kind == task_kind::none) break;
                if (laneid() == 0) next = static_cast<int>(atomicAdd(g.task_cursor, 1u));
                if (tk.kind == task_kind::w13) {
                    if (laneid() == 0)
                        wait_geq_or_trap<false>(g, g.c.x_ready + tk.minibatch,
                                                static_cast<unsigned int>(x_ready_target(s, tk.minibatch)),
                                                SITE_WARPROLE_X_READY, tk.minibatch);
                    __syncwarp();
                    asm volatile("{fence.proxy.async.global;}" ::: "memory");
                    const int expert = g.c.m_indices[tk.m_tile * M_TILE];
                    epilogue::producer_w13_task<NC, STAGES>(g.routed_x_gl, g.c.routed_x_scale, g.w13i, smem, full, empty,
                                                            stage_counter, tk.m_tile, expert, tk.n_index);
                } else {
                    if (laneid() == 0)
                        wait_geq_or_trap<false>(g, g.hidden_ready + tk.m_tile,
                                                static_cast<unsigned int>(hidden_ready_target<NC>()),
                                                SITE_WARPROLE_HIDDEN_READY, tk.m_tile);
                    __syncwarp();
                    asm volatile("{fence.proxy.async.global;}" ::: "memory");
                    const int expert = g.c.m_indices[tk.m_tile * M_TILE];
                    gemm::producer_task<NC, STAGES>(g.hidden_gl, g.hidden_scale, g.w2, smem, full, empty, stage_counter,
                                                    tk.m_tile, expert, tk.n_index * NC, W2_K_BLOCKS);
                }
            }
        }
    } else {
        // ------------------------------------------------ consumer warpgroup(s)
        warpgroup::increase_registers<gemm::consumer_regs<NC, 1>()>();
        int64_t stage_counter = 0;
        const int barrier_id = role + 1;
        bool first_task = true;
        for (unsigned int k = 0;; ++k) {
            unsigned long long iters = 0;
            while (*reinterpret_cast<volatile unsigned int *>(&task_seq) < k + 1u) {
                __nanosleep(32);
                if (++iters >= g.spin_limit) trap(g, ERR_TIMEOUT, SITE_WARPROLE_TASK_BOX, k, k + 1u, 0, iters);
            }
            __threadfence_block();
            const int t = *reinterpret_cast<volatile int *>(&task_box[k & 1u]);
            if (t < 0) break;
            const task tk = decode_task<NC>(static_cast<int64_t>(t), s);
            if (tk.kind == task_kind::w13) {
                if (warpgroup::laneid() == 0)
                    wait_geq_or_trap<false>(g, g.c.x_ready + tk.minibatch,
                                            static_cast<unsigned int>(x_ready_target(s, tk.minibatch)),
                                            SITE_WARPROLE_X_READY, tk.minibatch);
                warpgroup::sync(barrier_id);
                if (first_task) {
                    if (threadIdx.x == 0) stamp(g, 3);
                    first_task = false;
                }
                const int expert = g.c.m_indices[tk.m_tile * M_TILE];
                epilogue::consumer_w13_gemm_to_d<NC, STAGES>(
                    smem, full, empty, stage_counter, role, g.w13i_scale, expert, tk.n_index,
                    [] { asm volatile("bar.sync %0, %1;" :: "n"(BAR_D_EMPTY), "n"(handoff_threads<NC>()) : "memory"); });
                if (threadIdx.x == 0) {
                    *reinterpret_cast<volatile int *>(&epi_task[0]) = tk.m_tile;
                    *reinterpret_cast<volatile int *>(&epi_task[1]) = tk.n_index;
                }
                // the epilogue warps take it from here (SwiGLU/quant + hidden_ready)
                asm volatile("bar.arrive %0, %1;" :: "n"(BAR_D_FULL), "n"(handoff_threads<NC>()) : "memory");
            } else {
                if (warpgroup::laneid() == 0)
                    wait_geq_or_trap<false>(g, g.hidden_ready + tk.m_tile,
                                            static_cast<unsigned int>(hidden_ready_target<NC>()),
                                            SITE_WARPROLE_HIDDEN_READY, tk.m_tile);
                warpgroup::sync(barrier_id);
                const int expert = g.c.m_indices[tk.m_tile * M_TILE];
                const int n_tile = tk.n_index * NC + role;
                gemm::stage_b_scale_row<NC, STAGES>(smem, role, g.w2_scale, expert, W2_N_TILES_128, n_tile, W2_K_BLOCKS);
                warpgroup::sync(barrier_id);
                gemm::acc_rt acc;
                gemm::consumer_task<NC, STAGES>(smem, full, empty, stage_counter, role, smem.b_scale[role], W2_K_BLOCKS, acc);
                gemm::store_bf16_tile_via(smem.w2d[role], barrier_id, g.routed_y_gl, acc, tk.m_tile, n_tile);
                if (warpgroup::laneid() == 0) {
                    tma::store_async_wait();                                   // routed_y tile landed
                    asm volatile("{fence.proxy.async.global;}" ::: "memory");
                    asm volatile("{fence.acq_rel.gpu;}" ::: "memory");
                    comm::add_release_gpu(g.c.y_ready + tk.m_tile, 1u);
                }
            }
        }
        if (threadIdx.x == 0) stamp(g, 4);
        // end of stream: release the epilogue warps with the sentinel
        asm volatile("bar.sync %0, %1;" :: "n"(BAR_D_EMPTY), "n"(handoff_threads<NC>()) : "memory");
        if (threadIdx.x == 0) *reinterpret_cast<volatile int *>(&epi_task[0]) = -1;
        asm volatile("bar.arrive %0, %1;" :: "n"(BAR_D_FULL), "n"(handoff_threads<NC>()) : "memory");
    }

    // ---------------------------------------------------- phase 5: all roles rejoin
    __syncthreads();
    if (role >= NC) return;   // producer and comm warps are done; consumers reduce
    if (threadIdx.x == 0) {
        wait_geq_or_trap<true>(g, g.c.push_done, static_cast<unsigned int>(g.c.ep_size), SITE_WARPROLE_PUSH_DONE, 0);
        stamp(g, 6);
    }
    asm volatile("bar.sync 4, %0;" :: "n"(128 * NC) : "memory");
    if (!g.reduce_off) {   // MOK_WARPROLE_REDUCE_OFF: benchmark-only
        // Weighted reduce of the local tokens.  Register-only loads kept too little in flight
        // (about a quarter of HBM bandwidth), so thread 0 stages the combine rows of the next
        // RSTAGES tokens into the idle GEMM ring with TMA bulk copies and all consumer threads
        // reduce from shared memory; each thread owns 8 consecutive columns per pass.
        constexpr int RSTAGES = reduce_stages<NC, STAGES>();
        constexpr int THREADS = 128 * NC;
        constexpr int PASSES = HIDDEN / (8 * THREADS);
        char *slots = reinterpret_cast<char *>(&smem);
        const int stride = static_cast<int>(gridDim.x);
        const int n = (g.c.num_local_tokens - static_cast<int>(blockIdx.x) + stride - 1) / stride;
        auto issue = [&](int j) {   // thread 0: stage this CTA's j-th token into slot j % RSTAGES
            const size_t route_base = static_cast<size_t>(blockIdx.x + j * stride) * TOPK;
            char *slot = slots + (j % RSTAGES) * REDUCE_TOKEN_BYTES;
            uint32_t bytes = 0;
#pragma unroll
            for (int r = 0; r < TOPK; ++r) bytes += g.topk_ids[route_base + r] < 0 ? 0u : REDUCE_ROW_BYTES;
            tma::expect_bytes(reduce_full[j % RSTAGES], bytes);
#pragma unroll
            for (int r = 0; r < TOPK; ++r)
                if (g.topk_ids[route_base + r] >= 0)
                    bulk_load(slot + r * REDUCE_ROW_BYTES, g.combine_local + (route_base + r) * HIDDEN,
                              REDUCE_ROW_BYTES, reduce_full[j % RSTAGES]);
        };
        if (threadIdx.x == 0) {
            // The rows were written by the peers' comm warps (generic proxy, acquired above through
            // push_done); the bulk copies read them through the async proxy.
            asm volatile("{fence.proxy.async.global;}" ::: "memory");
            for (int j = 0; j < RSTAGES && j < n; ++j) issue(j);
        }
        for (int j = 0; j < n; ++j) {
            const int token = static_cast<int>(blockIdx.x) + j * stride;
            const size_t route_base = static_cast<size_t>(token) * TOPK;
            int ids[TOPK];
            float w[TOPK];
#pragma unroll
            for (int r = 0; r < TOPK; ++r) {
                ids[r] = g.topk_ids[route_base + r];
                w[r] = g.weights[route_base + r];
            }
            const char *slot = slots + (j % RSTAGES) * REDUCE_TOKEN_BYTES;
            wait(reduce_full[j % RSTAGES], (j / RSTAGES) & 1);
            __nv_bfloat16 *out_row = g.output + static_cast<size_t>(token) * HIDDEN;
#pragma unroll
            for (int p = 0; p < PASSES; ++p) {
                const int col = (p * THREADS + static_cast<int>(threadIdx.x)) * 8;
                uint4 raw[TOPK];
#pragma unroll
                for (int r = 0; r < TOPK; ++r)
                    raw[r] = ids[r] < 0 ? make_uint4(0u, 0u, 0u, 0u)
                                        : *reinterpret_cast<const uint4 *>(slot + r * REDUCE_ROW_BYTES + col * 2);
                *reinterpret_cast<uint4 *>(out_row + col) = reduce_chunk(raw, ids, w);
            }
            asm volatile("bar.sync 4, %0;" :: "n"(THREADS) : "memory");   // every thread is done with the slot
            if (threadIdx.x == 0 && j + RSTAGES < n) issue(j + RSTAGES);
        }
    }
    if (g.probe != nullptr) {   // benchmark-only: one extra barrier so slot 7 covers every reduce warp
        asm volatile("bar.sync 4, %0;" :: "n"(128 * NC) : "memory");
        if (threadIdx.x == 0) stamp(g, 7);
    }
    if (threadIdx.x == 0) {
        // Every producer has stopped claiming (phase-5 __syncthreads above), so the last CTA
        // to arrive rewinds the cursor for the next launch; the prepare kernel zeroes it too,
        // but the benchmark knob modes call the fused kernel back to back without prepare.
        if (atomicAdd(g.cta_done, 1u) + 1u == gridDim.x) {
            atomicExch(g.task_cursor, 0u);
            atomicExch(g.cta_done, 0u);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && !g.skip_waits) {
        // Contract closure: every counter must have landed exactly on its target.
        for (int q = 0; q < minibatches(s); ++q) {
            const unsigned int v = comm::load_acquire_gpu(g.c.x_ready + q);
            if (v != static_cast<unsigned int>(x_ready_target(s, q)))
                trap(g, ERR_CONTRACT, SITE_WARPROLE_X_READY, q, x_ready_target(s, q), v, 0);
        }
        for (int m = 0; m < m_tiles; ++m) {
            const unsigned int h = comm::load_acquire_gpu(g.hidden_ready + m);
            if (h != static_cast<unsigned int>(hidden_ready_target<NC>()))
                trap(g, ERR_CONTRACT, SITE_WARPROLE_HIDDEN_READY, m, hidden_ready_target<NC>(), h, 0);
            const unsigned int y = comm::load_acquire_gpu(g.c.y_ready + m);
            if (y != static_cast<unsigned int>(y_ready_target<NC>()))
                trap(g, ERR_CONTRACT, SITE_WARPROLE_Y_READY, m, y_ready_target<NC>(), y, 0);
        }
        const unsigned int p = comm::load_acquire_sys(g.c.push_done);
        if (p != static_cast<unsigned int>(g.c.ep_size))
            trap(g, ERR_CONTRACT, SITE_WARPROLE_PUSH_DONE, 0, g.c.ep_size, p, 0);
    }
}

// ---------------------------------------------------------------------------
// prepare: zero the per-launch counters.
// ---------------------------------------------------------------------------
struct prepare_globals {
    unsigned int *x_ready; int64_t x_ready_count;
    unsigned int *hidden_ready; int64_t hidden_ready_count;
    unsigned int *y_ready; int64_t y_ready_count;
    unsigned int *push_done_local;
    unsigned int *push_done;
    unsigned int *input_expected_scratch;
};

__global__ void prepare_kernel(prepare_globals p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t max_count = p.x_ready_count > p.hidden_ready_count ? p.x_ready_count : p.hidden_ready_count;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < max_count; i += stride) {
        if (i < p.x_ready_count) p.x_ready[i] = 0u;
        if (i < p.hidden_ready_count) p.hidden_ready[i] = 0u;
        if (i < p.y_ready_count) p.y_ready[i] = 0u;
        if (i == 0) {
            *p.push_done_local = 0u;
            *p.push_done = 0u;
            p.input_expected_scratch[0] = 0u;   // rank-barrier scratch
            p.input_expected_scratch[1] = 0u;   // task cursor
            p.input_expected_scratch[2] = 0u;   // CTAs done
        }
    }
}

inline unsigned int *u32_ptr(const at::Tensor &t) { return reinterpret_cast<unsigned int *>(t.data_ptr<int>()); }

inline void entry_prepare_out(at::Tensor x_ready, at::Tensor hidden_ready, at::Tensor y_ready,
                              at::Tensor push_done_local, at::Tensor push_done, at::Tensor input_expected_scratch) {
    for (const at::Tensor *t : {&x_ready, &hidden_ready, &y_ready, &push_done_local, &push_done, &input_expected_scratch})
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous() && t->dim() == 1 && t->numel() >= 1,
                    "warprole counters must be 1-D contiguous CUDA int32");
    TORCH_CHECK(input_expected_scratch.numel() >= 3,
                "input_expected_scratch must be int32 [3]: barrier scratch, task cursor, CTAs done");
    c10::cuda::CUDAGuard device_guard(x_ready.device());
    prepare_globals p{u32_ptr(x_ready), x_ready.numel(), u32_ptr(hidden_ready), hidden_ready.numel(),
                      u32_ptr(y_ready), y_ready.numel(), u32_ptr(push_done_local), u32_ptr(push_done),
                      u32_ptr(input_expected_scratch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_ready.get_device());
    const int64_t max_count = std::max(x_ready.numel(), hidden_ready.numel());
    const int blocks = static_cast<int>(std::min<int64_t>((max_count + 255) / 256, 64));
    prepare_kernel<<<blocks, 256, 0, stream>>>(p);
    CUDACHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// forward entry
// ---------------------------------------------------------------------------
// "1" enables a benchmark-only knob; anything else (including unset) leaves it off.
// Benchmark-only timeline probe buffer, [num_sms, PROBE_SLOTS] int64 ns, allocated on
// first use when MOK_WARPROLE_PROBE=1 and read back through fp8_block_warprole_probe_read.
inline at::Tensor &probe_buffer() {
    static at::Tensor buffer;
    return buffer;
}
inline at::Tensor probe_read() {
    at::Tensor &buffer = probe_buffer();
    TORCH_CHECK(buffer.defined(), "no probe recorded: set MOK_WARPROLE_PROBE=1 before a fused call");
    return buffer;
}

inline int env_flag(const char *name) {
    const char *v = std::getenv(name);
    return (v != nullptr && v[0] == '1' && v[1] == '\0') ? 1 : 0;
}

template <int NC, int STAGES>
inline void entry_out(
        at::Tensor x, std::vector<int64_t> x_ptrs, at::Tensor x_scale, std::vector<int64_t> x_scale_ptrs,
        at::Tensor routed_x, at::Tensor routed_x_scale, at::Tensor m_indices,
        at::Tensor schedule_peer_rank, at::Tensor schedule_peer_token_idx, at::Tensor num_tokens,
        at::Tensor tokens_per_expert, int64_t topk,
        at::Tensor w13i, at::Tensor w13i_scale, at::Tensor w2, at::Tensor w2_scale,
        at::Tensor hidden, at::Tensor hidden_scale, at::Tensor routed_y,
        std::vector<int64_t> combine_ptrs, at::Tensor combine_local, at::Tensor weights, at::Tensor topk_ids,
        at::Tensor output, std::vector<int64_t> push_done_ptrs, int64_t ep_rank,
        at::Tensor x_ready, at::Tensor hidden_ready, at::Tensor y_ready, at::Tensor push_done_local,
        at::Tensor barrier_buffer, at::Tensor barrier_target, int64_t barrier_multicast_ptr,
        at::Tensor input_expected_scratch, int64_t trap_record_ptr, double swiglu_limit, int64_t spin_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    comm::globals cg{};
    comm::bench::fill_comm_globals(cg, x, x_ptrs, x_scale, x_scale_ptrs, routed_x, routed_x_scale, m_indices,
                                   schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert, topk,
                                   routed_y, combine_ptrs, push_done_ptrs, ep_rank, x_ready, y_ready, push_done_local);
    const int64_t capacity = routed_x.size(0);
    const int64_t experts = w13i.size(0);
    TORCH_CHECK(cg.topk == TOPK, "warprole is specialized for top-6 routing");
    TORCH_CHECK(w13i.dim() == 3 && w13i.size(1) == 2 * INTER && w13i.size(2) == HIDDEN
                    && w13i.scalar_type() == at::kFloat8_e4m3fn && w13i.is_contiguous() && w13i.is_cuda(),
                "w13i must be contiguous CUDA fp8 [E,4096,4096] (interleaved gate/up)");
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == experts && w2.size(1) == HIDDEN && w2.size(2) == INTER
                    && w2.scalar_type() == at::kFloat8_e4m3fn && w2.is_contiguous() && w2.is_cuda(),
                "w2 must be contiguous CUDA fp8 [E,4096,2048]");
    TORCH_CHECK(w13i_scale.is_cuda() && w13i_scale.scalar_type() == at::kFloat && w13i_scale.is_contiguous()
                    && w13i_scale.dim() == 3 && w13i_scale.size(0) == experts && w13i_scale.size(1) == W13_N_TILES_128
                    && w13i_scale.size(2) == W13_K_BLOCKS,
                "w13i_scale must be float32 [E,32,32]");
    TORCH_CHECK(w2_scale.is_cuda() && w2_scale.scalar_type() == at::kFloat && w2_scale.is_contiguous()
                    && w2_scale.dim() == 3 && w2_scale.size(0) == experts && w2_scale.size(1) == W2_N_TILES_128
                    && w2_scale.size(2) == W2_K_BLOCKS,
                "w2_scale must be float32 [E,32,16]");
    TORCH_CHECK(experts == cg.num_local_experts, "weights must cover exactly the local experts");
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::kFloat8_e4m3fn && hidden.is_contiguous()
                    && hidden.dim() == 2 && hidden.size(0) == capacity && hidden.size(1) == INTER,
                "hidden must be contiguous CUDA fp8 [capacity,2048]");
    TORCH_CHECK(hidden_scale.is_cuda() && hidden_scale.scalar_type() == at::kFloat && hidden_scale.is_contiguous()
                    && hidden_scale.dim() == 2 && hidden_scale.size(0) == capacity && hidden_scale.size(1) == INTER / 128,
                "hidden_scale must be float32 [capacity,16]");
    const int64_t routes = static_cast<int64_t>(cg.num_local_tokens) * TOPK;
    TORCH_CHECK(combine_local.is_cuda() && combine_local.scalar_type() == at::kBFloat16 && combine_local.is_contiguous()
                    && combine_local.dim() == 2 && combine_local.size(0) == routes && combine_local.size(1) == HIDDEN,
                "combine_local must be bf16 [tokens*topk,4096]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat && weights.is_contiguous() && weights.numel() == routes,
                "weights must be float32 [tokens*topk]");
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && topk_ids.numel() == routes,
                "topk_ids must be int32 [tokens*topk]");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16 && output.is_contiguous() && output.dim() == 2
                    && output.size(0) == cg.num_local_tokens && output.size(1) == HIDDEN,
                "output must be bf16 [tokens,4096]");
    TORCH_CHECK(hidden_ready.is_cuda() && hidden_ready.scalar_type() == at::kInt && hidden_ready.is_contiguous()
                    && hidden_ready.numel() >= capacity / 64, "hidden_ready must be int32 [m_tiles]");
    for (const at::Tensor *t : {&barrier_buffer, &barrier_target, &input_expected_scratch})
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous() && t->numel() >= 1,
                    "barrier state must be CUDA int32");
    TORCH_CHECK(input_expected_scratch.numel() >= 3,
                "input_expected_scratch must be int32 [3]: barrier scratch, task cursor, CTAs done");
    TORCH_CHECK(barrier_multicast_ptr > 0 && trap_record_ptr > 0 && spin_limit > 0,
                "barrier multicast pointer, trap record pointer and spin limit must be positive");
    kittens::py::tensor_check<gemm::a_gl>(routed_x);
    kittens::py::tensor_check<gemm::a_gl>(hidden);
    kittens::py::tensor_check<gemm::b_gl>(w13i);
    kittens::py::tensor_check<gemm::b_gl>(w2);
    kittens::py::tensor_check<gemm::d_gl>(routed_y);
    kittens::py::device_check(x, w13i, w2, hidden, routed_y, combine_local, weights, topk_ids, output);

    globals g(kittens::py::tensor_to_gl<gemm::a_gl>(routed_x), kittens::py::tensor_to_gl<gemm::a_gl>(hidden),
              kittens::py::tensor_to_gl<gemm::b_gl>(w13i), kittens::py::tensor_to_gl<gemm::b_gl>(w2),
              kittens::py::tensor_to_gl<gemm::d_gl>(routed_y));
    g.c = cg;
    g.w13i_scale = w13i_scale.data_ptr<float>();
    g.w2_scale = w2_scale.data_ptr<float>();
    g.hidden = static_cast<uint8_t *>(hidden.data_ptr());
    g.hidden_scale = hidden_scale.data_ptr<float>();
    g.hidden_ready = u32_ptr(hidden_ready);
    g.limit = static_cast<float>(swiglu_limit);
    g.combine_local = reinterpret_cast<const __nv_bfloat16 *>(combine_local.data_ptr());
    g.weights = weights.data_ptr<float>();
    g.topk_ids = topk_ids.data_ptr<int>();
    g.output = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
    g.barrier_flag = u32_ptr(barrier_buffer);
    g.barrier_target = u32_ptr(barrier_target);
    g.barrier_multicast_ptr = reinterpret_cast<unsigned int *>(barrier_multicast_ptr);
    g.input_expected_scratch = u32_ptr(input_expected_scratch);
    g.task_cursor = g.input_expected_scratch + 1;
    g.cta_done = g.input_expected_scratch + 2;
    g.trap_record = utils::mok_resolve_trap_record(trap_record_ptr);
    g.spin_limit = static_cast<unsigned long long>(spin_limit);
    g.skip_waits = env_flag("MOK_WARPROLE_NO_DEPS");
    g.comm_off = env_flag("MOK_WARPROLE_COMM_OFF");
    g.reduce_off = env_flag("MOK_WARPROLE_REDUCE_OFF");
    TORCH_CHECK(!g.comm_off || g.skip_waits,
                "MOK_WARPROLE_COMM_OFF=1 requires MOK_WARPROLE_NO_DEPS=1: without dispatch the GEMM roles would wait forever");
    TORCH_CHECK(!g.reduce_off || g.skip_waits,
                "MOK_WARPROLE_REDUCE_OFF=1 requires MOK_WARPROLE_NO_DEPS=1: it is a benchmark-only knob");

    constexpr int SMEM = sizeof(gemm::smem_layout<NC, STAGES>) + 1024;
    constexpr int THREADS = gemm::num_threads<NC>();
    auto *kernel_ptr = kernel<NC, STAGES>;
    CUDACHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int num_sms = 0;
    CUDACHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.get_device()));
    int blocks_per_sm = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr, THREADS, SMEM));
    TORCH_CHECK(blocks_per_sm >= 1, "warprole kernel does not fit one CTA per SM (smem ", SMEM, " bytes)");
    if (env_flag("MOK_WARPROLE_PROBE")) {
        at::Tensor &buffer = probe_buffer();
        if (!buffer.defined() || buffer.numel() != static_cast<int64_t>(num_sms) * PROBE_SLOTS
            || buffer.device() != x.device())
            buffer = at::zeros({num_sms, PROBE_SLOTS}, x.options().dtype(at::kLong));
        buffer.zero_();
        g.probe = reinterpret_cast<unsigned long long *>(buffer.data_ptr<int64_t>());
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    kernel_ptr<<<num_sms, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::warprole::fused
#endif  // KITTENS_SM90
