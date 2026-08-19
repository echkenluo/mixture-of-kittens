#pragma once

// Numerical compute-stage helpers for the fixed-resident terminal worker.
//
// The device cursor and its 33-ticket/M64 decode live in
// sm90_fp8_block_megakernel.cuh.  This header turns those logical tasks into
// real cluster-2 FP8 block-scale WGMMA work and publishes the three numerical
// stage boundaries:
//
//   gate/up N256 ticket -> two gate_up_ready[m,n128] entries += 1
//   activation M64    -> hidden_ready[m] = 1
//   W2 N256 ticket    -> y_ready[m] += 2
//
// Each logical N256 ticket executes two complete legacy N128 tiles in order.
// Each subtile drains WGMMA/global stores, updates the persistent phase bits,
// and crosses a cluster barrier before the next subtile reuses the same A/B/D
// shared storage and the same single-output accumulator.  A is deliberately
// reloaded: this candidate isolates cursor/task-boundary savings from a wider
// arithmetic implementation.  W13 stores gate and up in the first and second
// 2048-column halves of one BF16 [M,4096] tensor respectively.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sm90_fp8_block_megakernel.cuh"
#include "sm90_fp8_block_pipeline_primitives.cuh"

namespace mok_sm90::fp8_block_terminal_compute {

namespace terminal = mok_sm90::fp8_block_terminal;
namespace pipeline = mok_sm90::fp8_block_pipeline;

#if defined(__CUDACC__)
#define MOK_TERMINAL_COMPUTE_HD __host__ __device__ constexpr
#else
#define MOK_TERMINAL_COMPUTE_HD constexpr
#endif

MOK_TERMINAL_COMPUTE_HD bool is_w13_stage(terminal::logical_stage stage) {
    return stage == terminal::logical_stage::gate
        || stage == terminal::logical_stage::up;
}

// CTA-local N64 coordinate for one of the two sequential N128 subtiles.
MOK_TERMINAL_COMPUTE_HD int logical_n64(
        const terminal::logical_coordinate &coordinate, int subtask,
        int cta_rank) {
    return terminal::n64_for_subtask_cta(coordinate, subtask, cta_rank);
}

// W13 packs gate followed by up.  Only the storage/weight view receives the
// +32 N64 offset; the logical ticket remains stage-relative N256.
MOK_TERMINAL_COMPUTE_HD int w13_storage_n64(
        const terminal::logical_coordinate &coordinate, int subtask,
        int cta_rank) {
    const int n64 = logical_n64(coordinate, subtask, cta_rank);
    if (n64 < 0 || !is_w13_stage(coordinate.stage))
        return -1;
    return n64
        + (coordinate.stage == terminal::logical_stage::up
               ? terminal::W13_N64_SUBTILES
               : 0);
}

#undef MOK_TERMINAL_COMPUTE_HD

#if defined(KITTENS_SM90)

using namespace kittens;

using pipeline::a_st;
using pipeline::b_st;
using pipeline::d_st;
using pipeline::PIPE_DEPTH;

constexpr int SEQUENTIAL_N128_LIVE_ACCUMULATOR_WORDS =
    2 * static_cast<int>(sizeof(pipeline::acc_rt) / sizeof(uint32_t));
static_assert(sizeof(pipeline::acc_rt) == 32 * sizeof(uint32_t),
              "legacy M64xN64 accumulator footprint changed");
static_assert(SEQUENTIAL_N128_LIVE_ACCUMULATOR_WORDS == 64,
              "sequential N256 must retain one total plus one partial");

struct readiness {
    unsigned int *gate_up_ready;  // [M64,16], expected value 2
    unsigned int *hidden_ready;   // [M64], expected value 1
    unsigned int *y_ready;        // [M64], expected value 32
};

struct activation_problem {
    const __nv_bfloat16 *gate_up;  // [M,4096], gate then up
    uint8_t *hidden;               // [M,2048], FP8 E4M3
    float *hidden_scale;           // [M,16], FP32 K128 scales
    float limit;
};

__device__ __forceinline__ unsigned int load_acquire_gpu(
        const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void add_release_gpu(
        unsigned int *address, unsigned int value) {
    asm volatile("{red.release.gpu.global.add.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void store_release_gpu(
        unsigned int *address, unsigned int value) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void wait_until_at_least(
        const unsigned int *address, unsigned int expected) {
    while (load_acquire_gpu(address) < expected)
        __nanosleep(128);
}

// Execute one unchanged N128 arithmetic tile.  The final warpgroup sync drains
// the generic D store, while the acq_rel cluster barrier keeps the two CTA
// ranks lockstep before either rank mutates the shared phase state for the next
// N128 subtile.
template <typename GemmProblem>
__device__ __forceinline__ void run_sequential_n128_subtask(
        const GemmProblem &problem, int n64, int global_m, int expert,
        int cta_rank, uint32_t &phasebits, uint32_t &ready_phase,
        a_st (&a_smem)[PIPE_DEPTH], b_st (&b_smem)[PIPE_DEPTH],
        d_st &d_smem, semaphore (&inputs_arrived)[PIPE_DEPTH],
        semaphore (&inputs_finished)[PIPE_DEPTH],
        semaphore (&inputs_ready)[PIPE_DEPTH]) {
    const pipeline::tile tile{
        n64,
        global_m,
        global_m * terminal::M_TILE,
    };
    pipeline::run_tile(
        problem, tile, expert, cta_rank, phasebits, ready_phase,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);
    warpgroup::sync(0);
    everyone::tma::cluster::sync();
}

// The caller supplies the persistent shared WGMMA workspace and phase state;
// both survive across heterogeneous W13/activation/W2 task claims.
template <typename GemmProblem>
__device__ __forceinline__ void run_w13_task(
        const GemmProblem &problem,
        const terminal::logical_coordinate &coordinate, int expert,
        int cta_rank, const readiness &ready, uint32_t &phasebits,
        uint32_t &ready_phase, a_st (&a_smem)[PIPE_DEPTH],
        b_st (&b_smem)[PIPE_DEPTH], d_st &d_smem,
        semaphore (&inputs_arrived)[PIPE_DEPTH],
        semaphore (&inputs_finished)[PIPE_DEPTH],
        semaphore (&inputs_ready)[PIPE_DEPTH]) {
    // Reuse the legacy N128 primitive twice.  The helper's cluster barrier is
    // the task boundary: all phase-bit and mbarrier transitions from subtile 0
    // happen-before either CTA starts subtile 1.
#pragma unroll 1
    for (int subtask = 0;
         subtask < terminal::N128_SUBTASKS_PER_N256; ++subtask) {
        const int n64 = w13_storage_n64(
            coordinate, subtask, cta_rank);
        run_sequential_n128_subtask(
            problem, n64, coordinate.global_m, expert, cta_rank,
            phasebits, ready_phase, a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    }

    // All four N64 stores are complete before rank 0 publishes the two legacy
    // N128 counter entries covered by this logical N256 ticket.
    if (cta_rank == 0 && threadIdx.x == 0) {
#pragma unroll
        for (int subtask = 0;
             subtask < terminal::N128_SUBTASKS_PER_N256; ++subtask) {
            const int n128 = terminal::n128_for_subtask(
                coordinate, subtask);
            const int64_t index = terminal::counter_index(
                terminal::ready_counter::gate_up_tile,
                coordinate.global_m, n128);
            add_release_gpu(ready.gate_up_ready + index, 1u);
        }
    }
    everyone::tma::cluster::sync();
}

__device__ __forceinline__ void run_activation_task(
        const activation_problem &problem,
        const terminal::logical_coordinate &coordinate, int cta_rank,
        const readiness &ready) {
    const int64_t counter_base = terminal::counter_index(
        terminal::ready_counter::gate_up_tile,
        coordinate.global_m, 0);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int n128 = 0; n128 < terminal::W13_N128_COUNTERS; ++n128)
            wait_until_at_least(
                ready.gate_up_ready + counter_base + n128, 2u);
    }
    __syncthreads();

    const int worker = cta_rank * terminal::THREADS_PER_CTA + threadIdx.x;
    const int first_row = coordinate.global_m * terminal::M_TILE;
#pragma unroll 1
    for (int row = first_row; row < first_row + terminal::M_TILE; ++row) {
        pipeline::activate_quant_worker(
            problem.gate_up, problem.hidden, problem.hidden_scale,
            row, worker, problem.limit);
    }

    // W2 consumes hidden through the TMA async proxy in run_tile.  Bridge
    // each CTA's completed generic stores into that proxy before publishing
    // hidden_ready; the CTA barrier brings all 128 writers to its thread 0.
    __syncthreads();
    if (threadIdx.x == 0)
        asm volatile("{fence.proxy.async.global;}" ::: "memory");
    everyone::tma::cluster::sync();
    if (cta_rank == 0 && threadIdx.x == 0) {
        const int64_t index = terminal::counter_index(
            terminal::ready_counter::hidden_row_block,
            coordinate.global_m);
        store_release_gpu(ready.hidden_ready + index, 1u);
    }
    everyone::tma::cluster::sync();
}

template <typename GemmProblem>
__device__ __forceinline__ void run_w2_task(
        const GemmProblem &problem,
        const terminal::logical_coordinate &coordinate, int expert,
        int cta_rank, const readiness &ready, uint32_t &phasebits,
        uint32_t &ready_phase, a_st (&a_smem)[PIPE_DEPTH],
        b_st (&b_smem)[PIPE_DEPTH], d_st &d_smem,
        semaphore (&inputs_arrived)[PIPE_DEPTH],
        semaphore (&inputs_finished)[PIPE_DEPTH],
        semaphore (&inputs_ready)[PIPE_DEPTH]) {
    const int64_t hidden_index = terminal::counter_index(
        terminal::ready_counter::hidden_row_block,
        coordinate.global_m);
    if (threadIdx.x == 0)
        wait_until_at_least(ready.hidden_ready + hidden_index, 1u);
    __syncthreads();

    // hidden was produced through the generic proxy while run_tile consumes
    // it through a rank-0 multicast TMA load.  The ready counter orders the
    // generic writes, but the TMA initiator must still bridge those writes
    // into the async proxy after its acquire and before issuing the load.
    if (cta_rank == 0 && threadIdx.x == 0)
        asm volatile("{fence.proxy.async.global;}" ::: "memory");

#pragma unroll 1
    for (int subtask = 0;
         subtask < terminal::N128_SUBTASKS_PER_N256; ++subtask) {
        const int n64 = logical_n64(coordinate, subtask, cta_rank);
        run_sequential_n128_subtask(
            problem, n64, coordinate.global_m, expert, cta_rank,
            phasebits, ready_phase, a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    }

    if (cta_rank == 0 && threadIdx.x == 0) {
        const int64_t index = terminal::counter_index(
            terminal::ready_counter::y_routed, coordinate.global_m);
        add_release_gpu(
            ready.y_ready + index,
            terminal::N128_SUBTASKS_PER_N256);
    }
    everyone::tma::cluster::sync();
}

#endif  // defined(KITTENS_SM90)

}  // namespace mok_sm90::fp8_block_terminal_compute
