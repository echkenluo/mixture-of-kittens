#pragma once

// Numerical compute-stage helpers for the fixed-resident terminal worker.
//
// The device cursor and its 65-task/M64 decode live in
// sm90_fp8_block_megakernel.cuh.  This header turns those logical tasks into
// real cluster-2 FP8 block-scale WGMMA work and publishes the three numerical
// stage boundaries:
//
//   gate/up N128 task -> gate_up_ready[m,n128] += 1
//   activation M64    -> hidden_ready[m] = 1
//   W2 N128 task      -> y_ready[m] += 1
//
// A logical N128 task is always expanded as two CTA-local N64 tiles,
// n64 = 2*n128 + cta_rank.  W13 stores gate and up in the first and second
// 2048-column halves of one BF16 [M,4096] tensor respectively.  The helper
// intentionally owns no dispatch, combine, or terminal communication work.

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

// CTA-local N64 coordinate within one logical stage.  Keep this spelling
// explicit: the terminal scheduler commits N128 tasks, while WGMMA consumes
// one N64 half per CTA in the cluster.
MOK_TERMINAL_COMPUTE_HD int logical_n64(
        const terminal::logical_coordinate &coordinate, int cta_rank) {
    return terminal::n64_for_cta(coordinate, cta_rank);
}

// W13 packs gate followed by up.  The relative coordinate is still exactly
// 2*n128+cta_rank; only the storage/weight view receives the +32 N64 offset.
MOK_TERMINAL_COMPUTE_HD int w13_storage_n64(
        const terminal::logical_coordinate &coordinate, int cta_rank) {
    const int n64 = logical_n64(coordinate, cta_rank);
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
    const int n64 = w13_storage_n64(coordinate, cta_rank);
    const pipeline::tile tile{
        n64,
        coordinate.global_m,
        coordinate.global_m * terminal::M_TILE,
    };
    pipeline::run_tile(
        problem, tile, expert, cta_rank, phasebits, ready_phase,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);
    warpgroup::sync(0);

    // Both N64 stores must be globally complete before the single logical
    // N128 arrival is published by rank 0.
    everyone::tma::cluster::sync();
    if (cta_rank == 0 && threadIdx.x == 0) {
        const int64_t index = terminal::counter_index(
            terminal::ready_counter::gate_up_tile,
            coordinate.global_m, coordinate.n128);
        add_release_gpu(ready.gate_up_ready + index, 1u);
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
        for (int n128 = 0; n128 < terminal::W13_N_TILES; ++n128)
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

    const int n64 = logical_n64(coordinate, cta_rank);
    const pipeline::tile tile{
        n64,
        coordinate.global_m,
        coordinate.global_m * terminal::M_TILE,
    };
    pipeline::run_tile(
        problem, tile, expert, cta_rank, phasebits, ready_phase,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);
    warpgroup::sync(0);

    everyone::tma::cluster::sync();
    if (cta_rank == 0 && threadIdx.x == 0) {
        const int64_t index = terminal::counter_index(
            terminal::ready_counter::y_routed, coordinate.global_m);
        add_release_gpu(ready.y_ready + index, 1u);
    }
    everyone::tma::cluster::sync();
}

#endif  // defined(KITTENS_SM90)

}  // namespace mok_sm90::fp8_block_terminal_compute
