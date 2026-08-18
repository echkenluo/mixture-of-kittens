#pragma once

// Single-GPU peer-emulation milestone for the terminal SM90 FP8 forward.
// The M0 probe launches one comm plus one compute cluster, both required to
// be co-resident.  The interface accepts one comm plus N compute clusters so
// a later occupancy-selected launch can widen the cursor consumer set without
// changing task semantics.  This is a correctness scaffold, not the terminal
// production occupancy choice.  There is no grid/rank barrier, split path, or
// fallback in this kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sm90_fp8_block_megakernel.cuh"
#include "sm90_fp8_block_terminal_comm_primitives.cuh"
#include "sm90_fp8_block_terminal_compute.cuh"
#include "sm90_fp8_block_terminal_route_flags.cuh"

namespace mok_sm90::fp8_block_terminal_full {

namespace terminal = fp8_block_terminal;
namespace comm = fp8_block_terminal_comm;
namespace compute = fp8_block_terminal_compute;
namespace route = fp8_block_terminal_route_flags;

constexpr int COMM_CLUSTERS = 1;
constexpr int M0_COMPUTE_CLUSTERS = 1;
constexpr int M0_FIXED_CLUSTERS = COMM_CLUSTERS + M0_COMPUTE_CLUSTERS;
constexpr int COMM_CLUSTER = 0;
constexpr int COMPUTE_CLUSTER = 1;
constexpr int MAX_EXPERTS = 16;
constexpr unsigned int STOP_TICKET = ~0u;

#if defined(KITTENS_SM90)

using namespace kittens;

template <typename GemmProblem>
struct globals {
    GemmProblem w13;
    GemmProblem w2;
    compute::activation_problem activation;
    compute::readiness ready;

    const uint8_t *x_peer[terminal::EP_SIZE];
    const float *x_scale_peer[terminal::EP_SIZE];
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

    const uint8_t *routed_y;
    uint8_t *combine_peer[terminal::EP_SIZE];
    unsigned int *route_ready_peer[terminal::EP_SIZE];
    const int *push_order;

    const float *weights;
    const int *topk_ids;
    __nv_bfloat16 *output;
    unsigned int *epilogue_claim;

    unsigned int *x_ready;
    unsigned int *cursor;
    unsigned int *worker_ticket;
    unsigned int *worker_failed;
    unsigned int *next_reduce_probe;
    unsigned int *reduce_done;
    unsigned int *comm_closed;
    unsigned int *comm_failed;
    unsigned int *task_visits;
    unsigned int *dispatch_visits;
    unsigned int *push_visits;
    unsigned int *reduce_visits;
    unsigned int *errors;
    unsigned int *dispatch_tiles_done;
    unsigned int *compute_started;
    unsigned int *overlap_witness;

    int compute_clusters;
    int minibatch_rows;
    int macrobatch_rows;
    unsigned int overlap_delay_after_first_dispatch_cycles;
    unsigned long long spin_limit;
};

__device__ __forceinline__ unsigned int claim_bounded(
        unsigned int *cursor, unsigned int limit) {
    while (true) {
        const unsigned int current = compute::load_acquire_gpu(cursor);
        if (current >= limit)
            return STOP_TICKET;
        const unsigned int prior = atomicCAS(cursor, current, current + 1u);
        if (prior == current)
            return current;
    }
}

__device__ __forceinline__ bool bounded_wait_gpu(
        const unsigned int *address, unsigned int expected,
        unsigned long long spin_limit) {
    for (unsigned long long spin = 0; spin < spin_limit; ++spin) {
        if (compute::load_acquire_gpu(address) >= expected)
            return true;
        __nanosleep(64);
    }
    return false;
}

template <typename GemmProblem>
__device__ void communication_role(
        const globals<GemmProblem> &g, int cta_rank) {
    __shared__ int expert_row_end[MAX_EXPERTS];
    __shared__ int wait_ok;
    if (threadIdx.x == 0)
        comm::build_expert_row_ends(g, expert_row_end);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int WARPS_PER_CTA = terminal::THREADS_PER_CTA / 32;
    constexpr int WARPS_PER_CLUSTER =
        terminal::CLUSTER_CTAS * WARPS_PER_CTA;
    const int cluster_warp = cta_rank * WARPS_PER_CTA + warp;
    const int active_rows = comm::bounded_valid_rows(g);
    const int m_tiles = active_rows / terminal::M_TILE;
    const terminal::logical_shape shape = terminal::make_logical_shape(
        active_rows, g.schedule_capacity,
        g.minibatch_rows, g.macrobatch_rows);
    if (!shape.valid) {
        if (cta_rank == 0 && threadIdx.x == 0)
            atomicAdd(g.errors, 1u);
        return;
    }

    // Dispatch never waits on compute.  Publishing complete M64s first is the
    // fixed-role progress edge that prevents a waiting compute worker from
    // excluding its producer from residency.  M64 publication follows the
    // same reverse-macrobatch ordered-minibatch map as the compute decoder,
    // so the first compute coordinate never waits behind an unrelated m=0.
    int dispatched_tiles = 0;
    for (int j = 0; j < shape.num_global_minibatches; ++j) {
        const terminal::ordered_minibatch_range minibatch =
            terminal::decode_ordered_minibatch(shape, j);
        for (int r = 0; r < minibatch.active_m_tiles; ++r) {
            const int m = minibatch.first_m_tile + r;
            const int first_row = m * terminal::M_TILE;
            for (int local = cluster_warp; local < terminal::M_TILE;
                 local += WARPS_PER_CLUSTER) {
                const int row = first_row + local;
                comm::dispatch_copy_row(g, expert_row_end, row, lane);
                __syncwarp(0xffffffffu);
                if (lane == 0)
                    atomicAdd(g.dispatch_visits + row, 1u);
            }
            __syncthreads();
            everyone::tma::cluster::sync();
            if (cta_rank == 0 && threadIdx.x == 0)
                compute::store_release_gpu(
                    g.x_ready + m, terminal::M_TILE);
            ++dispatched_tiles;
            if (cta_rank == 0 && threadIdx.x == 0)
                compute::store_release_gpu(
                    g.dispatch_tiles_done, dispatched_tiles);
            everyone::tma::cluster::sync();

            // Test-only finite delay makes the M0 overlap witness
            // deterministic.  It never observes compute state; the
            // production-default value is zero.
            if (dispatched_tiles == 1 && m_tiles > 1
                    && g.overlap_delay_after_first_dispatch_cycles != 0u) {
                unsigned int remaining =
                    g.overlap_delay_after_first_dispatch_cycles;
                while (remaining != 0u) {
                    const unsigned int step = remaining > 1024u
                        ? 1024u
                        : remaining;
                    __nanosleep(step);
                    remaining -= step;
                }
            }
        }
    }

    // Each M64 push waits only after every x producer has been published and
    // the fixed compute cluster is known resident.  Rows are consumed through
    // an explicit permutation so publication order is not assumed.
    for (int j = 0; j < shape.num_global_minibatches; ++j) {
        const terminal::ordered_minibatch_range minibatch =
            terminal::decode_ordered_minibatch(shape, j);
        for (int r = 0; r < minibatch.active_m_tiles; ++r) {
        const int m = minibatch.first_m_tile + r;
        if (threadIdx.x == 0) {
            wait_ok = bounded_wait_gpu(
                g.ready.y_ready + m, terminal::W2_N_TILES,
                g.spin_limit) ? 1 : 0;
        }
        __syncthreads();
        if (!wait_ok && threadIdx.x == 0) {
            atomicExch(g.comm_failed, 1u);
            atomicAdd(g.errors, 1u);
        }
        everyone::tma::cluster::sync();
        if (compute::load_acquire_gpu(g.comm_failed) != 0u)
            return;

        const int first = m * terminal::M_TILE;
        for (int position = cluster_warp; position < terminal::M_TILE;
             position += WARPS_PER_CLUSTER) {
            const int row = g.push_order[first + position];
            const bool in_tile = row >= first
                && row < first + terminal::M_TILE;
            if (in_tile) {
                route::push_routed_row_and_publish(g, row, lane);
                __syncwarp(0xffffffffu);
                if (lane == 0)
                    atomicAdd(g.push_visits + row, 1u);
            } else if (lane == 0) {
                atomicAdd(g.errors, 1u);
            }
        }
        __syncthreads();
        everyone::tma::cluster::sync();
        }
    }

    if (cta_rank == 0 && threadIdx.x == 0)
        compute::store_release_gpu(g.comm_closed, 1u);
}

template <typename GemmProblem>
__device__ void reduce_ready_tokens(
        const globals<GemmProblem> &g) {
    __shared__ int selected_peer;
    __shared__ int selected_token;
    __shared__ int selected_result;
    const int tokens_per_peer = g.num_local_tokens;
    const int total_tokens = g.ep_size * tokens_per_peer;

    for (unsigned long long iteration = 0;
         iteration < g.spin_limit; ++iteration) {
        if (threadIdx.x == 0) {
            const unsigned int probe = atomicAdd(g.next_reduce_probe, 1u);
            const int global_token = total_tokens == 0
                ? -1
                : static_cast<int>(probe % total_tokens);
            selected_peer = global_token < 0
                ? -1
                : global_token / tokens_per_peer;
            selected_token = global_token < 0
                ? -1
                : global_token % tokens_per_peer;
            selected_result = global_token < 0
                ? static_cast<int>(route::claim_result::invalid_token)
                : static_cast<int>(route::try_claim_ready_token(
                    g.route_ready_peer[selected_peer],
                    g.epilogue_claim + selected_peer * tokens_per_peer,
                    selected_token, tokens_per_peer));
        }
        __syncthreads();

        if (selected_result
                == static_cast<int>(route::claim_result::claimed)) {
            const size_t route_stride =
                static_cast<size_t>(tokens_per_peer) * terminal::TOP_K;
            const size_t output_stride =
                static_cast<size_t>(tokens_per_peer) * g.hidden_size;
            const auto *combine = reinterpret_cast<const __nv_bfloat16 *>(
                g.combine_peer[selected_peer]);
            route::reduce_claimed_token(
                combine, g.weights + selected_peer * route_stride,
                g.topk_ids + selected_peer * route_stride,
                g.output + selected_peer * output_stride,
                selected_token, g.hidden_size,
                threadIdx.x, blockDim.x);
            route::release_fence_system();
        }
        __syncthreads();

        if (threadIdx.x == 0
                && selected_result
                    == static_cast<int>(route::claim_result::claimed)) {
            const int global_token =
                selected_peer * tokens_per_peer + selected_token;
            atomicAdd(g.reduce_visits + global_token, 1u);
            atomicAdd(g.reduce_done, 1u);
        }
        __syncthreads();

        if (compute::load_acquire_gpu(g.reduce_done)
                    >= static_cast<unsigned int>(total_tokens)
                && compute::load_acquire_gpu(g.comm_closed) >= 1u)
            return;
        if (selected_result
                != static_cast<int>(route::claim_result::claimed))
            __nanosleep(64);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(g.errors, 1u);
}

template <typename GemmProblem>
__device__ void compute_and_reduce_role(
        const globals<GemmProblem> &g, int cta_rank, int worker_cluster,
        compute::a_st (&a_smem)[compute::PIPE_DEPTH],
        compute::b_st (&b_smem)[compute::PIPE_DEPTH],
        compute::d_st &d_smem,
        semaphore (&inputs_arrived)[compute::PIPE_DEPTH],
        semaphore (&inputs_finished)[compute::PIPE_DEPTH],
        semaphore (&inputs_ready)[compute::PIPE_DEPTH]) {
    uint32_t phasebits = 0xFFFF0000u;
    uint32_t ready_phase = 0u;
    __shared__ int current_expert;

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const terminal::logical_shape shape =
                terminal::make_logical_shape(
                    g.num_tokens[0], g.schedule_capacity,
                    g.minibatch_rows, g.macrobatch_rows);
            const unsigned int ticket = shape.valid
                ? claim_bounded(
                    g.cursor, static_cast<unsigned int>(shape.total_tasks))
                : STOP_TICKET;
            if (!shape.valid)
                atomicAdd(g.errors, 1u);
            asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                         "l"(g.worker_ticket + worker_cluster), "r"(ticket)
                         : "memory");
        }
        everyone::tma::cluster::sync();
        unsigned int ticket;
        asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                     : "=r"(ticket)
                     : "l"(g.worker_ticket + worker_cluster) : "memory");
        if (ticket == STOP_TICKET)
            break;

        const terminal::logical_shape shape = terminal::make_logical_shape(
            g.num_tokens[0], g.schedule_capacity,
            g.minibatch_rows, g.macrobatch_rows);
        const terminal::logical_coordinate coordinate =
            terminal::decode_logical_cursor(shape, ticket);
        if (!coordinate.valid) {
            if (cta_rank == 0 && threadIdx.x == 0)
                atomicAdd(g.errors, 1u);
            everyone::tma::cluster::sync();
            continue;
        }
        if (cta_rank == 0 && threadIdx.x == 0)
            atomicAdd(g.task_visits + ticket, 1u);

        if (coordinate.stage != terminal::logical_stage::activation) {
            if (threadIdx.x == 0) {
                const bool ready = bounded_wait_gpu(
                    g.x_ready + coordinate.global_m,
                    terminal::M_TILE, g.spin_limit);
                current_expert = ready
                    ? g.m_indices[coordinate.global_m * terminal::M_TILE]
                    : -1;
                if (!ready) {
                    atomicExch(g.worker_failed + worker_cluster, 1u);
                    atomicAdd(g.errors, 1u);
                }
                // dispatch_copy_row uses generic stores while run_w13_task's
                // rank-0 TMA consumes through the async proxy.
                if (ready && cta_rank == 0
                        && (coordinate.stage == terminal::logical_stage::gate
                            || coordinate.stage
                                == terminal::logical_stage::up))
                    asm volatile("{fence.proxy.async.global;}" ::: "memory");
            }
            __syncthreads();
            everyone::tma::cluster::sync();
            if (compute::load_acquire_gpu(
                    g.worker_failed + worker_cluster) != 0u)
                return;
        }

        if (ticket == 0u && coordinate.stage == terminal::logical_stage::gate
                && cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int dispatched =
                compute::load_acquire_gpu(g.dispatch_tiles_done);
            compute::store_release_gpu(g.compute_started, 1u);
            if (dispatched
                    < static_cast<unsigned int>(
                        g.num_tokens[0] / terminal::M_TILE))
                compute::store_release_gpu(g.overlap_witness, 1u);
        }

        if (coordinate.stage == terminal::logical_stage::gate
                || coordinate.stage == terminal::logical_stage::up) {
            compute::run_w13_task(
                g.w13, coordinate, current_expert, cta_rank, g.ready,
                phasebits, ready_phase, a_smem, b_smem, d_smem,
                inputs_arrived, inputs_finished, inputs_ready);
        } else if (coordinate.stage
                       == terminal::logical_stage::activation) {
            compute::run_activation_task(
                g.activation, coordinate, cta_rank, g.ready);
        } else {
            compute::run_w2_task(
                g.w2, coordinate, current_expert, cta_rank, g.ready,
                phasebits, ready_phase, a_smem, b_smem, d_smem,
                inputs_arrived, inputs_finished, inputs_ready);
        }
    }
    everyone::tma::cluster::sync();
    reduce_ready_tokens(g);
}

template <typename GemmProblem>
__cluster_dims__(2, 1, 1)
__launch_bounds__(terminal::THREADS_PER_CTA, 1)
__global__ void kernel(const __grid_constant__ globals<GemmProblem> g) {
    const int cta_rank = cluster_ctarank();
    const int cluster = clusterIdx().x;
    if (cluster >= COMM_CLUSTERS + g.compute_clusters)
        return;

    extern __shared__ int __shm[];
    shared_allocator allocator((int *)&__shm[0]);
    auto &a_smem =
        allocator.allocate<compute::a_st, compute::PIPE_DEPTH>();
    auto &b_smem =
        allocator.allocate<compute::b_st, compute::PIPE_DEPTH>();
    compute::d_st &d_smem = allocator.allocate<compute::d_st>();
    __shared__ semaphore inputs_arrived[compute::PIPE_DEPTH];
    __shared__ semaphore inputs_finished[compute::PIPE_DEPTH];
    __shared__ semaphore inputs_ready[compute::PIPE_DEPTH];
    if (threadIdx.x < compute::PIPE_DEPTH) {
        init_semaphore(inputs_arrived[threadIdx.x], 0, 1);
        init_semaphore(inputs_finished[threadIdx.x], 0, 1);
        init_semaphore(inputs_ready[threadIdx.x], 0, 2);
    }
    everyone::tma::cluster::sync();

    if (cluster == COMM_CLUSTER) {
        communication_role(g, cta_rank);
        return;
    }
    compute_and_reduce_role(
        g, cta_rank, cluster - COMM_CLUSTERS,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);
}

#endif  // defined(KITTENS_SM90)

}  // namespace mok_sm90::fp8_block_terminal_full
