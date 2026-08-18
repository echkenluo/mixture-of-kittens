#pragma once

// Rank-local reduction milestone for the terminal SM90 FP8 forward.  The
// probe still supplies emulated peer pointers for remote pushes, but each
// invocation owns exactly one EP rank's combine/ready/claim/output domain.
// Production launches C dynamically assigned resident communication roles plus
// N compute roles, with C + N bounded by the measured active-cluster limit.
// Every compute CTA performs only one nonblocking ready-token probe at a task
// or wait boundary; NOT_READY work remains unclaimed, while a winning CTA
// reduces one local token and then resumes compute claims.  There is no
// grid/rank barrier, split path, or fallback in this kernel.

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

// Probe/default compatibility value.  Production supplies g.comm_clusters at
// runtime and keeps the total resident grid fixed while trading compute roles
// for communication roles.
constexpr int COMM_CLUSTERS = 1;
constexpr int COMM_CLUSTER = 0;
constexpr unsigned int UNCLAIMED_COMM = ~0u;
// DeepSeek-V4 has 256 routed experts globally.  Production currently targets
// EP4, so the common case is 64 local experts, but keeping the full model
// bound here preserves the workspace/entry contract and costs only 1 KiB of
// static shared memory for the expert row-end table.
constexpr int MAX_EXPERTS = 256;
constexpr unsigned int STOP_TICKET = ~0u;
constexpr unsigned int DONE_TICKET = STOP_TICKET - 1u;
constexpr unsigned int FAILED_TICKET = STOP_TICKET - 2u;
constexpr unsigned int WAIT_SIGNAL = STOP_TICKET - 3u;
constexpr unsigned int READY_EXPERT_BASE = 0u;

constexpr unsigned int OVERLAP_COMPUTE_DISPATCH = 1u << 0;
constexpr unsigned int OVERLAP_REDUCE_COMM = 1u << 1;
constexpr unsigned int OVERLAP_REDUCE_THEN_COMPUTE = 1u << 2;
constexpr unsigned int OVERLAP_DELAY_CLAIMED = 1u << 31;

// The communication decoder broadcasts only the state consumed after its
// warp-leader scope.  Keeping validity/closure beside the two-valued stage in
// one 32-bit word prevents the full coordinate from remaining live across row
// transport or owner-help compute.
constexpr unsigned int COMM_CONTROL_STAGE_MASK = 0xffu;
constexpr unsigned int COMM_CONTROL_VALID = 1u << 30;
constexpr unsigned int COMM_CONTROL_CLOSES_M64 = 1u << 31;

// Producer tasks outnumber final tokens by roughly six to one, and every
// bounded probe performs six system-scope acquire loads even when no incoming
// route is ready.  Keep opportunistic reduction in the compute loop, but let
// one CTA sample it once per eight logical tasks.  Cursor exhaustion and the
// communication-role drain remain exhaustive, so this cadence changes only
// overlap frequency, never completion or exactly-once ownership.
constexpr unsigned int REDUCE_TASK_PROBE_STRIDE = 8u;

// Production fatal record uses the same two-phase host-mapped protocol as
// K1: slot 0 is first claimed with ~0ull, slots 1..7 are populated, then the
// final non-zero code is release-published at system scope.  Only the winner
// traps; every loser parks without touching workspace memory again.
constexpr unsigned long long ERR_TIMEOUT = 1ull;
constexpr unsigned long long ERR_CONTRACT = 2ull;
constexpr unsigned long long TRAP_CLAIMED = ~0ull;
constexpr unsigned long long SITE_TERMINAL_CONTRACT = 20ull;
constexpr unsigned long long SITE_TERMINAL_INPUT_EXPECTED = 21ull;
constexpr unsigned long long SITE_TERMINAL_INPUT_BARRIER = 22ull;
constexpr unsigned long long SITE_TERMINAL_COMM_OWNER = 23ull;

#if defined(KITTENS_SM90)

using namespace kittens;

template <typename GemmProblem>
struct globals {
    GemmProblem w13;
    GemmProblem w2;

    __host__ globals(
            const GemmProblem &w13_problem,
            const GemmProblem &w2_problem)
        : w13(w13_problem), w2(w2_problem) {}

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
    int ep_rank;
    int num_local_tokens;
    int hidden_size;
    int scale_columns;
    int topk;
    int num_local_experts;
    int schedule_capacity;

    const uint8_t *routed_y;
    // Peer arrays are producer-only symmetric destinations.  Reduction must
    // never walk remote rank domains; it consumes these rank-local aliases.
    uint8_t *combine_peer[terminal::EP_SIZE];
    unsigned int *route_ready_peer[terminal::EP_SIZE];
    const uint8_t *combine_local;
    unsigned int *route_ready_local;
    const int *push_order;

    const float *weights;
    const int *topk_ids;
    __nv_bfloat16 *output;
    unsigned int *epilogue_claim;

    unsigned int *x_ready;
    unsigned int *cursor;
    unsigned int *worker_ticket;
    // Runtime role assignment is scheduling-order independent.  Each logical
    // compute/communication role owns one cluster-lockstep publication slot.
    // Probe launches leave role_cursor/cluster_role null and retain their
    // fixed physical role mapping.
    unsigned int *role_cursor = nullptr;
    unsigned int *cluster_role = nullptr;
    unsigned int *comm_worker_ticket = nullptr;
    // Legacy workspace name: this is the single dense cursor over the native
    // D(last), C(q)/D(q-1) communication sequence, not a dispatch-only queue.
    unsigned int *dispatch_tile_cursor = nullptr;
    // Number of completed combine communication tickets (debug/closure
    // receipt only; ownership comes exclusively from dispatch_tile_cursor).
    unsigned int *push_tile_cursor = nullptr;
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
    unsigned int *progress_timeouts;
    unsigned int *dispatch_tiles_done;
    unsigned int *compute_started;
    unsigned int *overlap_witness;

    // Production-only control plane.  Probe launches leave these null; the
    // numerical body remains shared while production owns the input barrier,
    // fatal record, and lease-release completion chain in this one kernel.
    unsigned int *barrier_flag = nullptr;
    unsigned int *barrier_target = nullptr;
    unsigned int *barrier_multicast_ptr = nullptr;
    unsigned int *input_expected_scratch = nullptr;
    unsigned int *in_use = nullptr;
    unsigned int *epilogue_done = nullptr;
    unsigned int *comm_owner = nullptr;
    unsigned int *producer_done = nullptr;
    unsigned int *push_done = nullptr;
    unsigned int *terminate = nullptr;
    unsigned long long *trap_record = nullptr;

    int comm_clusters = COMM_CLUSTERS;
    int compute_clusters;
    int minibatch_rows;
    int macrobatch_rows;
    unsigned int overlap_delay_after_first_dispatch_cycles;
    unsigned long long spin_limit;
};

__device__ __forceinline__ void park_forever() {
    while (true) __nanosleep(1u << 20);
}

template <typename GemmProblem>
__device__ __noinline__ void trap_commit(
        const globals<GemmProblem> &g, unsigned long long code,
        unsigned long long site, unsigned long long slot,
        unsigned long long expected, unsigned long long observed,
        unsigned long long ticket, unsigned long long iters) {
    const unsigned long long prior =
        atomicCAS(g.trap_record, 0ull, TRAP_CLAIMED);
    if (prior != 0ull)
        park_forever();
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

__device__ __forceinline__ unsigned int add_acq_rel_gpu(
        unsigned int *address, unsigned int value) {
    unsigned int old;
    asm volatile("{atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;}"
                 : "=r"(old)
                 : "l"(address), "r"(value)
                 : "memory");
    return old;
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

// A fixed physical cluster id is not a residency guarantee: under a
// concurrent context, a later grid cluster can be admitted before cluster 0.
// Physical clusters therefore claim dense immutable roles in admission order.
// The first C roles are communication roles and all later roles are compute
// roles.  No role waits for all peers to arrive, so the first resident roles
// can make progress even when another context temporarily limits residency.
template <typename GemmProblem>
__device__ int elect_runtime_role(
        const globals<GemmProblem> &g, int cluster, int cta_rank) {
    if (g.role_cursor == nullptr || g.cluster_role == nullptr)
        return cluster;  // fixed-role probe compatibility path

    if (cta_rank == 0 && threadIdx.x == 0) {
        const unsigned int role = atomicAdd(g.role_cursor, 1u);
        asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                     "l"(g.cluster_role + cluster), "r"(role) : "memory");
        if (role == 0u && g.comm_owner != nullptr)
            compute::store_release_gpu(
                g.comm_owner, static_cast<unsigned int>(cluster));
    }
    everyone::tma::cluster::sync();

    unsigned int role;
    asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                 : "=r"(role)
                 : "l"(g.cluster_role + cluster) : "memory");
    const unsigned int physical = static_cast<unsigned int>(
        g.comm_clusters + g.compute_clusters);
    if (role >= physical) {
        if (cta_rank == 0 && threadIdx.x == 0 && g.trap_record != nullptr)
            trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_COMM_OWNER,
                        cluster, physical, role, 0, 0);
        park_forever();
    }
    everyone::tma::cluster::sync();
    return static_cast<int>(role);
}

template <typename GemmProblem>
__device__ __forceinline__ bool production_control_enabled(
        const globals<GemmProblem> &g) {
    return g.terminate != nullptr;
}

// Four closure counters form the only production termination condition.
// Every observation is acquire; the successful CAS is release so all
// producer/push/reduce completion happens-before an acquire of terminate.
template <typename GemmProblem>
__device__ __forceinline__ void try_publish_terminate(
        const globals<GemmProblem> &g, unsigned int total_tasks,
        unsigned int active_rows, unsigned int total_tokens) {
    if (!production_control_enabled(g))
        return;
    const bool closed =
        compute::load_acquire_gpu(g.producer_done) >= total_tasks
        && compute::load_acquire_gpu(g.comm_closed)
            >= static_cast<unsigned int>(g.comm_clusters)
        && compute::load_acquire_gpu(g.push_done) >= active_rows
        && compute::load_acquire_gpu(g.reduce_done) >= total_tokens;
    if (!closed)
        return;
    unsigned int old;
    const unsigned int compare = 0u;
    const unsigned int value = 1u;
    // PTX CAS uses the bit type spelling; acq_rel is the accepted stronger
    // form of the required release publication on SM90.
    asm volatile("{atom.cas.acq_rel.gpu.global.b32 %0, [%1], %2, %3;}"
                 : "=r"(old)
                 : "l"(g.terminate), "r"(compare), "r"(value)
                 : "memory");
}

// One arrive per rank, performed by the elected resident communication owner
// before that same cluster can enter a wait.  All other resident clusters
// first wait for the rank-local expected value and then join the same bounded
// system-scope wait.  This absorbs the former standalone input barrier launch.
template <typename GemmProblem>
__device__ void production_input_barrier(
        const globals<GemmProblem> &g, int cluster, int cta_rank,
        int role) {
    if (g.barrier_flag == nullptr)
        return;
    if (role == 0 && cta_rank == 0 && threadIdx.x == 0) {
        const unsigned int expected =
            atomicAdd(g.barrier_target, static_cast<unsigned int>(g.ep_size))
            + static_cast<unsigned int>(g.ep_size);
        asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                     "l"(g.input_expected_scratch), "r"(expected)
                     : "memory");
        asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" ::
                     "l"(g.barrier_multicast_ptr) : "memory");
        asm volatile("{fence.proxy.alias;}" ::: "memory");
    }

    if (threadIdx.x == 0) {
        unsigned int expected = 0;
        unsigned long long iters = 0;
        while (expected == 0u) {
            asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                         : "=r"(expected)
                         : "l"(g.input_expected_scratch) : "memory");
            if (expected != 0u)
                break;
            __nanosleep(128);
            if (++iters >= g.spin_limit)
                trap_commit(g, ERR_TIMEOUT,
                            SITE_TERMINAL_INPUT_EXPECTED,
                            static_cast<unsigned long long>(cluster),
                            1, 0, 0, iters);
        }
        unsigned int observed = 0;
        iters = 0;
        while (observed < expected) {
            asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}"
                         : "=r"(observed)
                         : "l"(g.barrier_flag) : "memory");
            if (observed >= expected)
                break;
            __nanosleep(128);
            if (++iters >= g.spin_limit)
                trap_commit(g, ERR_TIMEOUT,
                            SITE_TERMINAL_INPUT_BARRIER,
                            static_cast<unsigned long long>(cluster),
                            expected, observed, 0, iters);
        }
        asm volatile("{fence.acquire.sys;}" ::: "memory");
    }
    everyone::tma::cluster::sync();
}

template <typename GemmProblem>
__device__ void production_completion_epilogue(
        const globals<GemmProblem> &g, int cta_rank) {
    if (g.in_use == nullptr)
        return;
    // All physical clusters take this path.  The CTA and cluster barriers
    // keep every writer ahead of its cluster leader's acq_rel completion
    // RMW.  The RMW release sequence transfers prior-cluster completion to
    // the last physical cluster, which releases the caller-owned lease.
    warpgroup::sync(0);
    __syncthreads();
    everyone::tma::cluster::sync();
    if (cta_rank == 0 && threadIdx.x == 0) {
        unsigned int old;
        asm volatile("{atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;}"
                     : "=r"(old) : "l"(g.epilogue_done) : "memory");
        const unsigned int physical = static_cast<unsigned int>(
            g.comm_clusters + g.compute_clusters);
        if (old + 1u == physical) {
            const unsigned int released = 0u;
            asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                         "l"(g.in_use), "r"(released) : "memory");
        }
    }
    everyone::tma::cluster::sync();
}

// Forward declaration: the communication owner uses the same bounded reducer
// after each pushed tile and while closing the production iteration.
template <typename GemmProblem>
__device__ route::claim_result try_reduce_one_ready_token(
        const globals<GemmProblem> &g);

// The numerical producer body is shared by ordinary workers and the elected
// communication owner.  The caller owns the persistent phase bits and the
// once-initialized mbarriers for its physical cluster.
template <typename GemmProblem>
__device__ __forceinline__ void run_producer_task_body(
        const globals<GemmProblem> &g,
        const terminal::logical_coordinate &coordinate,
        int current_expert, int cta_rank,
        uint32_t &phasebits, uint32_t &ready_phase,
        compute::a_st (&a_smem)[compute::PIPE_DEPTH],
        compute::b_st (&b_smem)[compute::PIPE_DEPTH],
        compute::d_st &d_smem,
        semaphore (&inputs_arrived)[compute::PIPE_DEPTH],
        semaphore (&inputs_finished)[compute::PIPE_DEPTH],
        semaphore (&inputs_ready)[compute::PIPE_DEPTH]) {
    if (coordinate.stage == terminal::logical_stage::gate
            || coordinate.stage == terminal::logical_stage::up) {
        compute::run_w13_task(
            g.w13, coordinate, current_expert, cta_rank, g.ready,
            phasebits, ready_phase, a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    } else if (coordinate.stage == terminal::logical_stage::activation) {
        compute::run_activation_task(
            g.activation, coordinate, cta_rank, g.ready);
    } else {
        compute::run_w2_task(
            g.w2, coordinate, current_expert, cta_rank, g.ready,
            phasebits, ready_phase, a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    }
}

template <typename GemmProblem>
__device__ __forceinline__ bool owner_task_ready(
        const globals<GemmProblem> &g,
        const terminal::logical_coordinate &coordinate) {
    if (coordinate.stage == terminal::logical_stage::gate
            || coordinate.stage == terminal::logical_stage::up) {
        return compute::load_acquire_gpu(
            g.x_ready + coordinate.global_m) >= terminal::M_TILE;
    }
    if (coordinate.stage == terminal::logical_stage::activation) {
        const int64_t base = terminal::counter_index(
            terminal::ready_counter::gate_up_tile,
            coordinate.global_m, 0);
#pragma unroll
        for (int n128 = 0; n128 < terminal::W13_N_TILES; ++n128) {
            if (compute::load_acquire_gpu(g.ready.gate_up_ready + base + n128)
                    < 2u)
                return false;
        }
        return true;
    }
    const int64_t hidden = terminal::counter_index(
        terminal::ready_counter::hidden_row_block, coordinate.global_m);
    return compute::load_acquire_gpu(g.ready.hidden_ready + hidden) >= 1u;
}

// Only advance the shared cursor when its current task is dependency-ready.
// In the sole-owner case cursor order is stage-topological (all gate, all up,
// activation, then W2 inside each ordered minibatch), so the owner can execute
// the whole chain serially.  With resident non-owners, a not-ready cursor means
// an earlier claimed task is still in flight; WAIT_SIGNAL leaves ownership with
// that worker and lets communication continue polling/reducing.
template <typename GemmProblem>
__device__ __forceinline__ unsigned int claim_ready_for_owner(
        const globals<GemmProblem> &g,
        const terminal::logical_shape &shape) {
    const unsigned int limit = static_cast<unsigned int>(shape.total_tasks);
    while (true) {
        const unsigned int current = compute::load_acquire_gpu(g.cursor);
        if (current >= limit)
            return STOP_TICKET;
        const terminal::logical_coordinate coordinate =
            terminal::decode_logical_cursor(shape, current);
        if (!coordinate.valid)
            return FAILED_TICKET;
        if (!owner_task_ready(g, coordinate))
            return WAIT_SIGNAL;
        const unsigned int prior = atomicCAS(
            g.cursor, current, current + 1u);
        if (prior == current)
            return current;
    }
}

// Claim and execute at most one producer task on the elected communication
// cluster.  The independent publication slot keeps both CTAs lockstep without
// colliding with any non-owner worker's ticket.  STOP_TICKET means the cursor
// is exhausted; it is not permission to push until y_ready says this tile is
// complete, because another resident cluster may still own an earlier task.
template <typename GemmProblem>
__device__ unsigned int owner_help_one_producer(
        const globals<GemmProblem> &g,
        const terminal::logical_shape &shape, int cta_rank,
        unsigned int *ticket_slot,
        uint32_t &phasebits, uint32_t &ready_phase,
        compute::a_st (&a_smem)[compute::PIPE_DEPTH],
        compute::b_st (&b_smem)[compute::PIPE_DEPTH],
        compute::d_st &d_smem,
        semaphore (&inputs_arrived)[compute::PIPE_DEPTH],
        semaphore (&inputs_finished)[compute::PIPE_DEPTH],
        semaphore (&inputs_ready)[compute::PIPE_DEPTH]) {
    const unsigned int total_tasks = static_cast<unsigned int>(
        shape.total_tasks);
    if (cta_rank == 0 && threadIdx.x == 0) {
        const unsigned int ticket = claim_ready_for_owner(g, shape);
        asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                     "l"(ticket_slot), "r"(ticket) : "memory");
    }
    everyone::tma::cluster::sync();
    unsigned int ticket;
    asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                 : "=r"(ticket)
                 : "l"(ticket_slot) : "memory");
    if (ticket == STOP_TICKET || ticket == WAIT_SIGNAL)
        return ticket;

    if (ticket == FAILED_TICKET) {
        if (cta_rank == 0 && threadIdx.x == 0)
            trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                        0, total_tasks,
                        compute::load_acquire_gpu(g.cursor), ticket, 0);
        park_forever();
    }

    const terminal::logical_coordinate coordinate =
        terminal::decode_logical_cursor(shape, ticket);
    if (!coordinate.valid) {
        if (cta_rank == 0 && threadIdx.x == 0)
            trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                        0, total_tasks, ticket, ticket, 0);
        park_forever();
    }
    if (cta_rank == 0 && threadIdx.x == 0 && g.task_visits != nullptr)
        atomicAdd(g.task_visits + ticket, 1u);

    int current_expert = -1;
    if (coordinate.stage != terminal::logical_stage::activation) {
        const unsigned int ready = compute::load_acquire_gpu(
            g.x_ready + coordinate.global_m);
        current_expert = g.m_indices[
            coordinate.global_m * terminal::M_TILE];
        if (ready < terminal::M_TILE || current_expert < 0
                || current_expert >= g.num_local_experts) {
            if (cta_rank == 0 && threadIdx.x == 0)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            coordinate.global_m, terminal::M_TILE, ready,
                            ticket, 0);
            park_forever();
        }
    }

    // The dispatch producer can be a different communication cluster.  The
    // x_ready release/acquire establishes visibility in the generic proxy;
    // bridge it into the async proxy before this role issues a W13 TMA load.
    if (cta_rank == 0 && threadIdx.x == 0
            && (coordinate.stage == terminal::logical_stage::gate
                || coordinate.stage == terminal::logical_stage::up))
        asm volatile("{fence.proxy.async.global;}" ::: "memory");
    everyone::tma::cluster::sync();

    run_producer_task_body(
        g, coordinate, current_expert, cta_rank, phasebits, ready_phase,
        a_smem, b_smem, d_smem,
        inputs_arrived, inputs_finished, inputs_ready);

    // run_* returns only after its WGMMA/TMA stores and ready publication are
    // drained.  Both CTAs cross the task boundary before producer completion.
    everyone::tma::cluster::sync();
    if (cta_rank == 0 && threadIdx.x == 0)
        compute::add_release_gpu(g.producer_done, 1u);
    everyone::tma::cluster::sync();
    return ticket;
}

template <typename GemmProblem>
__device__ void communication_role(
        const globals<GemmProblem> &g, int cta_rank, int comm_role,
        unsigned int *ticket_slot,
        compute::a_st (&a_smem)[compute::PIPE_DEPTH],
        compute::b_st (&b_smem)[compute::PIPE_DEPTH],
        compute::d_st &d_smem,
        semaphore (&inputs_arrived)[compute::PIPE_DEPTH],
        semaphore (&inputs_finished)[compute::PIPE_DEPTH],
        semaphore (&inputs_ready)[compute::PIPE_DEPTH]) {
    __shared__ int expert_row_end[MAX_EXPERTS];
    __shared__ int wait_ok;
    // logical_shape carries four int64 fields and otherwise immutable schedule
    // metadata.  Construct it once per CTA and retain only its shared address
    // across the resident loop instead of one materialized copy per thread.
    __shared__ terminal::logical_shape shape;
    // Only CTA-rank-0/thread-0 observes progress history.  Volatile shared
    // storage deliberately breaks these values' live ranges across the
    // owner-help WGMMA call; no other thread consumes them.
    __shared__ volatile unsigned long long leader_idle_windows;
    __shared__ volatile unsigned int leader_last_producer;
    __shared__ volatile unsigned int leader_last_push;
    __shared__ volatile unsigned int leader_last_reduce;
    __shared__ volatile unsigned int leader_last_dispatch;
    if (threadIdx.x == 0) {
        const int active_rows = comm::bounded_valid_rows(g);
        comm::build_expert_row_ends(g, expert_row_end);
        shape = terminal::make_logical_shape(
            active_rows, g.schedule_capacity,
            g.minibatch_rows, g.macrobatch_rows);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int WARPS_PER_CTA = terminal::THREADS_PER_CTA / 32;
    if (!shape.valid) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            if (g.trap_record != nullptr)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            0, g.schedule_capacity, g.num_tokens[0], 0, 0);
            if (g.errors != nullptr)
                atomicAdd(g.errors, 1u);
        }
        return;
    }

    // Preserve the native MoK communication timeline in one dense cursor.
    // Q=1 is D(0)->C(0).  For Q>=2, D(last) is followed by task-level
    // C(q),D(q-1) interleaving.  A claimed combine ticket may help its finite
    // producer prefix while the next dense ticket (normally D(q-1)) remains
    // available to another resident communication role.
    const bool owner_help_enabled = production_control_enabled(g);
    uint32_t owner_phasebits = 0xFFFF0000u;
    uint32_t owner_ready_phase = 0u;
    // The validated M64 contract makes the D+C cardinality exactly rows/4;
    // keep the resident hot path entirely 32-bit.
    const unsigned int total_comm_tickets =
        static_cast<unsigned int>(shape.num_tokens / 4);
    unsigned int probe_comm_ticket = 0u;
    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int ticket = g.dispatch_tile_cursor != nullptr
                ? claim_bounded(
                    g.dispatch_tile_cursor, total_comm_tickets)
                : (probe_comm_ticket < total_comm_tickets
                    ? probe_comm_ticket++ : STOP_TICKET);
            asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                         "l"(ticket_slot), "r"(ticket) : "memory");
        }
        everyone::tma::cluster::sync();
        unsigned int ticket;
        asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                     : "=r"(ticket) : "l"(ticket_slot) : "memory");
        if (ticket == STOP_TICKET)
            break;

        // The decoder is pure and warp-uniform.  One lane computes it, checks
        // the CTA extent, and broadcasts only two 32-bit scalars.  This removes
        // communication_coordinate from the transport and owner-help live
        // ranges without adding a CTA or cluster barrier.
        int first_row = -1;
        unsigned int comm_control = 0u;
        if (lane == 0) {
            const terminal::communication_coordinate coordinate =
                terminal::decode_communication_cursor(shape, ticket);
            const int macro_base =
                coordinate.macrobatch * shape.macrobatch_rows;
            const int decoded_first_row = macro_base
                + coordinate.round * terminal::COMM_ROWS_PER_TICKET
                + cta_rank * terminal::COMM_ROWS_PER_CTA_TASK;
            const int macro_end = macro_base
                + terminal::communication_rows(shape, coordinate.macrobatch);
            first_row = decoded_first_row;
            if (coordinate.valid && decoded_first_row >= 0
                    && decoded_first_row
                            + terminal::COMM_ROWS_PER_CTA_TASK <= macro_end) {
                comm_control = COMM_CONTROL_VALID
                    | static_cast<unsigned int>(coordinate.stage);
                if (coordinate.round
                            % terminal::COMM_TICKETS_PER_M_TILE
                        == terminal::COMM_TICKETS_PER_M_TILE - 1)
                    comm_control |= COMM_CONTROL_CLOSES_M64;
            }
        }
        first_row = __shfl_sync(0xffffffffu, first_row, 0);
        comm_control = __shfl_sync(0xffffffffu, comm_control, 0);
        if ((comm_control & COMM_CONTROL_VALID) == 0u) {
            if (cta_rank == 0 && threadIdx.x == 0 && g.trap_record != nullptr)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            comm_role, total_comm_tickets,
                            static_cast<unsigned long long>(first_row),
                            ticket, 0);
            park_forever();
        }

        const unsigned int comm_stage =
            comm_control & COMM_CONTROL_STAGE_MASK;
        if (comm_stage == static_cast<unsigned int>(
                terminal::communication_stage::dispatch)) {
            for (int local = warp;
                 local < terminal::COMM_ROWS_PER_CTA_TASK;
                 local += WARPS_PER_CTA) {
                const int row = first_row + local;
                comm::dispatch_copy_row(g, expert_row_end, row, lane);
                // Every lane wrote a stripe of the row.  The elected lane may
                // publish row readiness only after every writer has released
                // and converged.
                __threadfence();
                __syncwarp(0xffffffffu);
                if (lane == 0) {
                    const int m = row / terminal::M_TILE;
                    const unsigned int old = add_acq_rel_gpu(
                        g.x_ready + m, 1u);
                    if (old >= terminal::M_TILE) {
                        if (g.trap_record != nullptr)
                            trap_commit(
                                g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                                m, terminal::M_TILE, old, ticket, 0);
                        if (g.errors != nullptr)
                            atomicAdd(g.errors, 1u);
                    } else if (old + 1u == terminal::M_TILE
                            && g.dispatch_tiles_done != nullptr) {
                        compute::add_release_gpu(g.dispatch_tiles_done, 1u);
                    }
                    if (g.dispatch_visits != nullptr)
                        atomicAdd(g.dispatch_visits + row, 1u);
                }
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            // Test-only delay starts after at least one complete M64 has been
            // published, never after a partial row chunk.  Production is zero.
            if (cta_rank == 0 && threadIdx.x == 0
                    && shape.num_tokens > terminal::M_TILE
                    && g.dispatch_tiles_done != nullptr
                    && compute::load_acquire_gpu(g.dispatch_tiles_done) == 1u
                    && g.overlap_delay_after_first_dispatch_cycles != 0u) {
                const unsigned int prior = g.overlap_witness != nullptr
                    ? atomicOr(g.overlap_witness, OVERLAP_DELAY_CLAIMED)
                    : OVERLAP_DELAY_CLAIMED;
                if ((prior & OVERLAP_DELAY_CLAIMED) == 0u) {
                    unsigned int remaining =
                        g.overlap_delay_after_first_dispatch_cycles;
                    while (remaining != 0u) {
                        const unsigned int step = remaining > 1024u
                            ? 1024u : remaining;
                        __nanosleep(step);
                        remaining -= step;
                    }
                }
            }
            everyone::tma::cluster::sync();
            continue;
        }
        if (comm_stage != static_cast<unsigned int>(
                terminal::communication_stage::combine)) {
            if (cta_rank == 0 && threadIdx.x == 0 && g.trap_record != nullptr)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            comm_role,
                            static_cast<unsigned long long>(
                                terminal::communication_stage::combine),
                            static_cast<unsigned long long>(comm_stage),
                            ticket, 0);
            park_forever();
        }

        // COMM_ROWS_PER_TICKET divides M64 and every macrobatch starts on an
        // M64 boundary, so both CTA tasks in a combine ticket wait on exactly
        // one y_ready tile.
        const int m = first_row / terminal::M_TILE;
        if (first_row + terminal::COMM_ROWS_PER_CTA_TASK - 1
                    >= (m + 1) * terminal::M_TILE) {
            if (cta_rank == 0 && threadIdx.x == 0 && g.trap_record != nullptr)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            comm_role, terminal::M_TILE,
                            first_row, ticket, 0);
            park_forever();
        }
        if (owner_help_enabled) {
            if (cta_rank == 0 && threadIdx.x == 0) {
                leader_idle_windows = 0;
                leader_last_producer = compute::load_acquire_gpu(
                    g.producer_done);
            }
            while (true) {
                if (cta_rank == 0 && threadIdx.x == 0) {
                    const unsigned int ready = compute::load_acquire_gpu(
                        g.ready.y_ready + m);
                    const unsigned int decision =
                        ready >= terminal::W2_N_TILES
                            ? DONE_TICKET : WAIT_SIGNAL;
                    asm volatile(
                        "{st.release.cluster.global.u32 [%0], %1;}" ::
                        "l"(ticket_slot), "r"(decision)
                        : "memory");
                }
                everyone::tma::cluster::sync();
                unsigned int decision;
                asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                             : "=r"(decision)
                             : "l"(ticket_slot) : "memory");
                if (decision == DONE_TICKET)
                    break;

                owner_help_one_producer(
                    g, shape, cta_rank, ticket_slot,
                    owner_phasebits, owner_ready_phase,
                    a_smem, b_smem, d_smem,
                    inputs_arrived, inputs_finished, inputs_ready);

                // A prior tile may already have been pushed.  This one-shot
                // reducer never waits for the current tile and therefore does
                // not introduce a producer dependency.
                try_reduce_one_ready_token(g);
                everyone::tma::cluster::sync();

                if (cta_rank == 0 && threadIdx.x == 0) {
                    const unsigned int ready = compute::load_acquire_gpu(
                        g.ready.y_ready + m);
                    const unsigned int produced = compute::load_acquire_gpu(
                        g.producer_done);
                    unsigned int next = WAIT_SIGNAL;
                    if (ready >= terminal::W2_N_TILES) {
                        next = DONE_TICKET;
                    } else {
                        if (produced != leader_last_producer) {
                            leader_last_producer = produced;
                            leader_idle_windows = 0;
                        } else {
                            leader_idle_windows = leader_idle_windows + 1ull;
                        }
                        if (leader_idle_windows >= g.spin_limit)
                            trap_commit(
                                g, ERR_TIMEOUT, SITE_TERMINAL_CONTRACT,
                                m, terminal::W2_N_TILES, ready,
                                compute::load_acquire_gpu(g.cursor),
                                leader_idle_windows);
                    }
                    asm volatile(
                        "{st.release.cluster.global.u32 [%0], %1;}" ::
                        "l"(ticket_slot), "r"(next) : "memory");
                }
                everyone::tma::cluster::sync();
                asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                             : "=r"(decision)
                             : "l"(ticket_slot) : "memory");
                if (decision == DONE_TICKET)
                    break;
                if (threadIdx.x == 0)
                    __nanosleep(64);
                __syncthreads();
            }
        } else {
            if (threadIdx.x == 0) {
                wait_ok = bounded_wait_gpu(
                    g.ready.y_ready + m, terminal::W2_N_TILES,
                    g.spin_limit) ? 1 : 0;
            }
            __syncthreads();
            if (!wait_ok && threadIdx.x == 0) {
                if (g.trap_record != nullptr)
                    trap_commit(g, ERR_TIMEOUT, SITE_TERMINAL_CONTRACT,
                                m, terminal::W2_N_TILES, 0, 0,
                                g.spin_limit);
                if (g.comm_failed != nullptr)
                    atomicExch(g.comm_failed, 1u);
                if (g.progress_timeouts != nullptr)
                    atomicAdd(g.progress_timeouts, 1u);
                if (g.errors != nullptr)
                    atomicAdd(g.errors, 1u);
            }
            everyone::tma::cluster::sync();
            if (g.comm_failed != nullptr
                    && compute::load_acquire_gpu(g.comm_failed) != 0u)
                return;
        }

        for (int local = warp;
             local < terminal::COMM_ROWS_PER_CTA_TASK;
             local += WARPS_PER_CTA) {
            const int position = first_row + local;
            const int first = m * terminal::M_TILE;
            const int row = g.push_order != nullptr
                ? g.push_order[position]
                : position;
            const bool in_tile = row >= first
                && row < first + terminal::M_TILE;
            if (in_tile) {
                route::push_routed_row_and_publish(g, row, lane);
                __syncwarp(0xffffffffu);
                if (lane == 0) {
                    if (g.push_done != nullptr)
                        compute::add_release_gpu(g.push_done, 1u);
                    if (g.push_visits != nullptr)
                        atomicAdd(g.push_visits + row, 1u);
                }
            } else if (lane == 0 && g.errors != nullptr) {
                atomicAdd(g.errors, 1u);
            }
        }
        __syncthreads();
        everyone::tma::cluster::sync();
        if (cta_rank == 0 && threadIdx.x == 0
                && g.push_tile_cursor != nullptr)
            compute::add_release_gpu(g.push_tile_cursor, 1u);
        everyone::tma::cluster::sync();
        const bool closes_m64 =
            (comm_control & COMM_CONTROL_CLOSES_M64) != 0u;
        if (owner_help_enabled && closes_m64) {
            // Preserve the old once-per-M64 reducer cadence while the comm
            // cursor itself operates at native four-row CTA granularity.
            try_reduce_one_ready_token(g);
            everyone::tma::cluster::sync();
        }
    }

    if (cta_rank == 0 && threadIdx.x == 0)
        compute::add_release_gpu(g.comm_closed, 1u);
    everyone::tma::cluster::sync();

    if (production_control_enabled(g)) {
        const unsigned int total_tasks = static_cast<unsigned int>(
            shape.total_tasks);
        const unsigned int active_row_count = static_cast<unsigned int>(
            shape.num_tokens);
        const unsigned int total_tokens = static_cast<unsigned int>(
            g.num_local_tokens);
        if (cta_rank == 0 && threadIdx.x == 0) {
            leader_idle_windows = 0;
            leader_last_producer = 0u;
            leader_last_push = 0u;
            leader_last_reduce = 0u;
            leader_last_dispatch = 0u;
        }
        while (true) {
            // A role that exhausted the communication queues remains a useful
            // resident worker.  This is the progress edge for C-only residency.
            owner_help_one_producer(
                g, shape, cta_rank, ticket_slot,
                owner_phasebits, owner_ready_phase,
                a_smem, b_smem, d_smem,
                inputs_arrived, inputs_finished, inputs_ready);
            try_reduce_one_ready_token(g);
            everyone::tma::cluster::sync();
            if (cta_rank == 0 && threadIdx.x == 0) {
                try_publish_terminate(
                    g, total_tasks, active_row_count, total_tokens);
                const unsigned int producer =
                    compute::load_acquire_gpu(g.producer_done);
                const unsigned int pushed =
                    compute::load_acquire_gpu(g.push_done);
                const unsigned int reduced =
                    compute::load_acquire_gpu(g.reduce_done);
                const unsigned int dispatched = g.dispatch_tiles_done != nullptr
                    ? compute::load_acquire_gpu(g.dispatch_tiles_done) : 0u;
                unsigned int decision = WAIT_SIGNAL;
                if (compute::load_acquire_gpu(g.terminate) == 0u) {
                    if (producer != leader_last_producer
                            || pushed != leader_last_push
                            || reduced != leader_last_reduce
                            || dispatched != leader_last_dispatch) {
                        leader_last_producer = producer;
                        leader_last_push = pushed;
                        leader_last_reduce = reduced;
                        leader_last_dispatch = dispatched;
                        leader_idle_windows = 0;
                    } else {
                        leader_idle_windows = leader_idle_windows + 1ull;
                    }
                    if (leader_idle_windows >= g.spin_limit)
                        trap_commit(
                            g, ERR_TIMEOUT, SITE_TERMINAL_CONTRACT,
                            0, total_tasks, producer, 0,
                            leader_idle_windows);
                } else
                    decision = DONE_TICKET;
                asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                             "l"(ticket_slot), "r"(decision) : "memory");
            }
            everyone::tma::cluster::sync();
            unsigned int decision;
            asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                         : "=r"(decision) : "l"(ticket_slot) : "memory");
            if (decision == DONE_TICKET)
                return;
            if (threadIdx.x == 0)
                __nanosleep(64);
            __syncthreads();
        }
    }

    // A probe communication cluster never changes roles or leaves early.
    // Once every producer push is closed it remains resident until the
    // opportunistic compute CTAs have reduced every token.  This wait cannot
    // exclude a producer: all dispatch and push work is already complete.
    const unsigned int total_tokens = static_cast<unsigned int>(
        g.num_local_tokens);
    if (threadIdx.x == 0) {
        wait_ok = bounded_wait_gpu(
            g.reduce_done, total_tokens, g.spin_limit) ? 1 : 0;
    }
    __syncthreads();
    if (!wait_ok && threadIdx.x == 0) {
        if (g.trap_record != nullptr)
            trap_commit(g, ERR_TIMEOUT, SITE_TERMINAL_CONTRACT,
                        0, total_tokens,
                        compute::load_acquire_gpu(g.reduce_done), 0,
                        g.spin_limit);
        if (g.comm_failed != nullptr)
            atomicExch(g.comm_failed, 1u);
        if (g.progress_timeouts != nullptr)
            atomicAdd(g.progress_timeouts, 1u);
        if (g.errors != nullptr)
            atomicAdd(g.errors, 1u);
    }
    everyone::tma::cluster::sync();
}

// One CTA probes exactly one rank-local round-robin token and returns
// immediately for
// NOT_READY, ALREADY_CLAIMED, or an invalid/empty domain.  Only a CTA that won
// the ready-token CAS enters the column-parallel reduction.  This bounded
// primitive is safe to call between producer tasks: it never waits for a
// route whose compute/push producer is still outstanding.
template <typename GemmProblem>
__device__ route::claim_result try_reduce_one_ready_token(
        const globals<GemmProblem> &g) {
    __shared__ int selected_token;
    __shared__ int selected_result;
    const int total_tokens = g.num_local_tokens;

    if (threadIdx.x == 0) {
        const unsigned int probe = atomicAdd(g.next_reduce_probe, 1u);
        const int local_token = total_tokens == 0
            ? -1
            : static_cast<int>(probe % total_tokens);
        selected_token = local_token;
        selected_result = local_token < 0
            ? static_cast<int>(route::claim_result::invalid_token)
            : static_cast<int>(route::try_claim_ready_token(
                g.route_ready_local, g.epilogue_claim,
                selected_token, total_tokens));
    }
    __syncthreads();

    if (selected_result
            == static_cast<int>(route::claim_result::claimed)) {
        const auto *combine = reinterpret_cast<const __nv_bfloat16 *>(
            g.combine_local);
        route::reduce_claimed_token(
            combine, g.weights, g.topk_ids, g.output,
            selected_token, g.hidden_size,
            threadIdx.x, blockDim.x);
        route::release_fence_system();
    }
    __syncthreads();

    if (threadIdx.x == 0
            && selected_result
                == static_cast<int>(route::claim_result::claimed)) {
        if (g.reduce_visits != nullptr)
            atomicAdd(g.reduce_visits + selected_token, 1u);
        atomicAdd(g.reduce_done, 1u);
        if (g.overlap_witness != nullptr
                && compute::load_acquire_gpu(g.comm_closed)
                    < static_cast<unsigned int>(g.comm_clusters))
            atomicOr(g.overlap_witness, OVERLAP_REDUCE_COMM);
    }
    __syncthreads();
    return static_cast<route::claim_result>(selected_result);
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
    // All threads in both CTAs decode the same immutable schedule shape.  Keep
    // one copy per CTA rather than four int64 fields plus metadata live in
    // every producer thread across WGMMA/TMA task bodies.
    __shared__ terminal::logical_shape shape;
    // These values are consumed only by CTA thread 0.  Shared placement keeps
    // timeout/debug history out of the resident producer's register live set.
    __shared__ volatile unsigned long long leader_wait_windows;
    __shared__ volatile unsigned int leader_last_reduce_done;
    __shared__ volatile unsigned int leader_last_comm_closed;
    __shared__ volatile unsigned int leader_last_producer_done;
    __shared__ volatile unsigned int leader_last_push_done;
    __shared__ volatile unsigned int reduced_before_compute;
    if (threadIdx.x == 0) {
        shape = terminal::make_logical_shape(
            g.num_tokens[0], g.schedule_capacity,
            g.minibatch_rows, g.macrobatch_rows);
        leader_wait_windows = 0;
        leader_last_reduce_done = 0u;
        leader_last_comm_closed = 0u;
        leader_last_producer_done = 0u;
        leader_last_push_done = 0u;
        reduced_before_compute = 0u;
    }
    __syncthreads();

    uint32_t phasebits = 0xFFFF0000u;
    uint32_t ready_phase = 0u;
    if (!shape.valid) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            if (g.trap_record != nullptr)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            worker_cluster, g.schedule_capacity,
                            g.num_tokens[0], 0, 0);
            if (g.errors != nullptr)
                atomicAdd(g.errors, 1u);
        }
        return;
    }
    const unsigned int total_tasks =
        static_cast<unsigned int>(shape.total_tasks);
    const unsigned int total_tokens = static_cast<unsigned int>(
        g.num_local_tokens);
    const unsigned int active_rows = static_cast<unsigned int>(
        shape.num_tokens);

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int ticket = claim_bounded(g.cursor, total_tasks);
            asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                         "l"(g.worker_ticket + worker_cluster), "r"(ticket)
                         : "memory");
        }
        everyone::tma::cluster::sync();
        unsigned int ticket;
        asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                     : "=r"(ticket)
                     : "l"(g.worker_ticket + worker_cluster) : "memory");
        if (ticket == STOP_TICKET) {
            // Cursor exhaustion is not worker termination: outstanding
            // compute/comm producers may make another token ready.  Each CTA
            // performs one bounded probe, then the cluster leader decides
            // whether global progress closed or another poll window is due.
            const route::claim_result result =
                try_reduce_one_ready_token(g);
            if (threadIdx.x == 0
                    && result == route::claim_result::claimed)
                reduced_before_compute = 1u;
            everyone::tma::cluster::sync();

            if (cta_rank == 0 && threadIdx.x == 0) {
                const unsigned int reduced =
                    compute::load_acquire_gpu(g.reduce_done);
                const unsigned int closed =
                    compute::load_acquire_gpu(g.comm_closed);
                unsigned int decision = STOP_TICKET;
                if (production_control_enabled(g)) {
                    try_publish_terminate(
                        g, total_tasks, active_rows, total_tokens);
                    if (compute::load_acquire_gpu(g.terminate) != 0u)
                        decision = DONE_TICKET;
                } else if (reduced >= total_tokens && closed >= 1u) {
                    decision = DONE_TICKET;
                }
                if (decision == STOP_TICKET) {
                    const unsigned int produced =
                        g.producer_done != nullptr
                            ? compute::load_acquire_gpu(g.producer_done)
                            : 0u;
                    const unsigned int pushed = g.push_done != nullptr
                        ? compute::load_acquire_gpu(g.push_done)
                        : 0u;
                    if (reduced != leader_last_reduce_done
                            || closed != leader_last_comm_closed
                            || produced != leader_last_producer_done
                            || pushed != leader_last_push_done) {
                        leader_wait_windows = 0;
                        leader_last_reduce_done = reduced;
                        leader_last_comm_closed = closed;
                        leader_last_producer_done = produced;
                        leader_last_push_done = pushed;
                    } else {
                        leader_wait_windows = leader_wait_windows + 1ull;
                    }
                    if (leader_wait_windows >= g.spin_limit) {
                        if (g.trap_record != nullptr)
                            trap_commit(
                                g, ERR_TIMEOUT, SITE_TERMINAL_CONTRACT,
                                worker_cluster, total_tokens, reduced,
                                STOP_TICKET, leader_wait_windows);
                        if (g.worker_failed != nullptr)
                            atomicExch(
                                g.worker_failed + worker_cluster, 1u);
                        if (g.progress_timeouts != nullptr)
                            atomicAdd(g.progress_timeouts, 1u);
                        if (g.errors != nullptr)
                            atomicAdd(g.errors, 1u);
                        decision = FAILED_TICKET;
                    }
                }
                asm volatile(
                    "{st.release.cluster.global.u32 [%0], %1;}" ::
                    "l"(g.worker_ticket + worker_cluster), "r"(decision)
                    : "memory");
            }
            everyone::tma::cluster::sync();
            unsigned int decision;
            asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                         : "=r"(decision)
                         : "l"(g.worker_ticket + worker_cluster) : "memory");
            if (decision == DONE_TICKET || decision == FAILED_TICKET)
                return;
            if (threadIdx.x == 0)
                __nanosleep(64);
            __syncthreads();
            continue;
        }

        const terminal::logical_coordinate coordinate =
            terminal::decode_logical_cursor(shape, ticket);
        if (!coordinate.valid) {
            if (cta_rank == 0 && threadIdx.x == 0) {
                if (g.trap_record != nullptr)
                    trap_commit(
                        g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                        worker_cluster, total_tasks, ticket, ticket, 0);
                if (g.errors != nullptr)
                    atomicAdd(g.errors, 1u);
            }
            everyone::tma::cluster::sync();
            continue;
        }
        if (cta_rank == 0 && threadIdx.x == 0
                && g.task_visits != nullptr)
            atomicAdd(g.task_visits + ticket, 1u);

        int current_expert = -1;
        if (coordinate.stage != terminal::logical_stage::activation) {
            if (cta_rank == 0 && threadIdx.x == 0)
                leader_wait_windows = 0;
            while (true) {
                if (cta_rank == 0 && threadIdx.x == 0) {
                    unsigned int signal = WAIT_SIGNAL;
                    if (compute::load_acquire_gpu(
                            g.x_ready + coordinate.global_m)
                            >= terminal::M_TILE) {
                        const int expert = g.m_indices[
                            coordinate.global_m * terminal::M_TILE];
                        if (expert >= 0 && expert < g.num_local_experts)
                            signal = READY_EXPERT_BASE
                                + static_cast<unsigned int>(expert);
                        else
                            signal = FAILED_TICKET;
                    }
                    asm volatile(
                        "{st.release.cluster.global.u32 [%0], %1;}" ::
                        "l"(g.worker_ticket + worker_cluster), "r"(signal)
                        : "memory");
                }
                everyone::tma::cluster::sync();
                unsigned int signal;
                asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                             : "=r"(signal)
                             : "l"(g.worker_ticket + worker_cluster)
                             : "memory");
                if (signal != WAIT_SIGNAL) {
                    if (signal == FAILED_TICKET) {
                        if (cta_rank == 0 && threadIdx.x == 0) {
                            if (g.trap_record != nullptr)
                                trap_commit(
                                    g, ERR_CONTRACT,
                                    SITE_TERMINAL_CONTRACT,
                                    coordinate.global_m,
                                    g.num_local_experts, current_expert,
                                    ticket, 0);
                            if (g.worker_failed != nullptr)
                                atomicExch(
                                    g.worker_failed + worker_cluster, 1u);
                            if (g.errors != nullptr)
                                atomicAdd(g.errors, 1u);
                        }
                        return;
                    }
                    current_expert = static_cast<int>(
                        signal - READY_EXPERT_BASE);
                    break;
                }

                // A waiting task keeps ownership, but its CTA may help one
                // already-ready epilogue token.  NOT_READY returns directly;
                // no producer dependency is waited on by the reducer.
                const route::claim_result result =
                    try_reduce_one_ready_token(g);
                if (threadIdx.x == 0
                        && result == route::claim_result::claimed)
                    reduced_before_compute = 1u;
                everyone::tma::cluster::sync();
                if (cta_rank == 0 && threadIdx.x == 0) {
                    leader_wait_windows = leader_wait_windows + 1ull;
                    unsigned int decision = WAIT_SIGNAL;
                    if (leader_wait_windows >= g.spin_limit) {
                        if (g.trap_record != nullptr)
                            trap_commit(
                                g, ERR_TIMEOUT,
                                SITE_TERMINAL_CONTRACT,
                                coordinate.global_m, terminal::M_TILE,
                                compute::load_acquire_gpu(
                                    g.x_ready + coordinate.global_m),
                                ticket, leader_wait_windows);
                        if (g.worker_failed != nullptr)
                            atomicExch(
                                g.worker_failed + worker_cluster, 1u);
                        if (g.progress_timeouts != nullptr)
                            atomicAdd(g.progress_timeouts, 1u);
                        if (g.errors != nullptr)
                            atomicAdd(g.errors, 1u);
                        decision = FAILED_TICKET;
                    }
                    asm volatile(
                        "{st.release.cluster.global.u32 [%0], %1;}" ::
                        "l"(g.worker_ticket + worker_cluster), "r"(decision)
                        : "memory");
                }
                everyone::tma::cluster::sync();
                asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                             : "=r"(signal)
                             : "l"(g.worker_ticket + worker_cluster)
                             : "memory");
                if (signal == FAILED_TICKET)
                    return;
                if (threadIdx.x == 0)
                    __nanosleep(64);
                __syncthreads();
            }

            // dispatch_copy_row uses generic stores while run_w13_task's
            // rank-0 TMA consumes through the async proxy.
            if (cta_rank == 0 && threadIdx.x == 0
                    && (coordinate.stage == terminal::logical_stage::gate
                        || coordinate.stage == terminal::logical_stage::up))
                asm volatile("{fence.proxy.async.global;}" ::: "memory");
            everyone::tma::cluster::sync();
        }

        if (ticket == 0u && coordinate.stage == terminal::logical_stage::gate
                && cta_rank == 0 && threadIdx.x == 0
                && g.dispatch_tiles_done != nullptr
                && g.compute_started != nullptr
                && g.overlap_witness != nullptr) {
            const unsigned int dispatched =
                compute::load_acquire_gpu(g.dispatch_tiles_done);
            compute::store_release_gpu(g.compute_started, 1u);
            if (dispatched
                    < static_cast<unsigned int>(
                        g.num_tokens[0] / terminal::M_TILE))
                atomicOr(g.overlap_witness, OVERLAP_COMPUTE_DISPATCH);
        }

        if (reduced_before_compute != 0u && threadIdx.x == 0
                && g.overlap_witness != nullptr)
            atomicOr(g.overlap_witness, OVERLAP_REDUCE_THEN_COMPUTE);

        run_producer_task_body(
            g, coordinate, current_expert, cta_rank, phasebits, ready_phase,
            a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);

        everyone::tma::cluster::sync();
        if (cta_rank == 0 && threadIdx.x == 0
                && g.producer_done != nullptr)
            compute::add_release_gpu(g.producer_done, 1u);
        everyone::tma::cluster::sync();

        // Sampling every task from both CTAs produced tens of thousands of
        // guaranteed-empty system-scope probes before combine publication.
        // One CTA retains bounded overlap at a coarse cadence; the existing
        // STOP/communication drain paths still reduce every remaining token.
        if (cta_rank == 0
                && ticket % REDUCE_TASK_PROBE_STRIDE == 0u) {
            const route::claim_result result = try_reduce_one_ready_token(g);
            if (result == route::claim_result::claimed)
                reduced_by_this_cta = true;
        }
    }
}

template <typename GemmProblem>
__cluster_dims__(2, 1, 1)
__launch_bounds__(terminal::THREADS_PER_CTA, 1)
__global__ void kernel(const __grid_constant__ globals<GemmProblem> g) {
    const int cta_rank = cluster_ctarank();
    const int cluster = clusterIdx().x;
    if (cluster >= g.comm_clusters + g.compute_clusters)
        return;

    // Dynamic graph-replay dimensions cannot be validated by the host
    // without synchronizing.  Production therefore checks them in device
    // code before any route/combine payload is touched.
    const int active_tokens = g.num_tokens[0];
    if (g.trap_record != nullptr
            && (active_tokens < 0
                || active_tokens > g.schedule_capacity
                || (active_tokens % terminal::M_TILE) != 0)) {
        if (threadIdx.x == 0)
            trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                        cluster, g.schedule_capacity, active_tokens, 0, 0);
        park_forever();
    }
    if (g.trap_record != nullptr && g.in_use != nullptr) {
        unsigned int lease;
        asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                     : "=r"(lease) : "l"(g.in_use) : "memory");
        if (lease != 1u) {
            if (threadIdx.x == 0)
                trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_CONTRACT,
                            cluster, 1, lease, 0, 0);
            park_forever();
        }
    }
    if (g.trap_record != nullptr && production_control_enabled(g)
            && (g.comm_owner == nullptr || g.role_cursor == nullptr
                || g.cluster_role == nullptr
                || g.comm_worker_ticket == nullptr
                || g.dispatch_tile_cursor == nullptr
                || g.dispatch_tiles_done == nullptr
                || g.push_tile_cursor == nullptr)) {
        if (threadIdx.x == 0)
            trap_commit(g, ERR_CONTRACT, SITE_TERMINAL_COMM_OWNER,
                        cluster, 7,
                        static_cast<unsigned long long>(
                            (g.comm_owner != nullptr ? 1u : 0u)
                            + (g.role_cursor != nullptr ? 1u : 0u)
                            + (g.cluster_role != nullptr ? 1u : 0u)
                            + (g.comm_worker_ticket != nullptr ? 1u : 0u)
                            + (g.dispatch_tile_cursor != nullptr ? 1u : 0u)
                            + (g.dispatch_tiles_done != nullptr ? 1u : 0u)
                            + (g.push_tile_cursor != nullptr ? 1u : 0u)),
                        0, 0);
        park_forever();
    }

    const int role = elect_runtime_role(g, cluster, cta_rank);
    production_input_barrier(g, cluster, cta_rank, role);

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
        // mbarrier.init is a generic-proxy write.  Publish every slot to the
        // async proxy before any later TMA load names the barrier.
        asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
    }
    everyone::tma::cluster::sync();

    if (role < g.comm_clusters) {
        communication_role(
            g, cta_rank, role, g.comm_worker_ticket + role,
            a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    } else {
        const int worker_cluster = role - g.comm_clusters;
        compute_and_reduce_role(
            g, cta_rank, worker_cluster,
            a_smem, b_smem, d_smem,
            inputs_arrived, inputs_finished, inputs_ready);
    }
    production_completion_epilogue(g, cta_rank);
}

#endif  // defined(KITTENS_SM90)

}  // namespace mok_sm90::fp8_block_terminal_full
