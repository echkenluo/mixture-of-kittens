#pragma once

// Terminal per-route publication and ready-token consumption primitives.
//
// This header deliberately owns no cursor, queue, blocking wait, rank-wide
// barrier, or termination policy.  A fixed communication warp publishes one
// route only after every lane that wrote the BF16 row has made its stores
// system-visible.  A destination helper probes all TOPK flags once, claims a
// fully-ready token exactly once, and leaves NOT_READY work unowned.
//
// Invalid producer routes are legal no-ops: they write neither payload nor a
// flag.  Prepare must initialize the corresponding destination flag to READY
// and set topk_ids[token,slot] to a negative value.  The reducer still acquire
// loads that flag, but never reads the invalid combine row.  An all-invalid
// padded token is reduced to BF16 +0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sm90_fp8_block_pipeline_primitives.cuh"
#include "sm90_fp8_block_terminal_comm_primitives.cuh"

namespace mok_sm90::fp8_block_terminal_route_flags {

namespace comm = fp8_block_terminal_comm;
namespace pipeline = fp8_block_pipeline;

constexpr int TOPK = 6;
constexpr unsigned int ROUTE_NOT_READY = 0u;
constexpr unsigned int ROUTE_READY = 1u;
constexpr unsigned int CLAIM_FREE = 0u;
constexpr unsigned int CLAIM_OWNED = 1u;

enum class claim_result : int {
    invalid_token = -1,
    not_ready = 0,
    claimed = 1,
    already_claimed = 2,
};

// These instructions are intentionally spelled out.  Peer payload visibility
// is a system-scope contract, not a device-only ordering accident.
__device__ __forceinline__ void release_fence_system() {
    asm volatile("{fence.release.sys;}" ::: "memory");
}

__device__ __forceinline__ void store_release_system(
    unsigned int *address, unsigned int value) {
    asm volatile("{st.release.sys.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int load_acquire_system(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.sys.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

// A full warp owns one row.  push_routed_row performs the lane-striped BF16
// stores and returns the same route in every lane.  Every writer then executes
// its own system release fence; only after warp convergence may lane zero
// publish the peer-owned flattened route_ready[token,slot] flag.
//
// Globals must provide route_ready_peer[ep_size] in addition to the fields
// consumed by terminal_comm::push_routed_row.  All 32 lanes must call this
// helper together with lane == threadIdx.x % 32.
template <typename Globals>
__device__ __forceinline__ comm::route_mapping push_routed_row_and_publish(
    const Globals &g, int row, int lane) {
    const comm::route_mapping route = comm::push_routed_row(g, row, lane);
    if (!route.valid)
        return route;

    release_fence_system();
    __syncwarp(0xffffffffu);
    if (lane == 0) {
        unsigned int *peer_flag = g.route_ready_peer[route.peer_rank]
            + route.peer_token_idx;
        store_release_system(peer_flag, ROUTE_READY);
    }
    return route;
}

// Probe every one of the fixed top-6 flags.  There is deliberately no early
// return: even invalid/pre-ready slots participate in the acquire protocol.
// The helper performs one bounded pass and never spins.
__device__ __forceinline__ bool all_routes_ready_once(
    const unsigned int *route_ready, int token) {
    const size_t route_base = static_cast<size_t>(token) * TOPK;
    bool all_ready = true;
#pragma unroll
    for (int route = 0; route < TOPK; ++route) {
        const unsigned int observed =
            load_acquire_system(route_ready + route_base + route);
        all_ready = all_ready && observed == ROUTE_READY;
    }
    return all_ready;
}

// Leader-only scalar primitive.  A NOT_READY token remains unclaimed so the
// caller can immediately continue producer work or scan another token.  The
// CAS is reached only after all six system-acquire loads observed READY.  An
// empty token domain (token_count == 0), or any out-of-range token, returns
// invalid_token without touching flags or claim storage.
__device__ __forceinline__ claim_result try_claim_ready_token(
    const unsigned int *route_ready, unsigned int *epilogue_claim,
    int token, int token_count) {
    if (token < 0 || token >= token_count)
        return claim_result::invalid_token;
    if (!all_routes_ready_once(route_ready, token))
        return claim_result::not_ready;
    const unsigned int prior =
        atomicCAS(epilogue_claim + token, CLAIM_FREE, CLAIM_OWNED);
    return prior == CLAIM_FREE
        ? claim_result::claimed
        : claim_result::already_claimed;
}

// Reduce one output element after the caller has won the token claim.  Valid
// route values are compacted in slot order, so invalid storage is never read
// and the surviving routes retain the production mul-then-FMA order.  The
// shared weighted_reduce_element owns the arithmetic and final BF16 rounding.
__device__ __forceinline__ void weighted_reduce_valid_element(
    const __nv_bfloat16 *combine, const float *weights,
    const int *topk_ids, __nv_bfloat16 *output,
    int token, int column, int hidden) {
    const size_t route_base = static_cast<size_t>(token) * TOPK;
    __nv_bfloat16 compact_combine[TOPK];
    float compact_weights[TOPK];
    int valid_routes = 0;
#pragma unroll
    for (int route = 0; route < TOPK; ++route) {
        const size_t route_index = route_base + route;
        if (topk_ids[route_index] >= 0) {
            compact_combine[valid_routes] =
                combine[route_index * hidden + column];
            compact_weights[valid_routes] = weights[route_index];
            ++valid_routes;
        }
    }
    if (valid_routes == 0) {
        output[static_cast<size_t>(token) * hidden + column] =
            __float2bfloat16_rn(0.0f);
        return;
    }

    __nv_bfloat16 reduced;
    pipeline::weighted_reduce_element(
        compact_combine, compact_weights, &reduced,
        0, 0, valid_routes, 1);
    output[static_cast<size_t>(token) * hidden + column] = reduced;
}

// Column-parallel convenience core.  Synchronization and ownership broadcast
// remain with the resident caller because a warp, CTA, or cluster may supply
// the workers.  This helper contains no barrier and assumes the token is owned.
__device__ __forceinline__ void reduce_claimed_token(
    const __nv_bfloat16 *combine, const float *weights,
    const int *topk_ids, __nv_bfloat16 *output,
    int token, int hidden, int worker, int workers) {
    for (int column = worker; column < hidden; column += workers)
        weighted_reduce_valid_element(
            combine, weights, topk_ids, output, token, column, hidden);
}

}  // namespace mok_sm90::fp8_block_terminal_route_flags
