#pragma once

// Shared SM90 data-movement primitives for the DeepSeek-V4 routed FP8 path.
//
// These helpers deliberately stop below scheduling and publication.  They do
// not draw tickets, wait on barriers, update ready counters, manage a lease,
// or trap.  A fixed communication role can therefore call the same row core
// as K1/K2 while retaining its own progress and dependency protocol.
//
// push_routed_row returns the decoded destination needed by a terminal
// per-route publisher, but does not publish a route-ready flag.  The current
// K2 path has only a full-rank completion barrier, not the terminal protocol's
// per-route happens-before chain.  A terminal caller must separately perform
// every writer lane's system-scope release fence, converge those lanes, and
// issue the elected lane's release store only after that convergence.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mok_sm90::fp8_block_terminal_comm {

constexpr int WARP_LANES = 32;
constexpr int DISPATCH_VEC_CHUNK = 8;

struct route_mapping {
    int peer_rank;
    int peer_token_idx;
    int source_row;
    bool valid;
};

template <typename Globals>
__device__ __forceinline__ int bounded_valid_rows(const Globals &g) {
    const int device_rows = g.num_tokens[0];
    return device_rows < g.schedule_capacity
        ? device_rows
        : g.schedule_capacity;
}

template <typename Globals>
__device__ __forceinline__ route_mapping decode_route(
    const Globals &g, int row) {
    const int peer_rank = g.schedule_peer_rank[row];
    const int peer_token_idx = g.schedule_peer_token_idx[row];
    const bool valid = peer_rank >= 0 && peer_rank < g.ep_size
        && peer_token_idx >= 0
        && peer_token_idx < g.num_local_tokens * g.topk;
    return {
        peer_rank,
        peer_token_idx,
        valid ? peer_token_idx / g.topk : -1,
        valid,
    };
}

template <typename Globals>
__device__ __forceinline__ void build_expert_row_ends(
    const Globals &g, int *expert_row_end) {
    int offset = 0;
    for (int expert = 0; expert < g.num_local_experts; ++expert) {
        offset += g.tokens_per_expert[expert];
        expert_row_end[expert] = offset;
    }
}

// One warp copies one routed FP8 row, its K128 float scales, and its expert
// index.  The caller owns row assignment and must converge the warp before
// publishing row readiness.  Invalid routes preserve K1's zero-fill behavior
// while still assigning the row's expert segment.
template <typename Globals>
__device__ __forceinline__ void dispatch_copy_row(
    const Globals &g, const int *expert_row_end, int row, int lane) {
    const route_mapping route = decode_route(g, row);
    const int fp8_vectors = g.hidden_size / static_cast<int>(sizeof(uint4));
    auto *dst_vectors = reinterpret_cast<uint4 *>(g.routed_x)
        + static_cast<size_t>(row) * fp8_vectors;
    float *dst_scale = g.routed_x_scale
        + static_cast<size_t>(row) * g.scale_columns;
    if (route.valid) {
        const auto *src_vectors =
            reinterpret_cast<const uint4 *>(g.x_peer[route.peer_rank])
            + static_cast<size_t>(route.source_row) * fp8_vectors;
        const float *src_scale = g.x_scale_peer[route.peer_rank]
            + static_cast<size_t>(route.source_row) * g.scale_columns;
        // Stage through registers: all remote loads in a 4 KiB warp chunk
        // issue before any potentially aliasing destination store.
        uint4 buffer[DISPATCH_VEC_CHUNK];
        for (int base = 0;
             base < fp8_vectors;
             base += WARP_LANES * DISPATCH_VEC_CHUNK) {
#pragma unroll
            for (int item = 0; item < DISPATCH_VEC_CHUNK; ++item) {
                const int index = base + lane + item * WARP_LANES;
                if (index < fp8_vectors)
                    buffer[item] = src_vectors[index];
            }
#pragma unroll
            for (int item = 0; item < DISPATCH_VEC_CHUNK; ++item) {
                const int index = base + lane + item * WARP_LANES;
                if (index < fp8_vectors)
                    dst_vectors[index] = buffer[item];
            }
        }
        for (int index = lane; index < g.scale_columns;
             index += WARP_LANES)
            dst_scale[index] = src_scale[index];
    } else {
        const uint4 zero{0, 0, 0, 0};
#pragma unroll 4
        for (int index = lane; index < fp8_vectors; index += WARP_LANES)
            dst_vectors[index] = zero;
        for (int index = lane; index < g.scale_columns;
             index += WARP_LANES)
            dst_scale[index] = 0.0f;
    }
    if (lane == 0) {
        int expert = 0;
        while (expert < g.num_local_experts - 1
               && row >= expert_row_end[expert])
            ++expert;
        g.m_indices[row] = expert;
    }
}

// One warp pushes one BF16 routed row to its decoded peer/route slot.  The
// returned mapping is identical in every lane and is the minimum information
// a future terminal caller needs to publish the corresponding route flag.
// No fence, convergence, counter, or flag operation is hidden in this core.
template <typename Globals>
__device__ __forceinline__ route_mapping push_routed_row(
    const Globals &g, int row, int lane) {
    const route_mapping route = decode_route(g, row);
    if (!route.valid)
        return route;
    const int row_vectors =
        g.hidden_size * 2 / static_cast<int>(sizeof(uint4));
    const auto *src = reinterpret_cast<const uint4 *>(g.routed_y)
        + static_cast<size_t>(row) * row_vectors;
    auto *dst = reinterpret_cast<uint4 *>(g.combine_peer[route.peer_rank])
        + static_cast<size_t>(route.peer_token_idx) * row_vectors;
#pragma unroll 4
    for (int index = lane; index < row_vectors; index += WARP_LANES)
        dst[index] = src[index];
    return route;
}

}  // namespace mok_sm90::fp8_block_terminal_comm
