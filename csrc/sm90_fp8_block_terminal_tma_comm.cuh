#pragma once

// SM90 raw bulk-TMA dispatch payload for the resident terminal kernel.
//
// Scheduling deliberately remains outside this header.  The caller still owns
// the dense native D(last)-C(q)/D(q-1)-C(0) cursor, cluster-2 ticket transport,
// producer owner-help, ready/closure counters, traps, and the workspace lease.
// This helper replaces only one communication ticket's FP8 dispatch copy;
// combine remains on the existing generic peer-store path.
//
// One cluster ticket covers eight rows.  Each CTA owns four rows and assigns
// one warp to each row.  Warp lane zero moves the complete FP8 row and its
// K128 scales through a CTA-local TMA stage; all lanes participate in invalid
// route zero fill and in the task-boundary convergence.  Four independent
// mbarriers form the phase ring and are initialized once by communication_role.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "kittens.cuh"
#include "sm90_fp8_block_terminal_comm_primitives.cuh"

namespace mok_sm90::fp8_block_terminal_tma_comm {

namespace comm = fp8_block_terminal_comm;

constexpr int CTA_ROWS = 4;
constexpr int WARP_LANES = 32;
constexpr int HIDDEN_SIZE = 4096;
constexpr int SCALE_GROUP = 128;
constexpr int SCALE_COLUMNS = HIDDEN_SIZE / SCALE_GROUP;
constexpr int DATA_BYTES_PER_ROW = HIDDEN_SIZE * sizeof(uint8_t);
constexpr int SCALE_BYTES_PER_ROW = SCALE_COLUMNS * sizeof(float);
constexpr int BYTES_PER_VALID_ROW =
    DATA_BYTES_PER_ROW + SCALE_BYTES_PER_ROW;
constexpr int PIPE_DEPTH = CTA_ROWS;

struct alignas(128) dispatch_staging {
    uint8_t data[PIPE_DEPTH][DATA_BYTES_PER_ROW];
    float scale[PIPE_DEPTH][SCALE_COLUMNS];
};

constexpr int REQUIRED_SMEM_BYTES = sizeof(dispatch_staging);
constexpr int REQUIRED_SMEM_ALIGNMENT = alignof(dispatch_staging);

static_assert(CTA_ROWS == 4);
static_assert(PIPE_DEPTH == 4);
static_assert(SCALE_COLUMNS == 32);
static_assert(DATA_BYTES_PER_ROW == 4096);
static_assert(SCALE_BYTES_PER_ROW == 128);
static_assert(BYTES_PER_VALID_ROW == 4224);
static_assert(REQUIRED_SMEM_BYTES == 16896);
static_assert(REQUIRED_SMEM_ALIGNMENT == 128);
static_assert(DATA_BYTES_PER_ROW % 16 == 0);
static_assert(SCALE_BYTES_PER_ROW % 16 == 0);

#if defined(KITTENS_SM90)

__device__ __forceinline__ uint64_t align_smem_128(uint64_t address) {
    return (address + 127ull) & ~127ull;
}

template <typename Globals>
__device__ __forceinline__ void assign_expert(
        const Globals &g, const int *expert_row_end, int row) {
    int expert = 0;
    while (expert < g.num_local_experts - 1
            && row >= expert_row_end[expert])
        ++expert;
    g.m_indices[row] = expert;
}

// All 128 CTA threads call this helper exactly once for a dispatch ticket.
// Every warp consumes one independent stage and therefore toggles only that
// stage's phase bit.  The wait remains outside the valid-route branch: a zero-
// byte expect completes invalid stages while preserving identical phase counts.
//
// The TMA bulk group is per issuing thread.  Consequently the same lane-zero
// thread that issues both stores performs store_async_wait before bridging the
// async proxy into the caller's later device-release x_ready publication.
template <typename Globals>
__device__ __forceinline__ void dispatch_ticket(
        const Globals &g, const int *expert_row_end, int first_row,
        kittens::semaphore (&inputs_arrived)[PIPE_DEPTH],
        uint32_t &phasebits, uint64_t smem_base_addr) {
    auto &staging = *reinterpret_cast<dispatch_staging *>(
        align_smem_128(smem_base_addr));
    const int lane = static_cast<int>(threadIdx.x) & (WARP_LANES - 1);
    const int stage = static_cast<int>(threadIdx.x) / WARP_LANES;
    const int row = first_row + stage;

    int peer_rank = -1;
    int source_row = -1;
    unsigned int valid_word = 0u;
    if (lane == 0) {
        const comm::route_mapping mapping = comm::decode_route(g, row);
        peer_rank = mapping.peer_rank;
        source_row = mapping.source_row;
        valid_word = mapping.valid ? 1u : 0u;
    }
    peer_rank = __shfl_sync(0xffffffffu, peer_rank, 0);
    source_row = __shfl_sync(0xffffffffu, source_row, 0);
    valid_word = __shfl_sync(0xffffffffu, valid_word, 0);
    const bool route_valid = valid_word != 0u;

    if (lane == 0) {
        kittens::tma::expect_bytes(
            inputs_arrived[stage],
            route_valid ? BYTES_PER_VALID_ROW : 0u);
        if (route_valid) {
            // The source is made ready through the terminal's generic/system
            // input-barrier chain, while this load enters through async proxy.
            asm volatile("{fence.proxy.async.global;}" ::: "memory");
            const uint8_t *src = g.x_peer[peer_rank]
                + static_cast<size_t>(source_row) * HIDDEN_SIZE;
            const float *src_scale = g.x_scale_peer[peer_rank]
                + static_cast<size_t>(source_row) * SCALE_COLUMNS;
            kittens::tma::load_async(
                staging.data[stage], const_cast<uint8_t *>(src),
                DATA_BYTES_PER_ROW, inputs_arrived[stage]);
            kittens::tma::load_async(
                staging.scale[stage], const_cast<float *>(src_scale),
                SCALE_BYTES_PER_ROW, inputs_arrived[stage]);
        }
    }

    if (!route_valid) {
        auto *data = reinterpret_cast<uint4 *>(staging.data[stage]);
        constexpr int DATA_VECTORS = DATA_BYTES_PER_ROW / sizeof(uint4);
        for (int index = lane; index < DATA_VECTORS; index += WARP_LANES)
            data[index] = uint4{0u, 0u, 0u, 0u};
        auto *scale = reinterpret_cast<uint4 *>(staging.scale[stage]);
        constexpr int SCALE_VECTORS = SCALE_BYTES_PER_ROW / sizeof(uint4);
        for (int index = lane; index < SCALE_VECTORS; index += WARP_LANES)
            scale[index] = uint4{0u, 0u, 0u, 0u};
    }

    // Publish expect_tx and invalid-stage zero fill before any warp waits on
    // its phase.  The four raw TMA loads may still remain in flight here.
    __syncthreads();
    kittens::wait(
        inputs_arrived[stage],
        kittens::get_phasebit<0>(phasebits, stage));
    kittens::update_phasebit<0>(phasebits, stage);

    if (lane == 0) {
        uint8_t *dst = g.routed_x
            + static_cast<size_t>(row) * HIDDEN_SIZE;
        float *dst_scale = g.routed_x_scale
            + static_cast<size_t>(row) * SCALE_COLUMNS;
        kittens::tma::store_async(
            dst, staging.data[stage], DATA_BYTES_PER_ROW);
        kittens::tma::store_async(
            dst_scale, staging.scale[stage], SCALE_BYTES_PER_ROW);
        kittens::tma::store_async_wait();
        // The caller publishes x_ready through the generic proxy after this
        // helper returns.  Make both completed async stores visible to it.
        asm volatile("{fence.proxy.async.global;}" ::: "memory");
        assign_expert(g, expert_row_end, row);
    }

    // No bulk group or shared-memory reader may survive this boundary.  The
    // communication role can now alias the same workspace for owner-help WGMMA.
    __syncthreads();
}

#endif  // defined(KITTENS_SM90)

}  // namespace mok_sm90::fp8_block_terminal_tma_comm
