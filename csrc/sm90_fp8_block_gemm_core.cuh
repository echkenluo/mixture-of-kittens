#pragma once

// Shared SM90 FP8 block-scale GEMM primitive used by both fused cuts.
//
// One cluster owns an M64 x N128 output tile.  Each CTA rank owns one N64
// half and consumes the same multicast A[M64,K128] tiles with independent
// B[N64,K128] tiles.  Stage-specific readiness, completion, scheduling, and
// transport stay in the callers; this header contains only the common
// numerical pipeline and the arithmetic mapping from a cluster task to its
// M64/N64 tile.
#if defined(KITTENS_SM90)

#include "sm90_fp8_block_worker_test.cuh"

namespace mok_sm90::fp8_block_gemm_core {

using fp8_block_test::a_st;
using fp8_block_test::b_st;
using fp8_block_test::d_st;
using fp8_block_test::acc_rt;

constexpr int PIPE_DEPTH = 2;

struct tile {
    int n_tile;
    int m_tile;
    int global_row_base;
};

template <typename Globals>
__device__ __forceinline__ tile decode_tile(
    const Globals &g, int gemm_cluster_idx, int cta_rank) {
    const int n_pairs = g.n_tiles / 2;
    const int n_tile_base = 2 * (gemm_cluster_idx % n_pairs);
    const int m_tile = gemm_cluster_idx / n_pairs;
    return {
        n_tile_base + cta_rank,
        m_tile,
        m_tile * 64,
    };
}

template <typename Globals>
__device__ __forceinline__ void run_tile(
    const Globals &g, const tile &coord, int cta_rank,
    uint32_t &phasebits, uint32_t &ready_phase,
    a_st (&a_smem)[PIPE_DEPTH], b_st (&b_smem)[PIPE_DEPTH], d_st &d_smem,
    semaphore (&inputs_arrived)[PIPE_DEPTH],
    semaphore (&inputs_finished)[PIPE_DEPTH],
    semaphore (&inputs_ready)[PIPE_DEPTH]) {
    const int expert = g.m_indices[coord.global_row_base];

    acc_rt total;
    if (threadIdx.x == 0) {
        wait(inputs_finished[0], get_phasebit<1>(phasebits, 0));
        update_phasebit<1>(phasebits, 0);
        tma::cluster::expect_bytes(
            inputs_arrived[0], sizeof(a_st) + sizeof(b_st));
        tma::cluster::load_async(
            b_smem[0], g.B, {expert, coord.n_tile, 0}, inputs_arrived[0],
            static_cast<uint16_t>(1 << cta_rank));
        tma::cluster::arrive(inputs_ready[0], 0);
        if (cta_rank == 0) {
            wait(inputs_ready[0], get_phasebit<0>(ready_phase, 0));
            update_phasebit<0>(ready_phase, 0);
            tma::cluster::load_async(
                a_smem[0], g.A, {coord.m_tile, 0}, inputs_arrived[0], 0b11);
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
                    inputs_arrived[next_stage], sizeof(a_st) + sizeof(b_st));
                tma::cluster::load_async(
                    b_smem[next_stage], g.B,
                    {expert, coord.n_tile, kb + 1},
                    inputs_arrived[next_stage],
                    static_cast<uint16_t>(1 << cta_rank));
                tma::cluster::arrive(inputs_ready[next_stage], 0);
                if (cta_rank == 0) {
                    wait(inputs_ready[next_stage],
                         get_phasebit<0>(ready_phase, next_stage));
                    update_phasebit<0>(ready_phase, next_stage);
                    tma::cluster::load_async(
                        a_smem[next_stage], g.A,
                        {coord.m_tile, kb + 1}, inputs_arrived[next_stage],
                        0b11);
                }
            }
        }
        warpgroup::mma_async_wait<0>();

        typename acc_rt::col_vec row_scale;
        const int local_row = warpid() * 16 + laneid() / 4;
        const int global_row = coord.global_row_base + local_row;
        const float b_scale =
            g.B_scale[(expert * (g.n / 128) + coord.n_tile / 2)
                      * g.k_blocks + kb];
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
    warpgroup::store(g.D, d_smem, {coord.m_tile, coord.n_tile});
}

} // namespace mok_sm90::fp8_block_gemm_core

#endif // defined(KITTENS_SM90)
