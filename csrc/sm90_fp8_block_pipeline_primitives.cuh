#pragma once

// Production SM90 FP8 block-scale pipeline primitives.
//
// One cluster owns an M64 x N128 output tile.  Each CTA rank owns one N64
// half and consumes the same multicast A[M64,K128] tiles with independent
// B[N64,K128] tiles.  Stage readiness, completion, scheduling, transport,
// and expert selection stay in the caller; this header owns only the common
// layouts, arithmetic coordinate mapping, and numerical WGMMA pipeline.
//
// In particular, run_tile receives the already selected expert explicitly.
// A dispatch/compute caller must perform its x-ready acquire before reading
// m_indices and passing the result here.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mok_sm90::fp8_block_pipeline {

// DeepSeek-V4 terminal activation contract.  A logical 256-thread worker
// group covers one H4096 gate/up row: worker u owns eight adjacent output
// values, and every 16-worker subgroup owns one K128 FP8 scale.  Cluster-2
// callers form u as cta_rank * 128 + threadIdx.x; the contiguous reference
// kernel passes threadIdx.x directly.  Keep this mapping and the arithmetic
// order stable so both paths remain bitwise comparable.
constexpr int V4_INTERMEDIATE = 2048;
constexpr int V4_GATE_UP = 2 * V4_INTERMEDIATE;
constexpr int V4_FP8_GROUP = 128;
constexpr int V4_ACTIVATION_VALUES_PER_WORKER = 8;
constexpr int V4_ACTIVATION_WORKERS =
    V4_INTERMEDIATE / V4_ACTIVATION_VALUES_PER_WORKER;
constexpr float V4_FP8_MAX = 448.0f;

__device__ __forceinline__ uint16_t pack_v4_fp8x2(float x, float y) {
    x = fmaxf(fminf(x, V4_FP8_MAX), -V4_FP8_MAX);
    y = fmaxf(fminf(y, V4_FP8_MAX), -V4_FP8_MAX);
    uint16_t result;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
                 : "=h"(result) : "f"(x), "f"(y));
    return result;
}

__device__ __forceinline__ float subgroup_max_16(float value) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}

__device__ __forceinline__ void activate_quant_worker(
    const __nv_bfloat16 *input, uint8_t *output, float *output_scale,
    int row, int worker, float limit) {
    const auto *row_pairs = reinterpret_cast<const __nv_bfloat162 *>(
        input + static_cast<size_t>(row) * V4_GATE_UP);
    auto *out_pairs = reinterpret_cast<uint16_t *>(
        output + static_cast<size_t>(row) * V4_INTERMEDIATE);
    const int element = worker * V4_ACTIVATION_VALUES_PER_WORKER;
    const int pair = element / 2;
    const __nv_bfloat162 limit2 = __floats2bfloat162_rn(limit, limit);
    const __nv_bfloat162 neg_limit2 =
        __floats2bfloat162_rn(-limit, -limit);
    float values[V4_ACTIVATION_VALUES_PER_WORKER];
    float local_max = 0.0f;

#pragma unroll
    for (int index = 0; index < V4_ACTIVATION_VALUES_PER_WORKER / 2;
         ++index) {
        __nv_bfloat162 gate = __hmin2(row_pairs[pair + index], limit2);
        __nv_bfloat162 up = __hmax2(
            row_pairs[V4_INTERMEDIATE / 2 + pair + index], neg_limit2);
        up = __hmin2(up, limit2);
        const float2 gate_f = __bfloat1622float2(gate);
        const float2 up_f = __bfloat1622float2(up);
        const float x =
            gate_f.x / (1.0f + __expf(-gate_f.x)) * up_f.x;
        const float y =
            gate_f.y / (1.0f + __expf(-gate_f.y)) * up_f.y;
        values[2 * index] = x;
        values[2 * index + 1] = y;
        local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }

    const float absmax = fmaxf(subgroup_max_16(local_max), 1e-10f);
    const float scale = absmax / V4_FP8_MAX;
    const float inv_scale = 1.0f / scale;
#pragma unroll
    for (int index = 0; index < V4_ACTIVATION_VALUES_PER_WORKER / 2;
         ++index)
        out_pairs[pair + index] = pack_v4_fp8x2(
            values[2 * index] * inv_scale,
            values[2 * index + 1] * inv_scale);
    if ((threadIdx.x & 15) == 0)
        output_scale[static_cast<size_t>(row)
                         * (V4_INTERMEDIATE / V4_FP8_GROUP)
                     + worker / 16] = scale;
}

// One worker owns one output column.  Route slots are accumulated in the
// production routed-epilogue order: a rounded FP32 multiply for slot zero,
// then rounded FP32 FMAs for slots one through topk-1, followed by one BF16
// rounding.  Spelling out the instructions makes exact behavior independent
// of compiler contraction choices under --use_fast_math.
__device__ __forceinline__ void weighted_reduce_element(
    const __nv_bfloat16 *combine, const float *weights,
    __nv_bfloat16 *output, int token, int column, int topk, int hidden) {
    const size_t route_base = static_cast<size_t>(token) * topk;
    float accumulator = __fmul_rn(
        __bfloat162float(combine[route_base * hidden + column]),
        weights[route_base]);
    for (int route = 1; route < topk; ++route) {
        accumulator = __fmaf_rn(
            __bfloat162float(
                combine[(route_base + route) * hidden + column]),
            weights[route_base + route], accumulator);
    }
    output[static_cast<size_t>(token) * hidden + column] =
        __float2bfloat16_rn(accumulator);
}

}  // namespace mok_sm90::fp8_block_pipeline

#if defined(KITTENS_SM90)

#include "kittens.cuh"

namespace mok_sm90::fp8_block_pipeline {

using namespace kittens;

using a_st = st_fp8e4m3<64, 128>;
using b_st = st_fp8e4m3<64, 128>;
using d_st = st_bf<64, 64>;
using acc_rt = rt_fl<16, 64>;

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
    const Globals &g, const tile &coord, int expert, int cta_rank,
    uint32_t &phasebits, uint32_t &ready_phase,
    a_st (&a_smem)[PIPE_DEPTH], b_st (&b_smem)[PIPE_DEPTH], d_st &d_smem,
    semaphore (&inputs_arrived)[PIPE_DEPTH],
    semaphore (&inputs_finished)[PIPE_DEPTH],
    semaphore (&inputs_ready)[PIPE_DEPTH]) {
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

}  // namespace mok_sm90::fp8_block_pipeline

#endif  // defined(KITTENS_SM90)
