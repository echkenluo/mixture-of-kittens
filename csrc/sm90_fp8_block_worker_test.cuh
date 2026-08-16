#pragma once

// Standalone SM90 numeric primitive for the DeepSeek-V4 FP8 format:
//
//   D[64,64] = sum_k (A_fp8[:, k] @ B_fp8[:, k].T)
//                         * A_scale[:, k] * B_scale[k]
//
// A and B use E4M3 values.  Each k is one 128-element quantization block;
// A_scale is per row and B_scale is shared by the 64 output columns because
// they belong to the same production N128 weight-scale block.  WGMMA consumes
// four K32 chunks as one K128 partial.  Scaling happens before partials from
// different K128 blocks are accumulated, which is required when block scales
// differ along K.
//
// This is intentionally a single-CTA numerical harness.  It establishes the
// exact block-scale arithmetic before the path is pipelined or wired into the
// fused MoE kernel.
#if defined(KITTENS_SM90)
#include <ATen/ATen.h>

namespace mok_sm90 {
using namespace kittens;

namespace fp8_block_test {
using a_st = st_fp8e4m3<64, 128>;
using b_st = st_fp8e4m3<64, 128>;
using d_st = st_bf<64, 64>;
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_st>;
using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;
using acc_rt = rt_fl<16, 64>;

struct globals {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    int k_blocks;
};

__global__ __launch_bounds__(128, 1)
void kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    shared_allocator al((int *)&__shm[0]);
    a_st &a_smem = al.allocate<a_st>();
    b_st &b_smem = al.allocate<b_st>();
    d_st &d_smem = al.allocate<d_st>();

    acc_rt total;
    for (int kb = 0; kb < g.k_blocks; ++kb) {
        warpgroup::load(a_smem, g.A, {0, kb});
        warpgroup::load(b_smem, g.B, {0, kb});
        warpgroup::sync(0);

        acc_rt partial;
        warpgroup::mm_ABt(partial, a_smem, b_smem);
        warpgroup::mma_async_wait<0>();

        // A WGMMA warp owns 16 of the 64 output rows.  An ortho col-vector
        // matches the row layout of acc_rt: x addresses rows 0..7 and y rows
        // 8..15 within that warp's slice.
        typename acc_rt::col_vec row_scale;
        const int row_base = warpid() * 16 + laneid() / 4;
        const float b_scale = g.B_scale[kb];
        row_scale[0][0].x = g.A_scale[row_base * g.k_blocks + kb] * b_scale;
        row_scale[0][0].y = g.A_scale[(row_base + 8) * g.k_blocks + kb] * b_scale;
        warpgroup::mul_row(partial, partial, row_scale);

        if (kb == 0)
            warp::copy(total, partial);
        else
            warpgroup::add(total, total, partial);
        warpgroup::sync(0);
    }

    rt_bf<16, 64> out;
    warp::copy(out, total);
    warpgroup::store(d_smem, out);
    warpgroup::sync(0);
    warpgroup::store(g.D, d_smem, {0, 0});
}

inline at::Tensor entry(at::Tensor A, at::Tensor B,
                        at::Tensor A_scale, at::Tensor B_scale) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "A and B must be rank-2 tensors");
    kittens::py::tensor_check<a_gl>(A);
    kittens::py::tensor_check<b_gl>(B);
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
                "A and B must be CUDA tensors");
    TORCH_CHECK(A.size(0) == 64 && B.size(0) == 64
                    && A.size(1) == B.size(1),
                "expected A[64,K] and B[64,K]");
    TORCH_CHECK(A.size(1) >= 128 && A.size(1) % 128 == 0,
                "K must be positive and divisible by 128");

    TORCH_CHECK(A_scale.is_cuda() && B_scale.is_cuda(),
                "A_scale and B_scale must be CUDA tensors");
    TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Float
                    && B_scale.scalar_type() == at::ScalarType::Float,
                "A_scale and B_scale must be float32");
    TORCH_CHECK(A_scale.is_contiguous() && B_scale.is_contiguous(),
                "A_scale and B_scale must be contiguous");
    kittens::py::device_check(A, B, A_scale, B_scale);

    const int k_blocks = (int)(A.size(1) / 128);
    TORCH_CHECK(A_scale.dim() == 2 && A_scale.size(0) == 64
                    && A_scale.size(1) == k_blocks,
                "A_scale must have shape [64,K/128]");
    TORCH_CHECK(B_scale.dim() == 1 && B_scale.size(0) == k_blocks,
                "B_scale must have shape [K/128]");

    c10::cuda::CUDAGuard device_guard(A.device());
    auto D = at::empty({64, 64}, A.options().dtype(at::ScalarType::BFloat16));
    globals g{
        kittens::py::tensor_to_gl<a_gl>(A),
        kittens::py::tensor_to_gl<b_gl>(B),
        kittens::py::tensor_to_gl<d_gl>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        k_blocks,
    };
    constexpr int SMEM = sizeof(a_st) + sizeof(b_st) + sizeof(d_st) + 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    kernel<<<1, 128, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    return D;
}

// Masked expert-major form matching the tensor contract used by DeepGEMM's
// grouped masked GEMM on Hopper:
//
//   A       [E, max_m, K]       FP8 E4M3
//   B       [E, N, K]           FP8 E4M3
//   A_scale [E, max_m, K/128]   float32
//   B_scale [E, N/128, K/128]   float32
//   masked_m[E]                 int32, valid rows per expert
//
// One CTA owns one M64xN64 output tile.  The two N64 halves of an N128
// weight-scale block intentionally consume the same B scale.  Rows at or
// beyond masked_m are unspecified and must be ignored by the caller, just as
// they are by the masked grouped-GEMM contract.
namespace grouped {
using a_gl = gl<fp8e4m3, 1, -1, -1, -1, a_st>;
using b_gl = gl<fp8e4m3, 1, -1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, -1, -1, -1, d_st>;

struct globals {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    const int *masked_m;
    int max_m;
    int n;
    int k_blocks;
    int m_tiles;
    int n_tiles;
};

template <bool PIPELINED>
__global__ __launch_bounds__(128, 1)
void kernel(const __grid_constant__ globals g) {
    int task = blockIdx.x;
    const int n_tile = task % g.n_tiles;
    task /= g.n_tiles;
    const int m_tile = task % g.m_tiles;
    const int expert = task / g.m_tiles;
    if (m_tile * 64 >= g.masked_m[expert])
        return;

    extern __shared__ int __shm[];
    shared_allocator al((int *)&__shm[0]);
    constexpr int PIPE_DEPTH = PIPELINED ? 2 : 1;
    auto &a_smem = al.allocate<a_st, PIPE_DEPTH>();
    auto &b_smem = al.allocate<b_st, PIPE_DEPTH>();
    d_st &d_smem = al.allocate<d_st>();

    acc_rt total;
    if constexpr (PIPELINED) {
        warpgroup::load_async(a_smem[0], g.A, {expert, m_tile, 0});
        warpgroup::load_async(b_smem[0], g.B, {expert, n_tile, 0});
    }
    for (int kb = 0; kb < g.k_blocks; ++kb) {
        const int stage = PIPELINED ? kb % PIPE_DEPTH : 0;
        if constexpr (PIPELINED) {
            warpgroup::load_async_wait<0>(0);
        } else {
            warpgroup::load(a_smem[0], g.A, {expert, m_tile, kb});
            warpgroup::load(b_smem[0], g.B, {expert, n_tile, kb});
            warpgroup::sync(0);
        }

        acc_rt partial;
        warpgroup::mm_ABt(partial, a_smem[stage], b_smem[stage]);
        if constexpr (PIPELINED) {
            if (kb + 1 < g.k_blocks) {
                const int next_stage = (kb + 1) % PIPE_DEPTH;
                warpgroup::load_async(
                    a_smem[next_stage], g.A, {expert, m_tile, kb + 1});
                warpgroup::load_async(
                    b_smem[next_stage], g.B, {expert, n_tile, kb + 1});
            }
        }
        warpgroup::mma_async_wait<0>();

        typename acc_rt::col_vec row_scale;
        const int local_row = warpid() * 16 + laneid() / 4;
        const int global_row = m_tile * 64 + local_row;
        const float b_scale =
            g.B_scale[(expert * (g.n / 128) + n_tile / 2) * g.k_blocks + kb];
        row_scale[0][0].x =
            g.A_scale[(expert * g.max_m + global_row) * g.k_blocks + kb]
            * b_scale;
        row_scale[0][0].y =
            g.A_scale[(expert * g.max_m + global_row + 8) * g.k_blocks + kb]
            * b_scale;
        warpgroup::mul_row(partial, partial, row_scale);

        if (kb == 0)
            warp::copy(total, partial);
        else
            warpgroup::add(total, total, partial);
        warpgroup::sync(0);
    }

    rt_bf<16, 64> out;
    warp::copy(out, total);
    warpgroup::store(d_smem, out);
    warpgroup::sync(0);
    warpgroup::store(g.D, d_smem, {expert, m_tile, n_tile});
}

template <bool PIPELINED>
inline at::Tensor entry_impl_out(at::Tensor A, at::Tensor B,
                                 at::Tensor A_scale, at::Tensor B_scale,
                                 at::Tensor masked_m, at::Tensor D) {
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "A and B must have shapes [E,max_m,K] and [E,N,K]");
    kittens::py::tensor_check<a_gl>(A);
    kittens::py::tensor_check<b_gl>(B);
    const int experts = (int)A.size(0);
    const int max_m = (int)A.size(1);
    const int k = (int)A.size(2);
    const int n = (int)B.size(1);
    TORCH_CHECK(experts > 0 && B.size(0) == experts && B.size(2) == k,
                "A and B expert/K dimensions must match");
    TORCH_CHECK(max_m >= 64 && max_m % 64 == 0,
                "max_m must be positive and divisible by 64");
    TORCH_CHECK(n >= 128 && n % 128 == 0,
                "N must be positive and divisible by 128");
    TORCH_CHECK(k >= 128 && k % 128 == 0,
                "K must be positive and divisible by 128");

    TORCH_CHECK(A_scale.is_cuda() && B_scale.is_cuda() && masked_m.is_cuda(),
                "scales and masked_m must be CUDA tensors");
    TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Float
                    && B_scale.scalar_type() == at::ScalarType::Float,
                "A_scale and B_scale must be float32");
    TORCH_CHECK(masked_m.scalar_type() == at::ScalarType::Int,
                "masked_m must be int32");
    TORCH_CHECK(A_scale.is_contiguous() && B_scale.is_contiguous()
                    && masked_m.is_contiguous(),
                "scales and masked_m must be contiguous");
    const int k_blocks = k / 128;
    TORCH_CHECK(A_scale.dim() == 3 && A_scale.size(0) == experts
                    && A_scale.size(1) == max_m
                    && A_scale.size(2) == k_blocks,
                "A_scale must have shape [E,max_m,K/128]");
    TORCH_CHECK(B_scale.dim() == 3 && B_scale.size(0) == experts
                    && B_scale.size(1) == n / 128
                    && B_scale.size(2) == k_blocks,
                "B_scale must have shape [E,N/128,K/128]");
    TORCH_CHECK(masked_m.dim() == 1 && masked_m.size(0) == experts,
                "masked_m must have shape [E]");
    TORCH_CHECK(D.dim() == 3 && D.size(0) == experts
                    && D.size(1) == max_m && D.size(2) == n,
                "D must have shape [E,max_m,N]");
    TORCH_CHECK(D.is_cuda() && D.scalar_type() == at::ScalarType::BFloat16,
                "D must be a CUDA bfloat16 tensor");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    kittens::py::tensor_check<d_gl>(D);
    kittens::py::device_check(A, B, A_scale, B_scale, masked_m);
    kittens::py::device_check(A, D);

    c10::cuda::CUDAGuard device_guard(A.device());
    const int m_tiles = max_m / 64;
    const int n_tiles = n / 64;
    globals g{
        kittens::py::tensor_to_gl<a_gl>(A),
        kittens::py::tensor_to_gl<b_gl>(B),
        kittens::py::tensor_to_gl<d_gl>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        masked_m.data_ptr<int>(),
        max_m,
        n,
        k_blocks,
        m_tiles,
        n_tiles,
    };
    constexpr int PIPE_DEPTH = PIPELINED ? 2 : 1;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    CUDACHECK(cudaFuncSetAttribute(
        kernel<PIPELINED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    kernel<PIPELINED><<<experts * m_tiles * n_tiles, 128, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    return D;
}

template <bool PIPELINED>
inline at::Tensor entry_impl(at::Tensor A, at::Tensor B,
                             at::Tensor A_scale, at::Tensor B_scale,
                             at::Tensor masked_m) {
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "A and B must have shapes [E,max_m,K] and [E,N,K]");
    auto D = at::empty({A.size(0), A.size(1), B.size(1)},
                       A.options().dtype(at::ScalarType::BFloat16));
    return entry_impl_out<PIPELINED>(
        A, B, A_scale, B_scale, masked_m, D);
}

inline at::Tensor entry(at::Tensor A, at::Tensor B,
                        at::Tensor A_scale, at::Tensor B_scale,
                        at::Tensor masked_m) {
    return entry_impl<false>(A, B, A_scale, B_scale, masked_m);
}

inline at::Tensor entry_pipelined(at::Tensor A, at::Tensor B,
                                  at::Tensor A_scale, at::Tensor B_scale,
                                  at::Tensor masked_m) {
    return entry_impl<true>(A, B, A_scale, B_scale, masked_m);
}

inline at::Tensor entry_out(at::Tensor A, at::Tensor B,
                            at::Tensor A_scale, at::Tensor B_scale,
                            at::Tensor masked_m, at::Tensor D) {
    return entry_impl_out<false>(A, B, A_scale, B_scale, masked_m, D);
}

inline at::Tensor entry_pipelined_out(at::Tensor A, at::Tensor B,
                                      at::Tensor A_scale, at::Tensor B_scale,
                                      at::Tensor masked_m, at::Tensor D) {
    return entry_impl_out<true>(A, B, A_scale, B_scale, masked_m, D);
}

} // namespace grouped

// Contiguous expert-grouped form matching DeepGEMM's normal-DeepEP contract:
//
//   A         [M, K]               FP8 E4M3
//   B         [E, N, K]            FP8 E4M3
//   A_scale   [M, K/128]           float32
//   B_scale   [E, N/128, K/128]    float32
//   m_indices [M]                  int32 expert id per row
//   D         [M, N]               BF16
//
// DeepEP normal dispatch aligns every expert segment to 128 rows.  Therefore
// each M64 tile belongs to exactly one expert and its first m_indices entry is
// sufficient to select B.  Keeping A/D compact avoids an expert-major padding
// conversion in the production Prefill path.
namespace contiguous {
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_st>;
using b_gl = gl<fp8e4m3, 1, -1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;

struct globals {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    const int *m_indices;
    const int *num_tokens;
    int n;
    int k_blocks;
    int n_tiles;
};

__cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
__global__
void kernel(const __grid_constant__ globals g) {
    const int cta_rank = cluster_ctarank();
    const int n_pairs = g.n_tiles / 2;
    const int cluster_idx = clusterIdx().x;
    const int n_tile_base = 2 * (cluster_idx % n_pairs);
    const int n_tile = n_tile_base + cta_rank;
    const int m_tile = cluster_idx / n_pairs;
    const int global_row_base = m_tile * 64;
    if (g.num_tokens != nullptr && global_row_base >= g.num_tokens[0])
        return;
    const int expert = g.m_indices[global_row_base];

    extern __shared__ int __shm[];
    shared_allocator al((int *)&__shm[0]);
    constexpr int PIPE_DEPTH = 2;
    auto &a_smem = al.allocate<a_st, PIPE_DEPTH>();
    auto &b_smem = al.allocate<b_st, PIPE_DEPTH>();
    d_st &d_smem = al.allocate<d_st>();
    __shared__ semaphore inputs_arrived[PIPE_DEPTH];
    __shared__ semaphore inputs_finished[PIPE_DEPTH];
    __shared__ semaphore inputs_ready[PIPE_DEPTH];

    if (threadIdx.x < PIPE_DEPTH) {
        init_semaphore(inputs_arrived[threadIdx.x], 0, 1);
        init_semaphore(inputs_finished[threadIdx.x], 0, 1);
        init_semaphore(inputs_ready[threadIdx.x], 0, 2);
    }
    everyone::tma::cluster::sync();

    acc_rt total;
    uint32_t phasebits = 0xFFFF0000;
    uint32_t ready_phase = 0;
    if (threadIdx.x == 0) {
        wait(inputs_finished[0], get_phasebit<1>(phasebits, 0));
        update_phasebit<1>(phasebits, 0);
        tma::cluster::expect_bytes(
            inputs_arrived[0], sizeof(a_st) + sizeof(b_st));
        tma::cluster::load_async(
            b_smem[0], g.B, {expert, n_tile, 0}, inputs_arrived[0],
            static_cast<uint16_t>(1 << cta_rank));
        tma::cluster::arrive(inputs_ready[0], 0);
        if (cta_rank == 0) {
            wait(inputs_ready[0], get_phasebit<0>(ready_phase, 0));
            update_phasebit<0>(ready_phase, 0);
            tma::cluster::load_async(
                a_smem[0], g.A, {m_tile, 0}, inputs_arrived[0], 0b11);
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
                    inputs_arrived[next_stage],
                    sizeof(a_st) + sizeof(b_st));
                tma::cluster::load_async(
                    b_smem[next_stage], g.B,
                    {expert, n_tile, kb + 1}, inputs_arrived[next_stage],
                    static_cast<uint16_t>(1 << cta_rank));
                tma::cluster::arrive(inputs_ready[next_stage], 0);
                if (cta_rank == 0) {
                    wait(inputs_ready[next_stage],
                         get_phasebit<0>(ready_phase, next_stage));
                    update_phasebit<0>(ready_phase, next_stage);
                    tma::cluster::load_async(
                        a_smem[next_stage], g.A, {m_tile, kb + 1},
                        inputs_arrived[next_stage], 0b11);
                }
            }
        }
        warpgroup::mma_async_wait<0>();

        typename acc_rt::col_vec row_scale;
        const int local_row = warpid() * 16 + laneid() / 4;
        const int global_row = global_row_base + local_row;
        const float b_scale =
            g.B_scale[(expert * (g.n / 128) + n_tile / 2) * g.k_blocks + kb];
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
    warpgroup::store(g.D, d_smem, {m_tile, n_tile});
}

inline at::Tensor entry_pipelined_out_impl(
    at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale,
    at::Tensor m_indices, const at::Tensor *num_tokens, at::Tensor D) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 3,
                "A and B must have shapes [M,K] and [E,N,K]");
    kittens::py::tensor_check<a_gl>(A);
    kittens::py::tensor_check<b_gl>(B);
    const int total_m = (int)A.size(0);
    const int experts = (int)B.size(0);
    const int n = (int)B.size(1);
    const int k = (int)A.size(1);
    TORCH_CHECK(experts > 0 && B.size(2) == k,
                "A and B K dimensions must match");
    TORCH_CHECK(total_m >= 64 && total_m % 64 == 0,
                "M must be positive and divisible by 64");
    TORCH_CHECK(n >= 128 && n % 128 == 0,
                "N must be positive and divisible by 128");
    TORCH_CHECK(k >= 128 && k % 128 == 0,
                "K must be positive and divisible by 128");

    TORCH_CHECK(A_scale.is_cuda() && B_scale.is_cuda()
                    && m_indices.is_cuda(),
                "scales and m_indices must be CUDA tensors");
    TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Float
                    && B_scale.scalar_type() == at::ScalarType::Float,
                "A_scale and B_scale must be float32");
    TORCH_CHECK(m_indices.scalar_type() == at::ScalarType::Int,
                "m_indices must be int32");
    TORCH_CHECK(A_scale.is_contiguous() && B_scale.is_contiguous()
                    && m_indices.is_contiguous(),
                "scales and m_indices must be contiguous");
    const int k_blocks = k / 128;
    TORCH_CHECK(A_scale.dim() == 2 && A_scale.size(0) == total_m
                    && A_scale.size(1) == k_blocks,
                "A_scale must have shape [M,K/128]");
    TORCH_CHECK(B_scale.dim() == 3 && B_scale.size(0) == experts
                    && B_scale.size(1) == n / 128
                    && B_scale.size(2) == k_blocks,
                "B_scale must have shape [E,N/128,K/128]");
    TORCH_CHECK(m_indices.dim() == 1 && m_indices.size(0) == total_m,
                "m_indices must have shape [M]");
    TORCH_CHECK(D.dim() == 2 && D.size(0) == total_m && D.size(1) == n,
                "D must have shape [M,N]");
    TORCH_CHECK(D.is_cuda() && D.scalar_type() == at::ScalarType::BFloat16,
                "D must be a CUDA bfloat16 tensor");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    kittens::py::tensor_check<d_gl>(D);
    kittens::py::device_check(A, B, A_scale, B_scale, m_indices);
    kittens::py::device_check(A, D);
    if (num_tokens != nullptr) {
        TORCH_CHECK(
            num_tokens->is_cuda()
                && num_tokens->scalar_type() == at::ScalarType::Int
                && num_tokens->is_contiguous()
                && num_tokens->dim() == 1 && num_tokens->numel() == 1,
            "num_tokens must be contiguous CUDA int32 [1]");
        TORCH_CHECK(num_tokens->device() == A.device(),
                    "num_tokens must share the input device");
    }

    c10::cuda::CUDAGuard device_guard(A.device());
    const int m_tiles = total_m / 64;
    const int n_tiles = n / 64;
    globals g{
        kittens::py::tensor_to_gl<a_gl>(A),
        kittens::py::tensor_to_gl<b_gl>(B),
        kittens::py::tensor_to_gl<d_gl>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        m_indices.data_ptr<int>(),
        num_tokens == nullptr ? nullptr : num_tokens->data_ptr<int>(),
        n,
        k_blocks,
        n_tiles,
    };
    constexpr int PIPE_DEPTH = 2;
    constexpr int SMEM =
        PIPE_DEPTH * (sizeof(a_st) + sizeof(b_st)) + sizeof(d_st) + 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    kernel<<<m_tiles * n_tiles, 128, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    return D;
}

inline at::Tensor entry_pipelined_out(at::Tensor A, at::Tensor B,
                                      at::Tensor A_scale,
                                      at::Tensor B_scale,
                                      at::Tensor m_indices, at::Tensor D) {
    return entry_pipelined_out_impl(
        A, B, A_scale, B_scale, m_indices, nullptr, D);
}

inline at::Tensor entry_pipelined_dynamic_out(
    at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale,
    at::Tensor m_indices, at::Tensor num_tokens, at::Tensor D) {
    return entry_pipelined_out_impl(
        A, B, A_scale, B_scale, m_indices, &num_tokens, D);
}

} // namespace contiguous

} // namespace fp8_block_test
} // namespace mok_sm90
#endif
