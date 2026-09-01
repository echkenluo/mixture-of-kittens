#pragma once

// Warp-level FP8 block-scaled GEMM primitive for M16/M32 expert tails on
// Hopper.  One CTA owns one M16xN64 output tile and uses
// mma.sync.aligned.m16n8k32 with FP32 accumulation.  K128 partials are scaled
// before accumulation so the arithmetic matches DeepSeek's block-scaled FP8
// contract.
#if defined(KITTENS_SM90)
#include "pyutils/torchutils.cuh"
#include <ATen/ops/empty.h>

namespace mok_sm90::fp8_block_tail_test {
using namespace kittens;

using a_st = st_fp8e4m3<16, 128>;
using b_st = st_fp8e4m3<64, 128>;
using d_st = st_bf<16, 64>;
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_st>;
using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;
using a_rt = rt_fp8e4m3<16, 128>;
using b_rt = rt_fp8e4m3<64, 128>;
using acc_rt = rt_fl<16, 64>;

struct globals {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    int k_blocks;
    int n_tiles;
};

__global__ __launch_bounds__(32, 4)
void kernel(const __grid_constant__ globals g) {
    const int n_tile = blockIdx.x % g.n_tiles;
    const int m_tile = blockIdx.x / g.n_tiles;

    extern __shared__ int __shm[];
    shared_allocator<16> al((int *)&__shm[0]);
    a_st &a_smem = al.allocate<a_st>();
    b_st &b_smem = al.allocate<b_st>();
    d_st &d_smem = al.allocate<d_st>();

    acc_rt total;
    for (int kb = 0; kb < g.k_blocks; ++kb) {
        warp::load(a_smem, g.A, {m_tile, kb});
        warp::load(b_smem, g.B, {n_tile, kb});
        __syncwarp();

        a_rt a;
        b_rt b;
        warp::load(a, a_smem);
        warp::load(b, b_smem);

        acc_rt partial;
        warp::zero(partial);
        warp::mma_ABt(partial, a, b, partial);

        // Each lane owns two output rows for a column-vector operation: the
        // row laneid()/4 and the same row plus eight.
        typename acc_rt::col_vec row_scale;
        const int row_base = m_tile * 16 + laneid() / 4;
        const float b_scale =
            g.B_scale[(n_tile / 2) * g.k_blocks + kb];
        row_scale[0][0].x =
            g.A_scale[row_base * g.k_blocks + kb] * b_scale;
        row_scale[0][0].y =
            g.A_scale[(row_base + 8) * g.k_blocks + kb] * b_scale;
        warp::mul_row(partial, partial, row_scale);

        if (kb == 0)
            warp::copy(total, partial);
        else
            warp::add(total, total, partial);
        __syncwarp();
    }

    rt_bf<16, 64> out;
    warp::copy(out, total);
    warp::store(d_smem, out);
    __syncwarp();
    warp::store(g.D, d_smem, {m_tile, n_tile});
}

inline at::Tensor entry(at::Tensor A, at::Tensor B,
                        at::Tensor A_scale, at::Tensor B_scale) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "A and B must be rank-2 tensors");
    kittens::py::tensor_check<a_gl>(A);
    kittens::py::tensor_check<b_gl>(B);
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
                "A and B must be CUDA tensors");
    TORCH_CHECK((A.size(0) == 16 || A.size(0) == 32)
                    && A.size(1) == B.size(1),
                "expected A[M,K] with M in {16,32} and matching B K");
    TORCH_CHECK(B.size(0) >= 128 && B.size(0) % 128 == 0,
                "B N must be positive and divisible by 128");
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

    const int m = (int)A.size(0);
    const int n = (int)B.size(0);
    const int k_blocks = (int)(A.size(1) / 128);
    TORCH_CHECK(A_scale.dim() == 2 && A_scale.size(0) == m
                    && A_scale.size(1) == k_blocks,
                "A_scale must have shape [M,K/128]");
    TORCH_CHECK(B_scale.dim() == 2 && B_scale.size(0) == n / 128
                    && B_scale.size(1) == k_blocks,
                "B_scale must have shape [N/128,K/128]");

    c10::cuda::CUDAGuard device_guard(A.device());
    auto D = at::empty({m, n}, A.options().dtype(at::ScalarType::BFloat16));
    const int n_tiles = n / 64;
    globals g{
        kittens::py::tensor_to_gl<a_gl>(A),
        kittens::py::tensor_to_gl<b_gl>(B),
        kittens::py::tensor_to_gl<d_gl>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        k_blocks,
        n_tiles,
    };
    constexpr int SMEM = sizeof(a_st) + sizeof(b_st) + sizeof(d_st) + 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    kernel<<<m / 16 * n_tiles, 32, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    return D;
}

} // namespace mok_sm90::fp8_block_tail_test
#endif
