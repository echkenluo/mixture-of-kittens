#pragma once
// Standalone numeric test for the SM90 wgmma worker (review step 2):
// C[128,128] = A[128,K] x B[K,128], all bf16 in/out, fp32 accumulate.
// Single CTA, one consumer warpgroup, cooperative smem loads (no TMA, no
// semaphores) - isolates the accumulate + drain math from the megakernel.
#if defined(KITTENS_SM90)
#include <ATen/ATen.h>
namespace mok_sm90 {
using namespace kittens;

namespace wtest {
using a_st = st_bf<128, 64>;
using b_st = st_bf<64, 128>;
using d_st = st_bf<128, 128>;
using a_gl = gl<bf16, 1, 1, -1, -1, a_st>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;

struct globals { a_gl A; b_gl B; d_gl D; int k_chunks; };

__global__ __launch_bounds__(128, 1) void kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    shared_allocator al((int*)&__shm[0]);
    a_st &a_smem = al.allocate<a_st>();
    b_st &b_smem = al.allocate<b_st>();
    d_st &d_smem = al.allocate<d_st>();
    wgmma_acc<a_st, b_st, 128, 256> acc; // NB=256 -> acc cols 128 (N-half legacy param)
    for (int k = 0; k < g.k_chunks; ++k) {
        warpgroup::load(a_smem, g.A, {0, k});
        warpgroup::load(b_smem, g.B, {k, 0});
        warpgroup::sync(0);
        acc.step(a_smem, b_smem, k == 0);
        warpgroup::sync(0);
    }
    acc.drain_to(d_smem);
    warpgroup::sync(0);
    warpgroup::store(g.D, d_smem, {0, 0});
}

inline at::Tensor entry(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.size(0) == 128 && B.size(1) == 128 && A.size(1) == B.size(0));
    TORCH_CHECK(A.size(1) % 64 == 0);
    auto D = at::empty({128, 128}, A.options());
    globals g{
        a_gl{reinterpret_cast<bf16*>(A.data_ptr()), 1, 1, (int)A.size(0), (int)A.size(1)},
        b_gl{reinterpret_cast<bf16*>(B.data_ptr()), 1, 1, (int)B.size(0), (int)B.size(1)},
        d_gl{reinterpret_cast<bf16*>(D.data_ptr()), 1, 1, 128, 128},
        (int)(A.size(1) / 64)};
    constexpr int SMEM = sizeof(a_st) + sizeof(b_st) + sizeof(d_st) + 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    kernel<<<1, 128, SMEM>>>(g);
    return D;
}
} // namespace wtest
} // namespace mok_sm90
#endif
