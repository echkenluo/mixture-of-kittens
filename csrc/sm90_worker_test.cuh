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
using b_st = st_bf<64, 64>;
using d_st = st_bf<128, 128>;
using a_gl = gl<bf16, 1, 1, -1, -1, a_st>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;

struct globals { a_gl A; b_gl B; d_gl D; int k_chunks; };

template<bool IS_AB>
__global__ __launch_bounds__(128, 1) void kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[];
    shared_allocator al((int*)&__shm[0]);
    a_st &a_smem = al.allocate<a_st>();
    b_st &b_smem0 = al.allocate<b_st>();
    b_st &b_smem1 = al.allocate<b_st>();
    d_st &d_smem = al.allocate<d_st>();
    using a_half = st_bf<64, 64>;
    wgmma_quad<a_half, b_st, IS_AB> acc;
    for (int k = 0; k < g.k_chunks; ++k) {
        warpgroup::load(a_smem, g.A, {0, k});
        if constexpr (IS_AB) {
            warpgroup::load(b_smem0, g.B, {k, 0}); // B [K,N]
            warpgroup::load(b_smem1, g.B, {k, 1});
        } else {
            warpgroup::load(b_smem0, g.B, {0, k}); // B [N,K]
            warpgroup::load(b_smem1, g.B, {1, k});
        }
        warpgroup::sync(0);
        auto &a0 = *reinterpret_cast<a_half *>(&a_smem);
        auto &a1 = *reinterpret_cast<a_half *>(
            reinterpret_cast<char *>(&a_smem) + sizeof(a_half));
        acc.step(a0, a1, b_smem0, b_smem1, k == 0);
        warpgroup::sync(0);
    }
    acc.drain_to(d_smem);
    warpgroup::sync(0);
    warpgroup::store(g.D, d_smem, {0, 0});
}

inline at::Tensor entry(at::Tensor A, at::Tensor B, bool is_ab) {
    if (is_ab) { TORCH_CHECK(A.size(0) == 128 && B.size(1) == 128 && A.size(1) == B.size(0)); }
    else       { TORCH_CHECK(A.size(0) == 128 && B.size(0) == 128 && A.size(1) == B.size(1)); }
    TORCH_CHECK(A.size(1) % 64 == 0);
    auto D = at::empty({128, 128}, A.options());
    globals g{
        a_gl{reinterpret_cast<kittens::bf16*>(A.data_ptr()), nullptr, nullptr, (int)A.size(0), (int)A.size(1)},
        b_gl{reinterpret_cast<kittens::bf16*>(B.data_ptr()), nullptr, nullptr, (int)B.size(0), (int)B.size(1)},
        d_gl{reinterpret_cast<kittens::bf16*>(D.data_ptr()), nullptr, nullptr, 128, 128},
        (int)(A.size(1) / 64)};
    constexpr int SMEM = sizeof(a_st) + 2 * sizeof(b_st) + sizeof(d_st) + 1024;
    if (is_ab) {
        cudaFuncSetAttribute(kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        kernel<true><<<1, 128, SMEM>>>(g);
    } else {
        cudaFuncSetAttribute(kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        kernel<false><<<1, 128, SMEM>>>(g);
    }
    return D;
}
} // namespace wtest
} // namespace mok_sm90
#endif
