#pragma once

// Warp-role FP8 block GEMM core for H20 (SM90), step 1 of the warprole megakernel.
//
// One CTA per SM.  Warpgroups:  [0, NC) consumers, NC producer, NC+1 comm slot.
// The producer's warp 0 streams A (M64 x K128), B (N128 x K128 per consumer)
// and the 64 activation scales of one K128 block into a STAGES-deep ring with
// TMA; each consumer computes a M64 x N128 WGMMA partial per K128 block,
// promotes it with A_scale[row] * B_scale[block] using one FP32 FMA per later
// block. This intentionally differs from split's separate multiply and add:
// rounding the product first can cross a BF16 boundary relative to DeepGEMM.
//
// The standalone grouped kernel below launches with the same thread count as
// the composed kernel (the comm slot immediately hands back its registers) so
// the register economy measured here is the one the megakernel will have.
#if defined(KITTENS_SM90)
#include <ATen/ATen.h>

#include <algorithm>

#include "sm90_fp8_block_pipeline_primitives.cuh"
#include "sm90_fp8_block_warprole_config.cuh"

namespace mok_sm90::warprole::gemm {
using namespace kittens;

// The c4 screen keeps N256 per CTA while giving four independent consumers
// N64 each. Existing c1/c2 types and arithmetic remain N128.
template <int NC> constexpr int consumer_n() { return NC == 4 ? 64 : N_TILE; }
template <int NT> using b_st_for = st_fp8e4m3<NT, K_TILE>;
template <int NT> using d_st_for = st_bf<M_TILE, NT>;
template <int NT> using acc_rt_for = rt_fl<16, NT>;
template <int NT> using b_gl_for = gl<fp8e4m3, 1, -1, -1, -1, b_st_for<NT>>;
template <int NT> using d_gl_for = gl<bf16, 1, 1, -1, -1, d_st_for<NT>>;

using a_st = st_fp8e4m3<M_TILE, K_TILE>;   // 8 KB
using b_st = st_fp8e4m3<N_TILE, K_TILE>;   // 16 KB
using d_st = st_bf<M_TILE, N_TILE>;        // 16 KB
using acc_rt = rt_fl<16, N_TILE>;          // per warp 16 rows x 128 cols
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_st>;   // [M, K]
using b_gl = gl<fp8e4m3, 1, -1, -1, -1, b_st>;  // [E, N, K]
using d_gl = gl<bf16, 1, 1, -1, -1, d_st>;      // [M, N]

template <int NC> struct stage_smem {
    a_st a;
    b_st_for<consumer_n<NC>()> b[NC];
    float a_scale[M_TILE];   // activation scale of this K128 block, one per row
};
template <int NC, int STAGES> struct smem_layout {
    stage_smem<NC> stage[STAGES];
    d_st_for<consumer_n<NC>()> d[NC > 2 ? NC : 2]; // c1/c2 retain their gate/up pair
    float b_scale[NC > 2 ? NC : 2][W13_K_BLOCKS];   // weight block-scale rows of the task in flight (K/128 <= 32)
};

// 56 (not 40): warps 1-3 of the producer warpgroup run the fused W13 epilogue in
// the fused kernel and need the room; 2*128*192 + 128*56 + 128*64 = 64512 <= 64K.
constexpr int PRODUCER_REGS = 56;
constexpr int COMM_REGS = 64;
template <int NC> constexpr int num_threads() { return 128 * (NC + 2); }
template <int NC, int CTAS_PER_SM> constexpr int consumer_regs() {
    static_assert((NC == 1 && CTAS_PER_SM == 1) || (NC == 2 && CTAS_PER_SM == 1)
                  || (NC == 1 && CTAS_PER_SM == 2) || (NC == 4 && CTAS_PER_SM == 1),
                  "supported forms: (1,1) (2,1) (1,2) (4,1)");
    return NC == 4 ? 96 : ((NC == 1 && CTAS_PER_SM == 1) ? 232 : (NC == 2 ? 192 : 184));
}
template <int CTAS_PER_SM> constexpr int producer_regs() { return CTAS_PER_SM == 1 ? PRODUCER_REGS : 24; }
template <int CTAS_PER_SM> constexpr int comm_regs() { return CTAS_PER_SM == 1 ? COMM_REGS : 24; }
static_assert(4 * 128 * consumer_regs<4, 1>() + 128 * PRODUCER_REGS + 128 * COMM_REGS == 64512);

__device__ __forceinline__ void fence_async_proxy_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Producer side of one task: executed by every lane of warp 0 of the producer
// warpgroup.  `stage_counter` runs across tasks so slot and phase stay in step
// with the consumers, which advance the same counter.  Consumer c reads B tile
// `n_tile_base + c * n_tile_stride`: adjacent tiles for a plain GEMM, the gate
// and up tiles of one intermediate block for the W13 task.
template <int NC, int STAGES, typename A_GL, typename B_GL>
__device__ __forceinline__ void producer_task(
        const A_GL &A, const float *A_scale, const B_GL &B,
        smem_layout<NC, STAGES> &smem, semaphore (&full)[STAGES], semaphore (&empty)[STAGES],
        int64_t &stage_counter, int m_tile, int expert, int n_tile_base, int k_blocks,
        int n_tile_stride = 1) {
    const int lane = laneid();
    const float *scale_rows = A_scale + static_cast<size_t>(m_tile) * M_TILE * k_blocks;
    for (int kb = 0; kb < k_blocks; ++kb, ++stage_counter) {
        const int s = static_cast<int>(stage_counter % STAGES);
        const int phase = static_cast<int>((stage_counter / STAGES) & 1);
        wait(empty[s], phase ^ 1);   // a fresh ring passes immediately
        if (lane == 0) {
            tma::expect_bytes(full[s], sizeof(a_st) + sizeof(smem.stage[s].b));
            tma::load_async(smem.stage[s].a, A, {m_tile, kb}, full[s]);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                tma::load_async(smem.stage[s].b[c], B, {expert, n_tile_base + c * n_tile_stride, kb}, full[s]);
        }
        smem.stage[s].a_scale[lane] = scale_rows[static_cast<size_t>(lane) * k_blocks + kb];
        smem.stage[s].a_scale[lane + 32] = scale_rows[static_cast<size_t>(lane + 32) * k_blocks + kb];
        __syncwarp();
        if (lane == 0) {
            fence_async_proxy_shared();
            arrive(full[s]);
        }
    }
}

// Promote the unscaled Tensor Core partial with one rounding. Match TK's
// row-major row_map layout: even packed slots hold the top row, odd slots
// the row eight positions below; both float2 lanes use that row's scale.
// Keep the FMA explicit so compiler contraction choices cannot alter it.
template <typename ACC>
__device__ __forceinline__ void promote_fma(
        ACC &total, const ACC &partial, const typename ACC::col_vec &row_scale) {
#pragma unroll
    for (int i = 0; i < ACC::height; ++i) {
#pragma unroll
        for (int j = 0; j < ACC::width; ++j) {
#pragma unroll
            for (int k = 0; k < ACC::packed_per_tile; ++k) {
                float2 &t = total.tiles[i][j].data[k];
                const float2 &q = partial.tiles[i][j].data[k];
                const float scale = (k & 1) ? row_scale[i][0].y : row_scale[i][0].x;
                t.x = __fmaf_rn(q.x, scale, t.x);
                t.y = __fmaf_rn(q.y, scale, t.y);
            }
        }
    }
}

// Consumer side of one task: M64 x consumer_n<NC>() FP32 result in `total`.
// W13, W2 and the standalone primitive share this promotion order.
// `b_slot` selects the B tile of each stage, `b_scale_row` the staged block-scale row.
template <int NC, int STAGES>
__device__ __forceinline__ void consumer_task(
        smem_layout<NC, STAGES> &smem, semaphore (&full)[STAGES], semaphore (&empty)[STAGES],
        int64_t &stage_counter, int b_slot, const float *b_scale_row, int k_blocks, acc_rt_for<consumer_n<NC>()> &total) {
    using acc_t = acc_rt_for<consumer_n<NC>()>;
    const int local_row = warpgroup::warpid() * 16 + laneid() / 4;
    // One K128 block: unscaled WGMMA partial into `dst`, plus rounded row scales.
    // The first block is peeled so that `total` is never the target of a copy-or-add
    // join inside the loop: with `if (kb == 0) copy else add` in the loop body ptxas
    // coalesced `total` with the WGMMA accumulator in one of the two inlined loops of
    // the fused kernel and shuffled/spilled 27 to 64 registers per iteration.
    auto block = [&](int kb, acc_t &dst, typename acc_t::col_vec &row_scale) {
        const int s = static_cast<int>(stage_counter % STAGES);
        const int phase = static_cast<int>((stage_counter / STAGES) & 1);
        wait(full[s], phase);
        warpgroup::mm_ABt(dst, smem.stage[s].a, smem.stage[s].b[b_slot]);
        warpgroup::mma_async_wait<0>();
        const float b_scale = b_scale_row[kb];
        row_scale[0][0].x = __fmul_rn(smem.stage[s].a_scale[local_row], b_scale);
        row_scale[0][0].y = __fmul_rn(smem.stage[s].a_scale[local_row + 8], b_scale);
        if (laneid() == 0) arrive(empty[s]);   // this warp's WGMMA reads of slot s are complete
        ++stage_counter;
    };
    typename acc_t::col_vec first_scale;
    block(0, total, first_scale);
    warpgroup::mul_row(total, total, first_scale);  // no preceding sum for block 0
#pragma unroll 1
    for (int kb = 1; kb < k_blocks; ++kb) {
        acc_t partial;
        typename acc_t::col_vec row_scale;
        block(kb, partial, row_scale);
        promote_fma(total, partial, row_scale);
    }
}

// Stage the B block-scale row of one (expert, n_tile) task into shared memory.
// Called by all 128 threads of one consumer; followed by a warpgroup barrier by the caller.
template <int NC, int STAGES>
__device__ __forceinline__ void stage_b_scale_row(
        smem_layout<NC, STAGES> &smem, int slot, const float *B_scale, int expert, int n_tiles_128,
        int n_tile, int k_blocks) {
    const int t = warpgroup::laneid();
    if (t < k_blocks)
        smem.b_scale[slot][t] = B_scale[(static_cast<size_t>(expert) * n_tiles_128 + n_tile) * k_blocks + t];
}

// Write a finished M64 x N64/N128 tile through the consumer's own
// staging tile with a TMA store.  All 128 threads of consumer `c` call this.
template <typename D_ST, typename D_GL, typename ACC>
__device__ __forceinline__ void store_bf16_tile_via(
        D_ST &staging, int barrier_id, const D_GL &D, const ACC &total, int m_tile, int n_tile) {
    if (warpgroup::laneid() == 0) tma::store_async_read_wait();   // the previous store has read staging
    warpgroup::sync(barrier_id);
    rt_bf<16, ACC::cols> out;
    warp::copy(out, total);
    warpgroup::store(staging, out);
    fence_async_proxy_shared();          // make generic-proxy smem writes visible to the TMA unit
    warpgroup::sync(barrier_id);
    if (warpgroup::laneid() == 0) {
        tma::store_async(D, staging, {m_tile, n_tile});
        tma::store_commit_group();
    }
}
template <int NC, int STAGES, typename D_GL>
__device__ __forceinline__ void store_bf16_tile(
        smem_layout<NC, STAGES> &smem, int c, int barrier_id, const D_GL &D, const acc_rt_for<consumer_n<NC>()> &total,
        int m_tile, int n_tile) {
    store_bf16_tile_via(smem.d[c], barrier_id, D, total, m_tile, n_tile);
}

// ---------------------------------------------------------------------------
// Standalone grouped-contiguous kernel (step 1 microbenchmark).
// Tasks: (m_tile, n_group) covering NC adjacent consumer_n<NC>()-wide tiles.
// ---------------------------------------------------------------------------
namespace standalone {

template <int NT> struct globals_for {
    a_gl A;
    b_gl_for<NT> B;
    d_gl_for<NT> D;
    const float *A_scale;
    const float *B_scale;
    const int *m_indices;
    const int *num_tokens;
    int n;
    int k_blocks;
    int m_tiles;
};
using globals = globals_for<N_TILE>; // existing communication benchmark uses N128

// Producer + consumer roles of the standalone grouped GEMM over the static
// task stream (m_tile, n_group).  `role` is the caller's warpgroup index; the
// comm slot (role NC+1) must not call this.
template <int NC, int STAGES, int CTAS_PER_SM>
__device__ __forceinline__ void run_gemm_roles(
        const globals_for<consumer_n<NC>()> &g, smem_layout<NC, STAGES> &smem, semaphore (&full)[STAGES],
        semaphore (&empty)[STAGES], int role) {
    const int n_tasks_per_m = g.n / (consumer_n<NC>() * NC);
    const int64_t total = static_cast<int64_t>(g.m_tiles) * n_tasks_per_m;
    int64_t stage_counter = 0;
    if (role == NC) {
        warpgroup::decrease_registers<producer_regs<CTAS_PER_SM>()>();
        if (warpgroup::warpid() != 0) return;
        for (int64_t t = blockIdx.x; t < total; t += gridDim.x) {
            const int m_tile = static_cast<int>(t / n_tasks_per_m);
            if (g.num_tokens != nullptr && m_tile * M_TILE >= g.num_tokens[0]) break;
            const int n_base = static_cast<int>(t % n_tasks_per_m) * NC;
            const int expert = g.m_indices[m_tile * M_TILE];
            producer_task<NC, STAGES>(g.A, g.A_scale, g.B, smem, full, empty, stage_counter,
                                      m_tile, expert, n_base, g.k_blocks);
        }
        return;
    }
    warpgroup::increase_registers<consumer_regs<NC, CTAS_PER_SM>()>();
    const int barrier_id = role + 1;
    const int n_tiles_128 = g.n / N_TILE;
    for (int64_t t = blockIdx.x; t < total; t += gridDim.x) {
        const int m_tile = static_cast<int>(t / n_tasks_per_m);
        if (g.num_tokens != nullptr && m_tile * M_TILE >= g.num_tokens[0]) break;
        const int n_tile = static_cast<int>(t % n_tasks_per_m) * NC + role;
        const int expert = g.m_indices[m_tile * M_TILE];
        stage_b_scale_row<NC, STAGES>(smem, role, g.B_scale, expert, n_tiles_128, n_tile * consumer_n<NC>() / N_TILE, g.k_blocks);
        warpgroup::sync(barrier_id);
        acc_rt_for<consumer_n<NC>()> total_acc;
        consumer_task<NC, STAGES>(smem, full, empty, stage_counter, role, smem.b_scale[role], g.k_blocks, total_acc);
        store_bf16_tile<NC, STAGES>(smem, role, barrier_id, g.D, total_acc, m_tile, n_tile);
    }
    if (warpgroup::laneid() == 0) tma::store_async_wait();
}

template <int NC, int STAGES>
__device__ __forceinline__ void init_ring(semaphore (&full)[STAGES], semaphore (&empty)[STAGES]) {
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_semaphore(full[s], 1, 1);       // one producer arrive + one TMA transaction group
            init_semaphore(empty[s], NC * 4, 0); // one arrive per consumer warp
        }
    }
    __syncthreads();
}

template <int NC, int STAGES, int CTAS_PER_SM>
__global__ __launch_bounds__(num_threads<NC>(), CTAS_PER_SM)
void grouped_kernel(const __grid_constant__ globals_for<consumer_n<NC>()> g) {
    extern __shared__ int __shm[];
    auto &smem = *reinterpret_cast<smem_layout<NC, STAGES> *>(
        ((reinterpret_cast<uint64_t>(&__shm[0])) + 1023) & ~static_cast<uint64_t>(1023));
    __shared__ semaphore full[STAGES];
    __shared__ semaphore empty[STAGES];
    init_ring<NC, STAGES>(full, empty);
    const int role = warpgroup::groupid();   // 0..NC-1 consumers, NC producer, NC+1 comm slot
    if (role == NC + 1) {
        warpgroup::decrease_registers<comm_regs<CTAS_PER_SM>()>();
        return;   // standalone GEMM: the comm slot only gives its registers back
    }
    run_gemm_roles<NC, STAGES, CTAS_PER_SM>(g, smem, full, empty, role);
}

template <int NC, int STAGES, int CTAS_PER_SM>
inline at::Tensor entry_out_impl(at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale,
                                 at::Tensor m_indices, const at::Tensor *num_tokens, at::Tensor D) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 3, "A and B must have shapes [M,K] and [E,N,K]");
    kittens::py::tensor_check<a_gl>(A);
    kittens::py::tensor_check<b_gl_for<consumer_n<NC>()>>(B);
    const int total_m = (int)A.size(0);
    const int experts = (int)B.size(0);
    const int n = (int)B.size(1);
    const int k = (int)A.size(1);
    TORCH_CHECK(experts > 0 && B.size(2) == k, "A and B K dimensions must match");
    TORCH_CHECK(total_m >= 64 && total_m % 64 == 0, "M must be positive and divisible by 64");
    TORCH_CHECK(n >= consumer_n<NC>() * NC && n % (consumer_n<NC>() * NC) == 0,
                "N must be positive and divisible by the per-CTA column coverage");
    TORCH_CHECK(k >= 128 && k % 128 == 0 && k / 128 <= W13_K_BLOCKS,
                "K must be a positive multiple of 128 and at most 4096");
    TORCH_CHECK(A_scale.is_cuda() && B_scale.is_cuda() && m_indices.is_cuda(),
                "scales and m_indices must be CUDA tensors");
    TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Float
                    && B_scale.scalar_type() == at::ScalarType::Float,
                "A_scale and B_scale must be float32");
    TORCH_CHECK(m_indices.scalar_type() == at::ScalarType::Int, "m_indices must be int32");
    TORCH_CHECK(A_scale.is_contiguous() && B_scale.is_contiguous() && m_indices.is_contiguous(),
                "scales and m_indices must be contiguous");
    const int k_blocks = k / 128;
    TORCH_CHECK(A_scale.dim() == 2 && A_scale.size(0) == total_m && A_scale.size(1) == k_blocks,
                "A_scale must have shape [M,K/128]");
    TORCH_CHECK(B_scale.dim() == 3 && B_scale.size(0) == experts && B_scale.size(1) == n / 128
                    && B_scale.size(2) == k_blocks,
                "B_scale must have shape [E,N/128,K/128]");
    TORCH_CHECK(m_indices.dim() == 1 && m_indices.size(0) == total_m, "m_indices must have shape [M]");
    TORCH_CHECK(D.dim() == 2 && D.size(0) == total_m && D.size(1) == n, "D must have shape [M,N]");
    TORCH_CHECK(D.is_cuda() && D.scalar_type() == at::ScalarType::BFloat16,
                "D must be a CUDA bfloat16 tensor");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    kittens::py::tensor_check<d_gl_for<consumer_n<NC>()>>(D);
    kittens::py::device_check(A, B, A_scale, B_scale, m_indices);
    kittens::py::device_check(A, D);
    if (num_tokens != nullptr) {
        TORCH_CHECK(num_tokens->is_cuda() && num_tokens->scalar_type() == at::ScalarType::Int
                        && num_tokens->is_contiguous() && num_tokens->dim() == 1
                        && num_tokens->numel() == 1,
                    "num_tokens must be contiguous CUDA int32 [1]");
        TORCH_CHECK(num_tokens->device() == A.device(), "num_tokens must share the input device");
    }

    c10::cuda::CUDAGuard device_guard(A.device());
    const int m_tiles = total_m / 64;
    globals_for<consumer_n<NC>()> g{
        kittens::py::tensor_to_gl<a_gl>(A),
        kittens::py::tensor_to_gl<b_gl_for<consumer_n<NC>()>>(B),
        kittens::py::tensor_to_gl<d_gl_for<consumer_n<NC>()>>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        m_indices.data_ptr<int>(),
        num_tokens == nullptr ? nullptr : num_tokens->data_ptr<int>(),
        n,
        k_blocks,
        m_tiles,
    };
    constexpr int SMEM = sizeof(smem_layout<NC, STAGES>) + 1024;
    constexpr int THREADS = num_threads<NC>();
    auto *kernel_ptr = grouped_kernel<NC, STAGES, CTAS_PER_SM>;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    CUDACHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int num_sms = 0;
    CUDACHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, A.get_device()));
    int blocks_per_sm = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr, THREADS, SMEM));
    TORCH_CHECK(blocks_per_sm >= CTAS_PER_SM,
                "warprole gemm occupancy below target: ", blocks_per_sm, " < ", CTAS_PER_SM,
                " CTAs per SM (smem ", SMEM, " bytes, ", THREADS, " threads)");
    const int64_t total_tasks = static_cast<int64_t>(m_tiles) * (n / (consumer_n<NC>() * NC));
    const int grid = static_cast<int>(std::min<int64_t>(total_tasks, static_cast<int64_t>(num_sms) * CTAS_PER_SM));
    kernel_ptr<<<grid, THREADS, SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    return D;
}

template <int NC, int STAGES, int CTAS_PER_SM>
inline at::Tensor entry_out(at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale,
                            at::Tensor m_indices, at::Tensor num_tokens, at::Tensor D) {
    return entry_out_impl<NC, STAGES, CTAS_PER_SM>(A, B, A_scale, B_scale, m_indices, &num_tokens, D);
}

}  // namespace standalone
}  // namespace mok_sm90::warprole::gemm
#endif  // KITTENS_SM90
