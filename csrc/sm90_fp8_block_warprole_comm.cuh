#pragma once

// Communication warpgroup of the warprole kernel (step 2).
//
// One warpgroup per CTA pulls dispatch rows from peers (8-row tickets, two
// rows per warp, register staged) and pushes finished BF16 rows to peers,
// publishing progress through three counters:
//   x_ready[q]        rows of minibatch q landed in routed_x/routed_x_scale
//   y_ready[m]        W2 tasks of M64 tile m finished (written by consumers)
//   push_done         ranks whose pushes into this rank are complete (symmetric)
// The row copy cores come from fp8_block_terminal_comm; this file owns only the
// row assignment, fences and counter publication.
#if defined(KITTENS_SM90)
#include <ATen/ATen.h>

#include <vector>

#include "sm90_fp8_block_terminal_comm_primitives.cuh"
#include "sm90_fp8_block_warprole_gemm.cuh"

namespace mok_sm90::warprole::comm {
using namespace kittens;
namespace tc = mok_sm90::fp8_block_terminal_comm;

constexpr int MAX_EP_SIZE = 8;

// Fields consumed by the terminal_comm row cores plus the warprole counters.
struct globals {
    const uint8_t *x_peer[MAX_EP_SIZE];
    const float *x_scale_peer[MAX_EP_SIZE];
    uint8_t *routed_x;
    float *routed_x_scale;
    int *m_indices;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    const int *tokens_per_expert;
    int ep_size;
    int ep_rank;
    int num_local_tokens;
    int hidden_size;
    int scale_columns;
    int topk;
    int num_local_experts;
    int schedule_capacity;
    const uint8_t *routed_y;
    uint8_t *combine_peer[MAX_EP_SIZE];
    unsigned int *x_ready;                       // [minibatches]
    unsigned int *y_ready;                       // [m_tiles]
    unsigned int *push_done_local;               // [1], CTAs of this rank done pushing
    unsigned int *push_done_peer[MAX_EP_SIZE];   // symmetric: each rank's arrival counter
    unsigned int *push_done;                     // this rank's own arrival counter
};

__device__ __forceinline__ unsigned int load_acquire_gpu(const unsigned int *ptr) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}
__device__ __forceinline__ unsigned int load_acquire_sys(const unsigned int *ptr) {
    unsigned int v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}
__device__ __forceinline__ void wait_counter_geq(const unsigned int *ctr, unsigned int target) {
    while (load_acquire_gpu(ctr) < target) __nanosleep(256);
}
__device__ __forceinline__ void wait_counter_geq_sys(const unsigned int *ctr, unsigned int target) {
    while (load_acquire_sys(ctr) < target) __nanosleep(256);
}
__device__ __forceinline__ void add_release_gpu(unsigned int *ctr, unsigned int value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ctr), "r"(value) : "memory");
}
__device__ __forceinline__ void add_release_sys(unsigned int *ctr, unsigned int value) {
    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(ctr), "r"(value) : "memory");
}

// Number of routed rows this launch works on: M64 aligned by the schedule.
template <typename G> __device__ __forceinline__ shape active_shape(const G &g) {
    return shape{tc::bounded_valid_rows(g)};
}

// Dispatch every ticket of minibatch q that this CTA owns (t = cta, cta+grid, ...).
// Whole comm warpgroup; each warp copies two rows per ticket and publishes its
// own row count, so x_ready[q] reaches rows_in_minibatch(q) when all rows landed.
template <typename G>
__device__ __forceinline__ void dispatch_minibatch(const G &g, shape s, int q, const int *expert_row_end) {
    const int warp = warpgroup::warpid();
    const int lane = laneid();
    const int tickets = dispatch_tickets(s, q);
    unsigned int rows_done = 0;
    for (int t = blockIdx.x; t < tickets; t += gridDim.x) {
        const int first = ticket_first_row(s, q, t);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tc::dispatch_copy_row(g, expert_row_end, first + warp * 2 + r, lane);
        }
        rows_done += 2;
    }
    __syncwarp();
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if (lane == 0 && rows_done != 0) add_release_gpu(g.x_ready + q, rows_done);
}

// Push the 64 BF16 rows of M64 tile m to their peers: 16 rows per warp.
template <typename G>
__device__ __forceinline__ void combine_tile(const G &g, int m_tile) {
    const int warp = warpgroup::warpid();
    const int lane = laneid();
#pragma unroll 1
    for (int r = 0; r < 16; ++r)
        tc::push_routed_row(g, m_tile * M_TILE + warp * 16 + r, lane);
}

// Called by the whole comm warpgroup once its combine tiles are pushed.  The
// last CTA of this rank tells every rank (itself included) that this rank's
// pushes are complete.  Fence pattern copied from fp8_block_routed::combine_kernel.
template <typename G>
__device__ __forceinline__ void combine_finish(const G &g) {
    warpgroup::sync(BAR_COMM);
    if (warpgroup::laneid() == 0) {
        asm volatile("{fence.release.sys;}" ::: "memory");
        const unsigned int prior = atomicAdd(g.push_done_local, 1u);
        if (prior + 1u == gridDim.x) {
            asm volatile("{fence.acquire.sys;}" ::: "memory");
            *g.push_done_local = 0u;   // ready for the next launch
            for (int r = 0; r < g.ep_size; ++r) add_release_sys(g.push_done_peer[r], 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 2 microbenchmark kernel.  MODE 0: dispatch only.  MODE 1: combine only
// (y_ready pre-filled by the host).  MODE 2: dispatch while the producer and
// consumer run the standalone grouped GEMM over an independent problem.
// ---------------------------------------------------------------------------
namespace bench {

constexpr int NC = 1, STAGES = 6, CTAS_PER_SM = 1;
using gemm_smem = gemm::smem_layout<NC, STAGES>;

// kittens::gl has no default constructor, so the GEMM problem only travels in
// the mode-2 parameter struct and is built in place from tensors.
struct globals_plain {
    comm::globals c;
};
struct globals_with_gemm {
    comm::globals c;
    gemm::standalone::globals g;
    __host__ globals_with_gemm(const comm::globals &cc, const gemm::standalone::globals &gg) : c(cc), g(gg) {}
};

template <int MODE, typename P>
__global__ __launch_bounds__(gemm::num_threads<NC>(), CTAS_PER_SM)
void comm_bench_kernel(const __grid_constant__ P p) {
    extern __shared__ int __shm[];
    auto &smem = *reinterpret_cast<gemm_smem *>(
        ((reinterpret_cast<uint64_t>(&__shm[0])) + 1023) & ~static_cast<uint64_t>(1023));
    __shared__ semaphore full[STAGES];
    __shared__ semaphore empty[STAGES];
    __shared__ int expert_row_end[256];
    if (threadIdx.x == 0) tc::build_expert_row_ends(p.c, expert_row_end);
    gemm::standalone::init_ring<NC, STAGES>(full, empty);   // ends with __syncthreads
    const int role = warpgroup::groupid();
    if (role == NC + 1) {
        warpgroup::decrease_registers<gemm::comm_regs<CTAS_PER_SM, NC>()>();
        const shape s = active_shape(p.c);
        if constexpr (MODE == 0 || MODE == 2) {
            for (int q = 0; q < minibatches(s); ++q) dispatch_minibatch(p.c, s, q, expert_row_end);
        } else {
            const int m_tiles = s.num_rows / M_TILE;
            for (int m = blockIdx.x; m < m_tiles; m += gridDim.x) {
                wait_counter_geq(p.c.y_ready + m, y_ready_target<NC>());
                combine_tile(p.c, m);
            }
            combine_finish(p.c);
        }
        return;
    }
    if constexpr (MODE == 2) {
        gemm::standalone::run_gemm_roles<NC, STAGES, CTAS_PER_SM>(p.g, smem, full, empty, role);
    } else {
        if (role == NC) warpgroup::decrease_registers<gemm::producer_regs<CTAS_PER_SM>()>();
        else warpgroup::increase_registers<gemm::consumer_regs<NC, CTAS_PER_SM>()>();
    }
}

inline void fill_comm_globals(
        comm::globals &c, const at::Tensor &x, const std::vector<int64_t> &x_ptrs,
        const at::Tensor &x_scale, const std::vector<int64_t> &x_scale_ptrs,
        const at::Tensor &routed_x, const at::Tensor &routed_x_scale, const at::Tensor &m_indices,
        const at::Tensor &schedule_peer_rank, const at::Tensor &schedule_peer_token_idx,
        const at::Tensor &num_tokens, const at::Tensor &tokens_per_expert, int64_t topk,
        const at::Tensor &routed_y, const std::vector<int64_t> &combine_ptrs,
        const std::vector<int64_t> &push_done_ptrs, int64_t ep_rank,
        const at::Tensor &x_ready, const at::Tensor &y_ready, const at::Tensor &push_done_local) {
    TORCH_CHECK(x.dim() == 2 && x.is_cuda() && x.scalar_type() == at::kFloat8_e4m3fn && x.is_contiguous(),
                "x must be contiguous CUDA float8_e4m3fn [T,H]");
    const int64_t num_local_tokens = x.size(0);
    const int64_t hidden_size = x.size(1);
    TORCH_CHECK(hidden_size == HIDDEN, "hidden size must be 4096");
    TORCH_CHECK(x_scale.dim() == 2 && x_scale.is_cuda() && x_scale.scalar_type() == at::kFloat
                    && x_scale.is_contiguous() && x_scale.size(0) == num_local_tokens
                    && x_scale.size(1) == hidden_size / 128,
                "x_scale must be contiguous CUDA float32 [T,H/128]");
    TORCH_CHECK(routed_x.dim() == 2 && routed_x.is_cuda() && routed_x.scalar_type() == at::kFloat8_e4m3fn
                    && routed_x.is_contiguous() && routed_x.size(1) == hidden_size,
                "routed_x must be contiguous CUDA float8_e4m3fn [capacity,H]");
    const int64_t capacity = routed_x.size(0);
    TORCH_CHECK(capacity > 0 && capacity % 64 == 0, "schedule capacity must be positive and M64 aligned");
    TORCH_CHECK(routed_x_scale.dim() == 2 && routed_x_scale.is_cuda() && routed_x_scale.scalar_type() == at::kFloat
                    && routed_x_scale.is_contiguous() && routed_x_scale.size(0) == capacity
                    && routed_x_scale.size(1) == hidden_size / 128,
                "routed_x_scale must be contiguous CUDA float32 [capacity,H/128]");
    TORCH_CHECK(m_indices.dim() == 1 && m_indices.is_cuda() && m_indices.scalar_type() == at::kInt
                    && m_indices.is_contiguous() && m_indices.numel() == capacity,
                "m_indices must be contiguous CUDA int32 [capacity]");
    TORCH_CHECK(routed_y.dim() == 2 && routed_y.is_cuda() && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous() && routed_y.size(0) == capacity && routed_y.size(1) == hidden_size,
                "routed_y must be contiguous CUDA bf16 [capacity,H]");
    for (const at::Tensor *t : {&schedule_peer_rank, &schedule_peer_token_idx, &num_tokens, &tokens_per_expert,
                                &x_ready, &y_ready, &push_done_local}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->dim() == 1, "schedule/counter tensors must be 1-D contiguous CUDA");
    }
    TORCH_CHECK(schedule_peer_rank.scalar_type() == at::kInt && schedule_peer_rank.numel() == capacity
                    && schedule_peer_token_idx.scalar_type() == at::kInt && schedule_peer_token_idx.numel() == capacity,
                "schedule arrays must be int32 [capacity]");
    TORCH_CHECK(num_tokens.scalar_type() == at::kInt && num_tokens.numel() == 1, "num_tokens must be int32 [1]");
    TORCH_CHECK(tokens_per_expert.scalar_type() == at::kInt && tokens_per_expert.numel() > 0
                    && tokens_per_expert.numel() <= 256, "tokens_per_expert must be int32 [1..256]");
    TORCH_CHECK(x_ready.scalar_type() == at::kInt && x_ready.numel() >= (capacity + MINIBATCH_ROWS - 1) / MINIBATCH_ROWS,
                "x_ready must be int32 [minibatches]");
    TORCH_CHECK(y_ready.scalar_type() == at::kInt && y_ready.numel() >= capacity / 64, "y_ready must be int32 [m_tiles]");
    TORCH_CHECK(push_done_local.scalar_type() == at::kInt && push_done_local.numel() == 1, "push_done_local must be int32 [1]");
    const int ep_size = static_cast<int>(x_ptrs.size());
    TORCH_CHECK(ep_size > 0 && ep_size <= MAX_EP_SIZE && x_scale_ptrs.size() == x_ptrs.size()
                    && combine_ptrs.size() == x_ptrs.size() && push_done_ptrs.size() == x_ptrs.size(),
                "pointer lists must have equal length in [1,8]");
    TORCH_CHECK(ep_rank >= 0 && ep_rank < ep_size, "ep_rank out of range");
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    for (int r = 0; r < ep_size; ++r) {
        TORCH_CHECK(x_ptrs[r] != 0 && x_scale_ptrs[r] != 0 && combine_ptrs[r] != 0 && push_done_ptrs[r] != 0,
                    "peer pointers must be non-null");
        c.x_peer[r] = reinterpret_cast<const uint8_t *>(x_ptrs[r]);
        c.x_scale_peer[r] = reinterpret_cast<const float *>(x_scale_ptrs[r]);
        c.combine_peer[r] = reinterpret_cast<uint8_t *>(combine_ptrs[r]);
        c.push_done_peer[r] = reinterpret_cast<unsigned int *>(push_done_ptrs[r]);
    }
    c.routed_x = static_cast<uint8_t *>(routed_x.data_ptr());
    c.routed_x_scale = routed_x_scale.data_ptr<float>();
    c.m_indices = m_indices.data_ptr<int>();
    c.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    c.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    c.num_tokens = num_tokens.data_ptr<int>();
    c.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    c.ep_size = ep_size;
    c.ep_rank = static_cast<int>(ep_rank);
    c.num_local_tokens = static_cast<int>(num_local_tokens);
    c.hidden_size = static_cast<int>(hidden_size);
    c.scale_columns = static_cast<int>(hidden_size / 128);
    c.topk = static_cast<int>(topk);
    c.num_local_experts = static_cast<int>(tokens_per_expert.numel());
    c.schedule_capacity = static_cast<int>(capacity);
    c.routed_y = static_cast<const uint8_t *>(routed_y.data_ptr());
    c.x_ready = reinterpret_cast<unsigned int *>(x_ready.data_ptr<int>());
    c.y_ready = reinterpret_cast<unsigned int *>(y_ready.data_ptr<int>());
    c.push_done_local = reinterpret_cast<unsigned int *>(push_done_local.data_ptr<int>());
    c.push_done = c.push_done_peer[ep_rank];
}

template <int MODE, typename P>
inline void launch(const P &p, const at::Device &device) {
    constexpr int SMEM = sizeof(gemm_smem) + 1024;
    constexpr int THREADS = gemm::num_threads<NC>();
    auto *kernel_ptr = comm_bench_kernel<MODE, P>;
    CUDACHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int num_sms = 0;
    CUDACHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.index()));
    int blocks_per_sm = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr, THREADS, SMEM));
    TORCH_CHECK(blocks_per_sm >= 1, "warprole comm bench kernel does not fit one CTA per SM");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    kernel_ptr<<<num_sms, THREADS, SMEM, stream>>>(p);
    CUDACHECK(cudaGetLastError());
}

// mode 0: dispatch only; mode 1: combine only (y_ready must already hold the target).
inline void entry_comm_bench_out(
        int64_t mode, at::Tensor x, std::vector<int64_t> x_ptrs, at::Tensor x_scale,
        std::vector<int64_t> x_scale_ptrs, at::Tensor routed_x, at::Tensor routed_x_scale,
        at::Tensor m_indices, at::Tensor schedule_peer_rank, at::Tensor schedule_peer_token_idx,
        at::Tensor num_tokens, at::Tensor tokens_per_expert, int64_t topk, at::Tensor routed_y,
        std::vector<int64_t> combine_ptrs, std::vector<int64_t> push_done_ptrs, int64_t ep_rank,
        at::Tensor x_ready, at::Tensor y_ready, at::Tensor push_done_local) {
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (dispatch) or 1 (combine)");
    c10::cuda::CUDAGuard device_guard(x.device());
    globals_plain p{};
    fill_comm_globals(p.c, x, x_ptrs, x_scale, x_scale_ptrs, routed_x, routed_x_scale, m_indices,
                      schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert, topk,
                      routed_y, combine_ptrs, push_done_ptrs, ep_rank, x_ready, y_ready, push_done_local);
    if (mode == 0) launch<0, globals_plain>(p, x.device()); else launch<1, globals_plain>(p, x.device());
}

// mode 2: dispatch by the comm warpgroup while producer/consumer run the grouped GEMM (A,B,...,D).
inline void entry_comm_gemm_bench_out(
        at::Tensor x, std::vector<int64_t> x_ptrs, at::Tensor x_scale,
        std::vector<int64_t> x_scale_ptrs, at::Tensor routed_x, at::Tensor routed_x_scale,
        at::Tensor m_indices, at::Tensor schedule_peer_rank, at::Tensor schedule_peer_token_idx,
        at::Tensor num_tokens, at::Tensor tokens_per_expert, int64_t topk, at::Tensor routed_y,
        std::vector<int64_t> combine_ptrs, std::vector<int64_t> push_done_ptrs, int64_t ep_rank,
        at::Tensor x_ready, at::Tensor y_ready, at::Tensor push_done_local,
        at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale, at::Tensor gemm_m_indices,
        at::Tensor gemm_num_tokens, at::Tensor D) {
    c10::cuda::CUDAGuard device_guard(x.device());
    comm::globals c{};
    fill_comm_globals(c, x, x_ptrs, x_scale, x_scale_ptrs, routed_x, routed_x_scale, m_indices,
                      schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert, topk,
                      routed_y, combine_ptrs, push_done_ptrs, ep_rank, x_ready, y_ready, push_done_local);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 3 && D.dim() == 2 && A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "GEMM operands must be CUDA tensors [M,K], [E,N,K], [M,N]");
    kittens::py::tensor_check<gemm::a_gl>(A);
    kittens::py::tensor_check<gemm::b_gl>(B);
    kittens::py::tensor_check<gemm::d_gl>(D);
    const int n = static_cast<int>(B.size(1));
    const int k = static_cast<int>(A.size(1));
    TORCH_CHECK(A.size(0) % 64 == 0 && n % (128 * NC) == 0 && k % 128 == 0 && k / 128 <= W13_K_BLOCKS
                    && B.size(2) == k && D.size(0) == A.size(0) && D.size(1) == n,
                "GEMM shapes must be M64/N128/K128 aligned and consistent");
    TORCH_CHECK(A_scale.is_cuda() && A_scale.scalar_type() == at::kFloat && A_scale.is_contiguous()
                    && A_scale.dim() == 2 && A_scale.size(0) == A.size(0) && A_scale.size(1) == k / 128,
                "A_scale must be float32 [M,K/128]");
    TORCH_CHECK(B_scale.is_cuda() && B_scale.scalar_type() == at::kFloat && B_scale.is_contiguous()
                    && B_scale.dim() == 3 && B_scale.size(0) == B.size(0) && B_scale.size(1) == n / 128
                    && B_scale.size(2) == k / 128,
                "B_scale must be float32 [E,N/128,K/128]");
    TORCH_CHECK(gemm_m_indices.is_cuda() && gemm_m_indices.scalar_type() == at::kInt && gemm_m_indices.is_contiguous()
                    && gemm_m_indices.numel() == A.size(0), "gemm_m_indices must be int32 [M]");
    TORCH_CHECK(gemm_num_tokens.is_cuda() && gemm_num_tokens.scalar_type() == at::kInt && gemm_num_tokens.numel() == 1,
                "gemm_num_tokens must be int32 [1]");
    const gemm::standalone::globals gg{
        kittens::py::tensor_to_gl<gemm::a_gl>(A),
        kittens::py::tensor_to_gl<gemm::b_gl>(B),
        kittens::py::tensor_to_gl<gemm::d_gl>(D),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        gemm_m_indices.data_ptr<int>(),
        gemm_num_tokens.data_ptr<int>(),
        n,
        k / 128,
        static_cast<int>(A.size(0) / 64),
    };
    const globals_with_gemm p(c, gg);
    launch<2, globals_with_gemm>(p, x.device());
}

}  // namespace bench
}  // namespace mok_sm90::warprole::comm
#endif  // KITTENS_SM90
