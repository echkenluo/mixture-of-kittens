#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "../csrc/sm90_fp8_block_terminal_compute.cuh"
#include "../csrc/sm90_fp8_block_worker_test.cuh"

#if !defined(KITTENS_SM90)
#error "terminal_compute_pipeline_probe requires KITTENS_SM90"
#endif

namespace {

namespace terminal = mok_sm90::fp8_block_terminal;
namespace compute = mok_sm90::fp8_block_terminal_compute;
namespace pipeline = mok_sm90::fp8_block_pipeline;
namespace split = mok_sm90::fp8_block_test::contiguous;

using namespace kittens;

constexpr int kThreads = terminal::THREADS_PER_CTA;
constexpr int kReferenceActivationThreads = pipeline::V4_ACTIVATION_WORKERS;
constexpr unsigned int kStop = ~0u;

using a_gl = split::a_gl;
using b_gl = split::b_gl;
using d_gl = split::d_gl;

struct gemm_problem {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    int n;
    int k_blocks;
    int n_tiles;
};

struct terminal_globals {
    gemm_problem w13;
    gemm_problem w2;
    compute::activation_problem activation;
    compute::readiness ready;
    const int *m_indices;
    const int *num_tokens;
    unsigned int *cursor;
    unsigned int *worker_ticket;
    unsigned int *task_visits;
    unsigned int *errors;
    int schedule_capacity;
    int minibatch_rows;
    int macrobatch_rows;
};

__device__ __forceinline__ unsigned int claim_bounded(
        unsigned int *cursor, unsigned int limit) {
    while (true) {
        const unsigned int current = compute::load_acquire_gpu(cursor);
        if (current >= limit)
            return kStop;
        const unsigned int prior = atomicCAS(cursor, current, current + 1u);
        if (prior == current)
            return current;
    }
}

// Test-only fixed-resident worker.  The launch uses exactly the device-wide
// active-cluster limit reported for this kernel, so every cursor owner is
// resident.  Each cursor ordinal is decoded on device with the committed
// stage-major 65-task/M64 mapping.
__cluster_dims__(2, 1, 1) __launch_bounds__(kThreads, 1)
__global__ void terminal_kernel(const __grid_constant__ terminal_globals g) {
    const int cta_rank = cluster_ctarank();
    const int worker_cluster = clusterIdx().x;
    const terminal::logical_shape shape = terminal::make_logical_shape(
        g.num_tokens[0], g.schedule_capacity,
        g.minibatch_rows, g.macrobatch_rows);
    if (!shape.valid) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            atomicAdd(g.errors, 1u);
        return;
    }

    extern __shared__ int __shm[];
    shared_allocator allocator((int *)&__shm[0]);
    auto &a_smem = allocator.allocate<compute::a_st, compute::PIPE_DEPTH>();
    auto &b_smem = allocator.allocate<compute::b_st, compute::PIPE_DEPTH>();
    compute::d_st &d_smem = allocator.allocate<compute::d_st>();
    __shared__ semaphore inputs_arrived[compute::PIPE_DEPTH];
    __shared__ semaphore inputs_finished[compute::PIPE_DEPTH];
    __shared__ semaphore inputs_ready[compute::PIPE_DEPTH];
    if (threadIdx.x < compute::PIPE_DEPTH) {
        init_semaphore(inputs_arrived[threadIdx.x], 0, 1);
        init_semaphore(inputs_finished[threadIdx.x], 0, 1);
        init_semaphore(inputs_ready[threadIdx.x], 0, 2);
    }
    uint32_t phasebits = 0xFFFF0000u;
    uint32_t ready_phase = 0u;
    everyone::tma::cluster::sync();

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int ticket = claim_bounded(
                g.cursor, static_cast<unsigned int>(shape.total_tasks));
            asm volatile("{st.release.cluster.global.u32 [%0], %1;}" ::
                         "l"(g.worker_ticket + worker_cluster), "r"(ticket)
                         : "memory");
        }
        everyone::tma::cluster::sync();
        unsigned int ticket;
        asm volatile("{ld.acquire.cluster.global.u32 %0, [%1];}"
                     : "=r"(ticket)
                     : "l"(g.worker_ticket + worker_cluster) : "memory");
        if (ticket == kStop)
            break;

        const terminal::logical_coordinate coordinate =
            terminal::decode_logical_cursor(shape, ticket);
        if (!coordinate.valid) {
            if (cta_rank == 0 && threadIdx.x == 0)
                atomicAdd(g.errors, 1u);
            everyone::tma::cluster::sync();
            continue;
        }
        if (cta_rank == 0 && threadIdx.x == 0)
            atomicAdd(g.task_visits + ticket, 1u);
        const int expert = g.m_indices[
            coordinate.global_m * terminal::M_TILE];

        if (coordinate.stage == terminal::logical_stage::gate
                || coordinate.stage == terminal::logical_stage::up) {
            compute::run_w13_task(
                g.w13, coordinate, expert, cta_rank, g.ready,
                phasebits, ready_phase, a_smem, b_smem, d_smem,
                inputs_arrived, inputs_finished, inputs_ready);
        } else if (coordinate.stage == terminal::logical_stage::activation) {
            compute::run_activation_task(
                g.activation, coordinate, cta_rank, g.ready);
        } else if (coordinate.stage == terminal::logical_stage::w2) {
            compute::run_w2_task(
                g.w2, coordinate, expert, cta_rank, g.ready,
                phasebits, ready_phase, a_smem, b_smem, d_smem,
                inputs_arrived, inputs_finished, inputs_ready);
        } else {
            if (cta_rank == 0 && threadIdx.x == 0)
                atomicAdd(g.errors, 1u);
            everyone::tma::cluster::sync();
        }
    }
}

__global__ __launch_bounds__(kReferenceActivationThreads, 1)
void reference_activation_kernel(
        const __nv_bfloat16 *gate_up, uint8_t *hidden,
        float *hidden_scale, int rows, float limit) {
    const int row = blockIdx.x;
    if (row < rows) {
        pipeline::activate_quant_worker(
            gate_up, hidden, hidden_scale, row, threadIdx.x, limit);
    }
}

constexpr int kDynamicSmem =
    compute::PIPE_DEPTH * (sizeof(compute::a_st) + sizeof(compute::b_st))
    + sizeof(compute::d_st) + 1024;

int resident_clusters() {
    CUDACHECK(cudaFuncSetAttribute(
        terminal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmem));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = kDynamicSmem;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = terminal::CLUSTER_CTAS;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    CUDACHECK(cudaOccupancyMaxActiveClusters(
        &clusters, terminal_kernel, &config));
    TORCH_CHECK(clusters >= 1,
                "terminal compute kernel has zero cluster occupancy");
    return clusters;
}

void check_common(
        const at::Tensor &x, const at::Tensor &x_scale,
        const at::Tensor &w13, const at::Tensor &w13_scale,
        const at::Tensor &w2, const at::Tensor &w2_scale,
        const at::Tensor &m_indices, const at::Tensor &gate_up,
        const at::Tensor &hidden, const at::Tensor &hidden_scale,
        const at::Tensor &y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous()
                    && x.scalar_type() == at::kFloat8_e4m3fn
                    && x.dim() == 2 && x.size(0) > 0
                    && x.size(0) % terminal::M_TILE == 0
                    && x.size(1) == terminal::HIDDEN_SIZE,
                "x must be contiguous CUDA FP8 [M64,4096]");
    const int64_t rows = x.size(0);
    TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous()
                    && x_scale.scalar_type() == at::kFloat
                    && x_scale.sizes()
                           == at::IntArrayRef({rows,
                                terminal::HIDDEN_SIZE / 128}),
                "x_scale must be contiguous CUDA FP32 [M,32]");
    TORCH_CHECK(w13.is_cuda() && w13.is_contiguous()
                    && w13.scalar_type() == at::kFloat8_e4m3fn
                    && w13.dim() == 3 && w13.size(0) > 0
                    && w13.size(1) == 2 * terminal::INTERMEDIATE_SIZE
                    && w13.size(2) == terminal::HIDDEN_SIZE,
                "w13 must be contiguous CUDA FP8 [E,4096,4096]");
    TORCH_CHECK(w13_scale.is_cuda() && w13_scale.is_contiguous()
                    && w13_scale.scalar_type() == at::kFloat
                    && w13_scale.dim() == 3
                    && w13_scale.size(0) == w13.size(0)
                    && w13_scale.size(1) == 32
                    && w13_scale.size(2) == 32,
                "w13_scale must be contiguous CUDA FP32 [E,32,32]");
    TORCH_CHECK(w2.is_cuda() && w2.is_contiguous()
                    && w2.scalar_type() == at::kFloat8_e4m3fn
                    && w2.dim() == 3 && w2.size(0) == w13.size(0)
                    && w2.size(1) == terminal::HIDDEN_SIZE
                    && w2.size(2) == terminal::INTERMEDIATE_SIZE,
                "w2 must be contiguous CUDA FP8 [E,4096,2048]");
    TORCH_CHECK(w2_scale.is_cuda() && w2_scale.is_contiguous()
                    && w2_scale.scalar_type() == at::kFloat
                    && w2_scale.dim() == 3
                    && w2_scale.size(0) == w2.size(0)
                    && w2_scale.size(1) == 32
                    && w2_scale.size(2) == 16,
                "w2_scale must be contiguous CUDA FP32 [E,32,16]");
    TORCH_CHECK(m_indices.is_cuda() && m_indices.is_contiguous()
                    && m_indices.scalar_type() == at::kInt
                    && m_indices.dim() == 1 && m_indices.size(0) == rows,
                "m_indices must be contiguous CUDA int32 [M]");
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous()
                    && gate_up.scalar_type() == at::kBFloat16
                    && gate_up.sizes()
                           == at::IntArrayRef({rows,
                                2 * terminal::INTERMEDIATE_SIZE}),
                "gate_up must be contiguous CUDA BF16 [M,4096]");
    TORCH_CHECK(hidden.is_cuda() && hidden.is_contiguous()
                    && hidden.scalar_type() == at::kFloat8_e4m3fn
                    && hidden.sizes()
                           == at::IntArrayRef({rows,
                                terminal::INTERMEDIATE_SIZE}),
                "hidden must be contiguous CUDA FP8 [M,2048]");
    TORCH_CHECK(hidden_scale.is_cuda() && hidden_scale.is_contiguous()
                    && hidden_scale.scalar_type() == at::kFloat
                    && hidden_scale.sizes() == at::IntArrayRef({rows, 16}),
                "hidden_scale must be contiguous CUDA FP32 [M,16]");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous()
                    && y.scalar_type() == at::kBFloat16
                    && y.sizes()
                           == at::IntArrayRef({rows,
                                terminal::HIDDEN_SIZE}),
                "y must be contiguous CUDA BF16 [M,4096]");
    kittens::py::device_check(
        x, x_scale, w13, w13_scale, w2, w2_scale, m_indices,
        gate_up, hidden, hidden_scale, y);
}

void run_split(
        const at::Tensor &x, const at::Tensor &x_scale,
        const at::Tensor &w13, const at::Tensor &w13_scale,
        const at::Tensor &w2, const at::Tensor &w2_scale,
        const at::Tensor &m_indices, const at::Tensor &gate_up,
        const at::Tensor &hidden, const at::Tensor &hidden_scale,
        const at::Tensor &y, double limit) {
    check_common(
        x, x_scale, w13, w13_scale, w2, w2_scale, m_indices,
        gate_up, hidden, hidden_scale, y);
    c10::cuda::CUDAGuard guard(x.device());
    split::entry_pipelined_out(
        x, w13, x_scale, w13_scale, m_indices, gate_up);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    reference_activation_kernel<<<x.size(0),
                                  kReferenceActivationThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<uint8_t *>(hidden.data_ptr()),
        hidden_scale.data_ptr<float>(), static_cast<int>(x.size(0)),
        static_cast<float>(limit));
    CUDACHECK(cudaGetLastError());
    split::entry_pipelined_out(
        hidden, w2, hidden_scale, w2_scale, m_indices, y);
}

void run_terminal(
        const at::Tensor &x, const at::Tensor &x_scale,
        const at::Tensor &w13, const at::Tensor &w13_scale,
        const at::Tensor &w2, const at::Tensor &w2_scale,
        const at::Tensor &m_indices, const at::Tensor &num_tokens,
        const at::Tensor &gate_up, const at::Tensor &hidden,
        const at::Tensor &hidden_scale, const at::Tensor &y,
        const at::Tensor &cursor, const at::Tensor &worker_ticket,
        const at::Tensor &gate_up_ready, const at::Tensor &hidden_ready,
        const at::Tensor &y_ready, const at::Tensor &task_visits,
        const at::Tensor &errors, int64_t minibatch_rows,
        int64_t macrobatch_rows, double limit) {
    check_common(
        x, x_scale, w13, w13_scale, w2, w2_scale, m_indices,
        gate_up, hidden, hidden_scale, y);
    const int64_t rows = x.size(0);
    const int64_t m_tiles = rows / terminal::M_TILE;
    TORCH_CHECK(num_tokens.is_cuda() && num_tokens.is_contiguous()
                    && num_tokens.scalar_type() == at::kInt
                    && num_tokens.numel() == 1,
                "num_tokens must be contiguous CUDA int32 [1]");
    for (const at::Tensor *state : {&cursor, &errors}) {
        TORCH_CHECK(state->is_cuda() && state->is_contiguous()
                        && state->scalar_type() == at::kInt
                        && state->numel() == 1,
                    "cursor/errors must be contiguous CUDA int32 [1]");
    }
    TORCH_CHECK(gate_up_ready.is_cuda() && gate_up_ready.is_contiguous()
                    && gate_up_ready.scalar_type() == at::kInt
                    && gate_up_ready.numel()
                           == m_tiles * terminal::W13_N_TILES,
                "gate_up_ready must be int32 [M64,16]");
    TORCH_CHECK(hidden_ready.is_cuda() && hidden_ready.is_contiguous()
                    && hidden_ready.scalar_type() == at::kInt
                    && hidden_ready.numel() == m_tiles,
                "hidden_ready must be int32 [M64]");
    TORCH_CHECK(y_ready.is_cuda() && y_ready.is_contiguous()
                    && y_ready.scalar_type() == at::kInt
                    && y_ready.numel() == m_tiles,
                "y_ready must be int32 [M64]");
    TORCH_CHECK(task_visits.is_cuda() && task_visits.is_contiguous()
                    && task_visits.scalar_type() == at::kInt
                    && task_visits.numel()
                           == m_tiles * terminal::TASKS_PER_M64,
                "task_visits must have one int32 per logical task");
    TORCH_CHECK(minibatch_rows > 0 && minibatch_rows % terminal::M_TILE == 0
                    && macrobatch_rows >= minibatch_rows
                    && macrobatch_rows % minibatch_rows == 0,
                "minibatch/macrobatch rows violate the logical contract");

    c10::cuda::CUDAGuard guard(x.device());
    const int clusters = resident_clusters();
    TORCH_CHECK(worker_ticket.is_cuda() && worker_ticket.is_contiguous()
                    && worker_ticket.scalar_type() == at::kInt
                    && worker_ticket.numel() >= clusters,
                "worker_ticket must have one int32 per resident cluster");
    kittens::py::device_check(
        x, num_tokens, cursor, worker_ticket, gate_up_ready,
        hidden_ready, y_ready, task_visits, errors);

    terminal_globals globals{
        {
            kittens::py::tensor_to_gl<a_gl>(
                const_cast<at::Tensor &>(x)),
            kittens::py::tensor_to_gl<b_gl>(
                const_cast<at::Tensor &>(w13)),
            kittens::py::tensor_to_gl<d_gl>(
                const_cast<at::Tensor &>(gate_up)),
            x_scale.data_ptr<float>(),
            w13_scale.data_ptr<float>(),
            2 * terminal::INTERMEDIATE_SIZE,
            terminal::HIDDEN_SIZE / 128,
            (2 * terminal::INTERMEDIATE_SIZE) / 64,
        },
        {
            kittens::py::tensor_to_gl<a_gl>(
                const_cast<at::Tensor &>(hidden)),
            kittens::py::tensor_to_gl<b_gl>(
                const_cast<at::Tensor &>(w2)),
            kittens::py::tensor_to_gl<d_gl>(
                const_cast<at::Tensor &>(y)),
            hidden_scale.data_ptr<float>(),
            w2_scale.data_ptr<float>(),
            terminal::HIDDEN_SIZE,
            terminal::INTERMEDIATE_SIZE / 128,
            terminal::HIDDEN_SIZE / 64,
        },
        {
            reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
            reinterpret_cast<uint8_t *>(hidden.data_ptr()),
            hidden_scale.data_ptr<float>(),
            static_cast<float>(limit),
        },
        {
            reinterpret_cast<unsigned int *>(gate_up_ready.data_ptr<int>()),
            reinterpret_cast<unsigned int *>(hidden_ready.data_ptr<int>()),
            reinterpret_cast<unsigned int *>(y_ready.data_ptr<int>()),
        },
        m_indices.data_ptr<int>(),
        num_tokens.data_ptr<int>(),
        reinterpret_cast<unsigned int *>(cursor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(worker_ticket.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(task_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(errors.data_ptr<int>()),
        static_cast<int>(rows),
        static_cast<int>(minibatch_rows),
        static_cast<int>(macrobatch_rows),
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    terminal_kernel<<<clusters * terminal::CLUSTER_CTAS,
                      kThreads, kDynamicSmem, stream>>>(globals);
    CUDACHECK(cudaGetLastError());
}

std::vector<int64_t> attributes() {
    const int clusters = resident_clusters();
    cudaFuncAttributes terminal_attributes{};
    cudaFuncAttributes split_attributes{};
    cudaFuncAttributes activation_attributes{};
    CUDACHECK(cudaFuncGetAttributes(
        &terminal_attributes, terminal_kernel));
    CUDACHECK(cudaFuncGetAttributes(
        &split_attributes, split::kernel));
    CUDACHECK(cudaFuncGetAttributes(
        &activation_attributes, reference_activation_kernel));
    return {
        static_cast<int64_t>(terminal_attributes.numRegs),
        static_cast<int64_t>(terminal_attributes.sharedSizeBytes),
        static_cast<int64_t>(terminal_attributes.localSizeBytes),
        static_cast<int64_t>(terminal_attributes.maxDynamicSharedSizeBytes),
        static_cast<int64_t>(kDynamicSmem),
        static_cast<int64_t>(clusters),
        static_cast<int64_t>(split_attributes.numRegs),
        static_cast<int64_t>(activation_attributes.numRegs),
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_split", &run_split);
    module.def("run_terminal", &run_terminal);
    module.def("attributes", &attributes);
}
