#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr int THREADS = 128;
constexpr unsigned int TASK_NONE = 0u;
constexpr unsigned int TASK_STOP = ~0u;
constexpr unsigned int PAYLOAD_XOR = 0x5a5a3c3cu;

__device__ __forceinline__ unsigned int load_acquire(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_release(unsigned int *address,
                                               unsigned int value) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int cas_acq_rel(
    unsigned int *address, unsigned int expected, unsigned int desired) {
    unsigned int prior;
    asm volatile("{atom.cas.acq_rel.gpu.global.b32 %0, [%1], %2, %3;}"
                 : "=r"(prior)
                 : "l"(address), "r"(expected), "r"(desired)
                 : "memory");
    return prior;
}

__device__ __forceinline__ void help_visible(unsigned int *state,
                                             const unsigned int *commit,
                                             int items) {
    while (true) {
        const unsigned int current = load_acquire(state + 1);
        if (current >= static_cast<unsigned int>(items)
            || load_acquire(commit + current) == 0u)
            return;
        cas_acq_rel(state + 1, current, current + 1u);
    }
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void ready_queue_kernel(
    unsigned int *state, unsigned int *descriptor, unsigned int *commit,
    unsigned int *visits, unsigned int *worker_descriptor,
    unsigned int *mismatch, int items, unsigned long long delay_cycles) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    unsigned int phase = 0;
    cg::cluster_group cluster_group = cg::this_cluster();

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            unsigned int producer = static_cast<unsigned int>(items);
            while (true) {
                const unsigned int reserved = load_acquire(state + 0);
                if (reserved >= static_cast<unsigned int>(items)) break;
                if (cas_acq_rel(state + 0, reserved, reserved + 1u)
                    == reserved) {
                    producer = reserved;
                    break;
                }
            }
            if (producer < static_cast<unsigned int>(items)) {
                const unsigned int position = producer;
                if (delay_cycles != 0ull && producer % 17u == 0u) {
                    const unsigned long long start = clock64();
                    while (static_cast<unsigned long long>(clock64()) - start
                           < delay_cycles) { }
                }
                descriptor[position] = producer ^ PAYLOAD_XOR;
                store_release(commit + position, 1u);
            }

            help_visible(state, commit, items);

            unsigned int task = TASK_NONE;
            while (true) {
                const unsigned int visible = load_acquire(state + 1);
                const unsigned int head = load_acquire(state + 2);
                if (head >= visible) break;
                if (cas_acq_rel(state + 2, head, head + 1u) == head) {
                    task = descriptor[head];
                    break;
                }
            }

            if (task == TASK_NONE
                && load_acquire(state + 0) >= static_cast<unsigned int>(items)
                && load_acquire(state + 1) >= static_cast<unsigned int>(items)
                && load_acquire(state + 2) >= static_cast<unsigned int>(items))
                task = TASK_STOP;
            store_release(worker_descriptor + cluster * 2 + phase, task);
        }

        cluster_group.sync();
        const unsigned int task =
            load_acquire(worker_descriptor + cluster * 2 + phase);
        if (task == TASK_STOP) break;
        if (task != TASK_NONE && threadIdx.x == 0) {
            const unsigned int item = task ^ PAYLOAD_XOR;
            if (item >= static_cast<unsigned int>(items))
                atomicAdd(mismatch, 1u);
            else if (cta_rank == 0)
                atomicAdd(visits + item, 1u);
        }
        phase ^= 1u;
    }
}

void run_ready_queue_probe(const at::Tensor &state,
                           const at::Tensor &descriptor,
                           const at::Tensor &commit,
                           const at::Tensor &visits,
                           const at::Tensor &worker_descriptor,
                           const at::Tensor &mismatch,
                           int64_t delay_cycles) {
    for (const at::Tensor *tensor :
         {&state, &descriptor, &commit, &visits, &worker_descriptor,
          &mismatch}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "all queue tensors must be contiguous CUDA int32");
    }
    TORCH_CHECK(state.numel() == 3,
                "state must contain reserve_tail, visible_tail, and head");
    TORCH_CHECK(descriptor.numel() > 0
                    && descriptor.numel() == commit.numel()
                    && descriptor.numel() == visits.numel(),
                "descriptor, commit, and visits must share a positive size");
    TORCH_CHECK(worker_descriptor.numel() > 0
                    && worker_descriptor.numel() % 2 == 0,
                "worker_descriptor needs two phase slots per cluster");
    TORCH_CHECK(mismatch.numel() == 1, "mismatch must be int32 [1]");
    TORCH_CHECK(delay_cycles >= 0, "delay_cycles must be nonnegative");
    for (const at::Tensor *tensor :
         {&descriptor, &commit, &visits, &worker_descriptor, &mismatch})
        TORCH_CHECK(tensor->device() == state.device(),
                    "all queue tensors must share one device");

    const int items = static_cast<int>(descriptor.numel());
    const int clusters = static_cast<int>(worker_descriptor.numel() / 2);
    c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(state.get_device());
    ready_queue_kernel<<<clusters * 2, THREADS, 0, stream>>>(
        reinterpret_cast<unsigned int *>(state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(commit.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(worker_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch.data_ptr<int>()), items,
        static_cast<unsigned long long>(delay_cycles));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "ready queue probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_ready_queue_probe);
}
