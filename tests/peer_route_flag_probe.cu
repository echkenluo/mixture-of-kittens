#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint16_t pattern(int source_rank, int iteration,
                                             int route, int column) {
    uint32_t value = static_cast<uint32_t>(source_rank + 1) * 0x9e3779b9u;
    value ^= static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
    value ^= static_cast<uint32_t>(route + 1) * 0xc2b2ae35u;
    value ^= static_cast<uint32_t>(column + 1) * 0x27d4eb2du;
    value ^= value >> 16;
    return static_cast<uint16_t>(value | 1u);
}

__device__ __forceinline__ void store_release(unsigned int *address,
                                               unsigned int value) {
    asm volatile("{st.release.sys.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int load_acquire(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.sys.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ bool wait_phase(const unsigned int *address,
                                            unsigned int phase,
                                            unsigned long long spin_limit) {
    for (unsigned long long iteration = 0; iteration < spin_limit;
         ++iteration) {
        if (load_acquire(address) >= phase) return true;
        __nanosleep(128);
    }
    return false;
}

__global__ void peer_route_flag_kernel(
    uint16_t *local_payload, unsigned int *local_flags,
    unsigned int *local_ack, uint16_t *peer_payload,
    unsigned int *peer_flags, unsigned int *peer_ack,
    unsigned int *mismatch_count, unsigned int *timeout_count,
    int source_rank, int expected_source_rank, int routes, int columns,
    int iterations, int omit_source_rank, int omit_route,
    unsigned long long spin_limit) {
    __shared__ unsigned int iteration_failed;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const unsigned int phase = static_cast<unsigned int>(iteration + 1);
        const int elements = routes * columns;
        for (int index = threadIdx.x; index < elements;
             index += blockDim.x) {
            const int route = index / columns;
            const int column = index - route * columns;
            peer_payload[index] =
                pattern(source_rank, iteration, route, column);
        }

        // Every writer fences its own stores.  The block barrier then lets
        // thread 0 publish per-route flags only after all row stores are
        // system-visible.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int route = 0; route < routes; ++route) {
                if (source_rank == omit_source_rank && route == omit_route)
                    continue;
                store_release(peer_flags + route, phase);
            }
            iteration_failed = 0;
        }
        __syncthreads();

        // Each thread acquires the flag for every route whose payload
        // elements it validates.  This is deliberately stronger than a
        // single-thread acquire followed by a block barrier.
        for (int route = threadIdx.x; route < routes;
             route += blockDim.x) {
            if (!wait_phase(local_flags + route, phase, spin_limit)) {
                atomicAdd(timeout_count, 1u);
                atomicExch(&iteration_failed, 1u);
            }
        }
        __syncthreads();
        if (iteration_failed) return;

        for (int index = threadIdx.x; index < elements;
             index += blockDim.x) {
            const int route = index / columns;
            const int column = index - route * columns;
            if (local_payload[index]
                != pattern(expected_source_rank, iteration, route, column))
                atomicAdd(mismatch_count, 1u);
        }

        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int route = 0; route < routes; ++route)
                store_release(peer_ack + route, phase);
            iteration_failed = 0;
        }
        __syncthreads();

        for (int route = threadIdx.x; route < routes;
             route += blockDim.x) {
            if (!wait_phase(local_ack + route, phase, spin_limit)) {
                atomicAdd(timeout_count, 1u);
                atomicExch(&iteration_failed, 1u);
            }
        }
        __syncthreads();
        if (iteration_failed) return;
    }
}

void run_peer_route_flag_probe(
    const at::Tensor &local_payload, const at::Tensor &local_flags,
    const at::Tensor &local_ack, int64_t peer_payload_ptr,
    int64_t peer_flags_ptr, int64_t peer_ack_ptr,
    const at::Tensor &mismatch_count, const at::Tensor &timeout_count,
    int64_t source_rank, int64_t expected_source_rank, int64_t routes,
    int64_t columns, int64_t iterations, int64_t omit_source_rank,
    int64_t omit_route, int64_t spin_limit) {
    TORCH_CHECK(local_payload.is_cuda()
                    && local_payload.scalar_type() == at::kShort
                    && local_payload.is_contiguous()
                    && local_payload.numel() == routes * columns,
                "local_payload must be contiguous CUDA int16 [routes,columns]");
    for (const at::Tensor *tensor : {&local_flags, &local_ack,
                                      &mismatch_count, &timeout_count}) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "flag and result tensors must be contiguous CUDA int32");
    }
    TORCH_CHECK(local_flags.numel() == routes && local_ack.numel() == routes,
                "local flag arrays must have one entry per route");
    TORCH_CHECK(mismatch_count.numel() == 1 && timeout_count.numel() == 1,
                "result counters must be scalar tensors");
    TORCH_CHECK(routes > 0 && columns > 0 && iterations > 0,
                "routes, columns, and iterations must be positive");
    TORCH_CHECK(peer_payload_ptr > 0 && peer_flags_ptr > 0
                    && peer_ack_ptr > 0,
                "peer pointers must be positive");

    c10::cuda::CUDAGuard guard(local_payload.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(local_payload.get_device());
    peer_route_flag_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<uint16_t *>(local_payload.data_ptr<int16_t>()),
        reinterpret_cast<unsigned int *>(local_flags.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(local_ack.data_ptr<int>()),
        reinterpret_cast<uint16_t *>(peer_payload_ptr),
        reinterpret_cast<unsigned int *>(peer_flags_ptr),
        reinterpret_cast<unsigned int *>(peer_ack_ptr),
        reinterpret_cast<unsigned int *>(mismatch_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        static_cast<int>(source_rank), static_cast<int>(expected_source_rank),
        static_cast<int>(routes), static_cast<int>(columns),
        static_cast<int>(iterations), static_cast<int>(omit_source_rank),
        static_cast<int>(omit_route),
        static_cast<unsigned long long>(spin_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "peer route flag probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_peer_route_flag_probe);
}
