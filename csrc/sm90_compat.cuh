#pragma once
// SM90/H20 port: parse stubs for Blackwell MX-scale types. The BF16-only
// build never instantiates the USE_MXFP8 paths; these stubs keep the
// discarded constexpr branches parsing with identical 1-byte layout math.
#if defined(KITTENS_SM90)
namespace kittens {
using fp8e8m0 = uint8;
template<int _height, int _width, bool _swizzle = true, int _swizzle_bytes = 0>
using st_fp8e8m0 = st<uint8, _height, _width, _swizzle, _swizzle_bytes>;
} // namespace kittens
#endif

#if defined(KITTENS_SM90)
namespace kittens {
// tmem scale tiles: referenced only inside discarded USE_ROUTED_MXFP8
// branches of the BF16 build; parse-only stub.
template<int _cols> struct full_tt_fp8e8m0 {
    template<typename T> __device__ inline T subtile(int) const { return T{}; }
};
} // namespace kittens
#endif

#if defined(KITTENS_SM90)
namespace kittens {
namespace clc {
// SM90 v1 shim: Hopper has no clusterlaunchcontrol. Work stealing is
// disabled -- query() always reports failure so each cluster runs exactly
// its initially-assigned task (all grid blocks launch normally on SM90).
// schedule() must still complete the handle-sized tx the caller expects on
// BOTH cluster CTAs' semaphores, otherwise their wait deadlocks.
struct handle { uint4 internal_value; };
struct result { uint32_t success, x, y, z; };
__device__ static inline void schedule(handle &h, semaphore &sem) {
    uint32_t bytes = sizeof(handle);
    uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    #pragma unroll
    for (uint32_t cta = 0; cta < 2; cta++) { // MoK config::CLUSTER_SIZE == 2
        uint32_t remote_bar;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
            : "=r"(remote_bar) : "r"(bar), "r"(cta));
        asm volatile("mbarrier.complete_tx.shared::cluster.b64 [%0], %1;\n"
            :: "r"(remote_bar), "r"(bytes) : "memory");
    }
}
__device__ static inline result query(handle &) { return {0u, 0u, 0u, 0u}; }
} // namespace clc
} // namespace kittens
#endif
