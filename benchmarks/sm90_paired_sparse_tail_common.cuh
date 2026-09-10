#pragma once
// Standalone feasibility probe, not a production MoK entry point.
// Two 32-row expert tails, N=128, K=128 or 4096, per-row/block scales.
// Both paths use 16-byte vector loads; sparse includes online B interleaving.
// Neither path has production TMA/staged scheduling; ratios are NOT c2s4 gains.
// Build: nvcc -std=c++20 -O3 -gencode arch=compute_90a,code=sm_90a -DKITTENS_SM90 --extended-lambda \
//   --expt-relaxed-constexpr -Ithird_party/ThunderKittens/include benchmarks/sm90_paired_sparse_tail_probe.cu \
//   -o paired-tail-probe -Xptxas=-v
#include "kittens.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
using namespace kittens;
using A = st_fp8e4m3<64, 128>;
using B = st_fp8e4m3<128, 128>;
using PackedB = st_fp8e4m3<128, 256>;
using Acc = rt_fl<16, 128>;
static void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
__device__ __forceinline__ void sparse_mma(Acc &acc, uint64_t a, uint64_t b,
                                           uint32_t metadata, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %67, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n128k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, 0, p, 1, 1;\n}\n"
        : "+f"(acc.tiles[0][0].data[0].x),
        "+f"(acc.tiles[0][0].data[0].y),
        "+f"(acc.tiles[0][0].data[1].x),
        "+f"(acc.tiles[0][0].data[1].y),
        "+f"(acc.tiles[0][0].data[2].x),
        "+f"(acc.tiles[0][0].data[2].y),
        "+f"(acc.tiles[0][0].data[3].x),
        "+f"(acc.tiles[0][0].data[3].y),
        "+f"(acc.tiles[0][1].data[0].x),
        "+f"(acc.tiles[0][1].data[0].y),
        "+f"(acc.tiles[0][1].data[1].x),
        "+f"(acc.tiles[0][1].data[1].y),
        "+f"(acc.tiles[0][1].data[2].x),
        "+f"(acc.tiles[0][1].data[2].y),
        "+f"(acc.tiles[0][1].data[3].x),
        "+f"(acc.tiles[0][1].data[3].y),
        "+f"(acc.tiles[0][2].data[0].x),
        "+f"(acc.tiles[0][2].data[0].y),
        "+f"(acc.tiles[0][2].data[1].x),
        "+f"(acc.tiles[0][2].data[1].y),
        "+f"(acc.tiles[0][2].data[2].x),
        "+f"(acc.tiles[0][2].data[2].y),
        "+f"(acc.tiles[0][2].data[3].x),
        "+f"(acc.tiles[0][2].data[3].y),
        "+f"(acc.tiles[0][3].data[0].x),
        "+f"(acc.tiles[0][3].data[0].y),
        "+f"(acc.tiles[0][3].data[1].x),
        "+f"(acc.tiles[0][3].data[1].y),
        "+f"(acc.tiles[0][3].data[2].x),
        "+f"(acc.tiles[0][3].data[2].y),
        "+f"(acc.tiles[0][3].data[3].x),
        "+f"(acc.tiles[0][3].data[3].y),
        "+f"(acc.tiles[0][4].data[0].x),
        "+f"(acc.tiles[0][4].data[0].y),
        "+f"(acc.tiles[0][4].data[1].x),
        "+f"(acc.tiles[0][4].data[1].y),
        "+f"(acc.tiles[0][4].data[2].x),
        "+f"(acc.tiles[0][4].data[2].y),
        "+f"(acc.tiles[0][4].data[3].x),
        "+f"(acc.tiles[0][4].data[3].y),
        "+f"(acc.tiles[0][5].data[0].x),
        "+f"(acc.tiles[0][5].data[0].y),
        "+f"(acc.tiles[0][5].data[1].x),
        "+f"(acc.tiles[0][5].data[1].y),
        "+f"(acc.tiles[0][5].data[2].x),
        "+f"(acc.tiles[0][5].data[2].y),
        "+f"(acc.tiles[0][5].data[3].x),
        "+f"(acc.tiles[0][5].data[3].y),
        "+f"(acc.tiles[0][6].data[0].x),
        "+f"(acc.tiles[0][6].data[0].y),
        "+f"(acc.tiles[0][6].data[1].x),
        "+f"(acc.tiles[0][6].data[1].y),
        "+f"(acc.tiles[0][6].data[2].x),
        "+f"(acc.tiles[0][6].data[2].y),
        "+f"(acc.tiles[0][6].data[3].x),
        "+f"(acc.tiles[0][6].data[3].y),
        "+f"(acc.tiles[0][7].data[0].x),
        "+f"(acc.tiles[0][7].data[0].y),
        "+f"(acc.tiles[0][7].data[1].x),
        "+f"(acc.tiles[0][7].data[1].y),
        "+f"(acc.tiles[0][7].data[2].x),
        "+f"(acc.tiles[0][7].data[2].y),
        "+f"(acc.tiles[0][7].data[3].x),
        "+f"(acc.tiles[0][7].data[3].y)
        : "l"(a), "l"(b), "r"(metadata), "r"(accumulate));
}

__device__ __forceinline__ void publish_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
}
__device__ __forceinline__ void store_acc(float *out, const Acc &acc, int expert) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int j=0;j<8;++j) {
#pragma unroll
        for (int k=0;k<4;++k) {
            int r=warp*16+lane/4+(k%2)*8;
            int c=j*16+(lane%4)*2+(k/2)*8;
            if (expert < 0 || r < 32) {
                int dest_r=expert < 0 ? r : expert*32+r;
                float2 v=acc.tiles[0][j].data[k];
                out[dest_r*128+c]=v.x;
                out[dest_r*128+c+1]=v.y;
            }
        }
    }
}
