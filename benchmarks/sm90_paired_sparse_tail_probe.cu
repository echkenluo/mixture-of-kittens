// Standalone feasibility probe, not a production MoK entry point.
// Two 32-row expert tails, N=128, original K=128. No block scaling/fusion.
// Both paths include global-to-shared loads; sparse includes B interleaving.
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
// Each CTA owns one distinct pair. All 128 threads execute every WGMMA.
__global__ __launch_bounds__(128) void paired_sparse(const fp8e4m3 *xs,
                                                   const fp8e4m3 *ws, float *out) {
    __shared__ A a;
    __shared__ PackedB b;
    int pair=blockIdx.x;
    for (int i=threadIdx.x;i<64*128;i+=128) {
        int r=i/128,k=i%128,e=r/32,er=r%32;
        a[{r,k}]=xs[((pair*2+e)*64+er)*128+k];
    }
    for (int i=threadIdx.x;i<128*256;i+=128) {
        int n=i/256,ek=i%256,e=(ek%4)/2,k=(ek/4)*2+ek%2;
        b[{n,ek}]=ws[((pair*2+e)*128+n)*128+k];
    }
    publish_shared();
    Acc acc;
    warpgroup::mma_fence(acc);
    st_descriptor<A,0> ad(a);
    st_descriptor<PackedB,0> bd(b);
    // Each warp provides metadata for 16 rows. With a 32/32 split, a warp
    // belongs entirely to one expert. 0x4 selects positions 0,1; 0xe selects 2,3.
    uint32_t metadata=threadIdx.x < 64 ? 0x44444444u : 0xeeeeeeeeu;
#pragma unroll
    for (int k=0;k<4;++k) sparse_mma(acc,ad.chunk_descriptor(k),bd.chunk_descriptor(2*k),metadata,k!=0);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait<0>();
    store_acc(out+pair*64*128,acc,-1);
}
__global__ __launch_bounds__(128) void two_dense(const fp8e4m3 *xs,
                                               const fp8e4m3 *ws, float *out) {
    __shared__ A a;
    __shared__ B b;
    int pair=blockIdx.x;
#pragma unroll
    for (int expert=0;expert<2;++expert) {
        for (int i=threadIdx.x;i<64*128;i+=128)
            a[{i/128,i%128}]=xs[(pair*2+expert)*64*128+i];
        for (int i=threadIdx.x;i<128*128;i+=128)
            b[{i/128,i%128}]=ws[(pair*2+expert)*128*128+i];
        publish_shared();
        Acc acc;
        warpgroup::mm_ABt(acc,a,b);
        warpgroup::mma_async_wait<0>();
        store_acc(out+pair*64*128,acc,expert);
        __syncthreads();
    }
}
int main(int argc,char **argv) {
    try {
        int pairs=132;
        if (argc==3 && std::string(argv[1])=="--pairs") pairs=std::stoi(argv[2]);
        else if (argc!=1) throw std::runtime_error("usage: paired-tail-probe [--pairs 1..1024]");
        if (pairs<1 || pairs>1024) throw std::runtime_error("pairs outside 1..1024");
        int devices=0; check(cudaGetDeviceCount(&devices));
        if (devices!=1) throw std::runtime_error("exactly one explicitly selected H20 required");
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
        if (prop.major!=9 || prop.minor!=0 || !std::strstr(prop.name,"H20"))
            throw std::runtime_error("H20 SM90 required");
        std::vector<fp8e4m3> x(pairs*2*64*128),w(pairs*2*128*128);
        std::vector<float> xf(x.size()),wf(w.size());
        uint32_t state=20260910;
        auto value=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return float(int(state%5)-2);};
        for (size_t i=0;i<x.size();++i) {
            float v=((i/128)%64)<32 ? value() : 0.f; xf[i]=v;x[i]=fp8e4m3(v);
        }
        for (size_t i=0;i<w.size();++i) {float v=value();wf[i]=v;w[i]=fp8e4m3(v);}
        fp8e4m3 *dx,*dw; float *dense,*sparse;
        size_t outputs=size_t(pairs)*64*128;
        check(cudaMalloc(&dx,x.size()));check(cudaMalloc(&dw,w.size()));
        check(cudaMalloc(&dense,outputs*sizeof(float)));check(cudaMalloc(&sparse,outputs*sizeof(float)));
        check(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        check(cudaMemset(dense,0xff,outputs*sizeof(float)));check(cudaMemset(sparse,0xff,outputs*sizeof(float)));
        two_dense<<<pairs,128>>>(dx,dw,dense); check(cudaGetLastError());
        paired_sparse<<<pairs,128>>>(dx,dw,sparse); check(cudaGetLastError());check(cudaDeviceSynchronize());
        std::vector<float> hd(outputs),hs(outputs);
        check(cudaMemcpy(hd.data(),dense,outputs*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hs.data(),sparse,outputs*sizeof(float),cudaMemcpyDeviceToHost));
        size_t dense_bad=0,sparse_bad=0;
        for (int p=0;p<pairs;++p) for(int r=0;r<64;++r) for(int n=0;n<128;++n) {
            float ref=0; int e=r/32,er=r%32;
            for(int k=0;k<128;++k) ref+=xf[((p*2+e)*64+er)*128+k]*wf[((p*2+e)*128+n)*128+k];
            size_t i=(size_t(p)*64+r)*128+n;
            dense_bad+=!std::isfinite(hd[i]) || hd[i]!=ref;
            sparse_bad+=!std::isfinite(hs[i]) || hs[i]!=ref;
        }
        if(dense_bad || sparse_bad) {
            std::cout<<"{\"numeric_pass\":false,\"dense_bad\":"<<dense_bad<<",\"sparse_bad\":"<<sparse_bad<<"}\n";
            return 6;
        }
        auto launch=[&](bool sp){if(sp)paired_sparse<<<pairs,128>>>(dx,dw,sparse);else two_dense<<<pairs,128>>>(dx,dw,dense);};
        for(int i=0;i<10;++i){launch(false);launch(true);}check(cudaDeviceSynchronize());
        cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        std::cout<<std::setprecision(9)<<"{\"numeric_pass\":true,\"outputs_checked\":"<<outputs
            <<",\"pairs\":"<<pairs<<",\"gpu\":\""<<prop.name<<"\",\"sms\":"<<prop.multiProcessorCount
            <<",\"scope\":\"32+32 rows, N128 K128, integer-valued FP8; includes scalar loads and B packing; no production TMA, scaling, communication or E2E claim\",\"blocks\":[";
        for(int order=0;order<4;++order) {
            bool sp=order==1 || order==2;
            if(order)std::cout<<',';
            std::cout<<"{\"arm\":\""<<(sp?"sparse":"dense")<<"\",\"ms_per_call\":[";
            for(int sample=0;sample<20;++sample) {
                check(cudaEventRecord(begin));for(int call=0;call<20;++call)launch(sp);
                check(cudaEventRecord(end));check(cudaEventSynchronize(end));check(cudaGetLastError());
                float ms=0;check(cudaEventElapsedTime(&ms,begin,end));if(sample)std::cout<<',';std::cout<<ms/20;
            }
            std::cout<<"]}";
        }
        std::cout<<"]}\n";
        check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
        check(cudaFree(dx));check(cudaFree(dw));check(cudaFree(dense));check(cudaFree(sparse));
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 10;}
}
