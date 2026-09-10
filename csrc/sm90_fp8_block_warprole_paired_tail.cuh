#pragma once
// Experimental W2 tail pairing. Caller drains the normal ring and owns the
// W13 epilogue staging before overlaying this larger two-stage layout.
#include "kittens.cuh"
#include "sm90_fp8_block_warprole_config.cuh"
#include "sm90_fp8_block_terminal_comm_primitives.cuh"
namespace mok_sm90::warprole::paired_tail {
using namespace kittens;
using A = st_fp8e4m3<64, 128>;
using B = st_fp8e4m3<128, 128>;
using PackedB = st_fp8e4m3<128, 256>;
using Acc = rt_fl<16, 128>;
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

using HalfA = st_fp8e4m3<32,128>;
using XGL = gl<fp8e4m3,1,1,-1,-1,HalfA>;
using WGL = gl<fp8e4m3,1,-1,-1,-1,B>;
// Scale tails would otherwise leave the next stage at a 16-byte offset,
// violating TMA's swizzled-destination alignment. Keep every ring slot aligned.
struct alignas(1024) Stage { A a; B b[2][2]; float as[64]; float bs[2][2]; };
struct Shared { Stage raw[2]; PackedB packed[2]; };
static_assert(sizeof(Stage)%1024==0 && offsetof(Shared,packed)%1024==0);
struct Task { int m0,m1,e0,e1,r0,r1; };
__device__ __forceinline__ void pack(const Stage &s,PackedB &b,int c) {
    for(int v=threadIdx.x%128;v<128*8;v+=128) {
        int n=v/8,k=(v%8)*16;
        uint4 x=*reinterpret_cast<const uint4*>(&s.b[0][c][{n,k}]);
        uint4 y=*reinterpret_cast<const uint4*>(&s.b[1][c][{n,k}]);
        uint4 lo=make_uint4(__byte_perm(x.x,y.x,0x5410),__byte_perm(x.x,y.x,0x7632),
                           __byte_perm(x.y,y.y,0x5410),__byte_perm(x.y,y.y,0x7632));
        uint4 hi=make_uint4(__byte_perm(x.z,y.z,0x5410),__byte_perm(x.z,y.z,0x7632),
                           __byte_perm(x.w,y.w,0x5410),__byte_perm(x.w,y.w,0x7632));
        *reinterpret_cast<uint4*>(&b[{n,2*k}])=lo;
        *reinterpret_cast<uint4*>(&b[{n,2*k+16}])=hi;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    warpgroup::sync(c+1);
}
template<bool First>
__device__ __forceinline__ void promote(Acc &total,const Acc &part,const Stage &s,int c) {
    int row=((threadIdx.x%128)/32)*16+(threadIdx.x%32)/4;
    float b=s.bs[row/32][c];
    float s0=__fmul_rn(s.as[row],b),s1=__fmul_rn(s.as[row+8],b);
#pragma unroll
    for(int j=0;j<8;++j) {
#pragma unroll
        for(int k=0;k<4;++k) {
            float f=k%2?s1:s0;float2 q=part.tiles[0][j].data[k];float2 &t=total.tiles[0][j].data[k];
            if constexpr(First){t.x=__fmul_rn(q.x,f);t.y=__fmul_rn(q.y,f);}
            else {t.x=__fmaf_rn(q.x,f,t.x);t.y=__fmaf_rn(q.y,f,t.y);}
        }
    }
}
__device__ __forceinline__ void mma(Acc &acc,Stage &s,PackedB &b,int c) {
    pack(s,b,c);
    warpgroup::mma_fence(acc);
    st_descriptor<A,0> ad(s.a);st_descriptor<PackedB,0> bd(b);
    uint32_t meta=threadIdx.x%128<64?0x44444444u:0xeeeeeeeeu;
#pragma unroll
    for(int k=0;k<4;++k)sparse_mma(acc,ad.chunk_descriptor(k),bd.chunk_descriptor(2*k),meta,k!=0);
    warpgroup::mma_commit_group();warpgroup::mma_async_wait<0>();
}


// The schedule holds a compact valid prefix followed by padding. Check that
// property directly; a non-prefix tile simply stays on the dense path.
// All CTA threads participate. No m_indices read before dispatch is ready.
template <typename CG>
__device__ __forceinline__ void build_pairs(const CG &g, const int *ends,
                                            int *partner, int *length) {
    for (int e=threadIdx.x/32; e<g.num_local_experts; e+=blockDim.x/32) {
        const int lane=threadIdx.x%32;
        const int start=e?ends[e-1]:0;
        const int last=ends[e]-64;
        bool a=false,b=false;
        if(last>=start && ends[e]<=g.schedule_capacity) {
            a=fp8_block_terminal_comm::decode_route(g,last+lane).valid;
            b=fp8_block_terminal_comm::decode_route(g,last+32+lane).valid;
        }
        const unsigned lo=__ballot_sync(0xffffffffu,a),hi=__ballot_sync(0xffffffffu,b);
        const int n=__popc(lo);
        if(lane==0) {
            // lo == 2^n-1, with an explicit n=32 case to avoid shift overflow.
            length[e]=(hi==0 && n>0 && (n==32 || lo==((1u<<n)-1u)))?n:0;
            partner[e]=-1;
        }
    }
    __syncthreads();
    if(threadIdx.x==0) {
        int pending=-1,q=-1;
        for(int e=0;e<g.num_local_experts;++e) {
            if(length[e]==0)continue;
            const int nextq=(ends[e]/64-1)/MINIBATCH_TILES;
            if(nextq!=q){pending=-1;q=nextq;}
            if(pending<0)pending=e;
            else {partner[pending]=e;partner[e]=pending;pending=-1;}
        }
    }
    __syncthreads();
}
__device__ __forceinline__ Task lookup(int m,int experts,const int *ends,
                                       const int *partner,const int *length) {
    int lo=0,hi=experts;
    while(lo<hi){int mid=(lo+hi)/2;if(ends[mid]<=m*64)lo=mid+1;else hi=mid;}
    if(lo==experts || ends[lo]/64-1!=m || partner[lo]<0)
        return Task{m,-1,lo,-1,0,0};
    const int e=partner[lo];
    return Task{m,ends[e]/64-1,lo,e,length[lo],length[e]};
}

// Reorder only the W2 phase of one original minibatch: all paired leaders
// (n-major), then all ordinary tiles (n-major), then unused partner slots.
// The global cursor still counts the original slots. A false return skips a
// partner slot; a true return gives a canonical task index for the mailbox,
// so consumers need no mapping state. Every W13 index stays unchanged.
__device__ __forceinline__ bool group_w2_task(int &index, task &tk, shape s,
        int experts, const int *ends, const int *partner, const int *length,
        int &cached_q, unsigned &leaders, unsigned &dense) {
    const int lane=threadIdx.x%32;
    const int tiles=tiles_in_minibatch(s,tk.minibatch);
    const int first=first_tile_of_minibatch(tk.minibatch);
    if(cached_q!=tk.minibatch) {
        Task p{};
        if(lane<tiles)p=lookup(first+lane,experts,ends,partner,length);
        leaders=__ballot_sync(0xffffffffu,lane<tiles && p.m1>=0 && p.m0<p.m1);
        dense=__ballot_sync(0xffffffffu,lane<tiles && p.m1<0);
        cached_q=tk.minibatch;
    }
    if(leaders==0)return true;
    constexpr int columns=geometry<2>::W2_TASKS_PER_TILE;
    const int local=tk.n_index*tiles+tk.m_tile-first;
    const int phase_base=index-local;
    const int np=__popc(leaders),nd=__popc(dense);
    const bool paired=local<np*columns;
    const int ordinal=paired?local:local-np*columns;
    const int count=paired?np:nd;
    if(ordinal>=count*columns)return false;
    const unsigned mask=paired?leaders:dense;
    const int n=ordinal/count,offset=ordinal%count;
    // Warp-wide select of the offset-th set bit, without a serial bit loop.
    const unsigned before=(1u<<lane)-1u;
    const unsigned selected=__ballot_sync(0xffffffffu,
        (mask&(1u<<lane))!=0 && __popc(mask&before)==offset);
    const int m=__ffs(selected)-1;
    index=phase_base+n*tiles+m;
    tk.m_tile=first+m;
    tk.n_index=n;
    return true;
}

template <typename G>
__device__ __forceinline__ void producer(const G &g, Shared &s,
        semaphore *full,semaphore *empty,const Task &task,int ng) {
    // K2048 has 16 blocks: both ring slots return to phase zero per task.
    const int lane=threadIdx.x%32;
    for(int kb=0;kb<W2_K_BLOCKS;++kb) {
        int slot=kb%2,phase=(kb/2)%2;wait(empty[slot],phase^1);
        Stage &v=s.raw[slot];
        if(lane==0) {
            tma::expect_bytes(full[slot],sizeof(A)+4*sizeof(B));
            auto &a0=*reinterpret_cast<HalfA*>(&v.a[{0,0}]);
            auto &a1=*reinterpret_cast<HalfA*>(&v.a[{32,0}]);
            tma::load_async(a0,g.hidden_half_gl,{task.m0*2,kb},full[slot]);
            tma::load_async(a1,g.hidden_half_gl,{task.m1*2,kb},full[slot]);
#pragma unroll
            for(int e=0;e<2;++e) {
#pragma unroll
                for(int c=0;c<2;++c)tma::load_async(v.b[e][c],g.w2,{e?task.e1:task.e0,ng*2+c,kb},full[slot]);
            }
        }
        v.as[lane]=g.hidden_scale[(task.m0*64+lane)*W2_K_BLOCKS+kb];
        v.as[lane+32]=g.hidden_scale[(task.m1*64+lane)*W2_K_BLOCKS+kb];
        if(lane<4)v.bs[lane/2][lane%2]=g.w2_scale[((lane/2?task.e1:task.e0)*(HIDDEN/128)+ng*2+lane%2)*W2_K_BLOCKS+kb];
        __syncwarp();
        if(lane==0){asm volatile("fence.proxy.async.shared::cta;" ::: "memory");arrive(full[slot]);}
    }
}

template <typename G>
__device__ __forceinline__ void consumer(const G &g,Shared &s,
        semaphore *full,semaphore *empty,const Task &task,int ng,int role) {
    Acc total;
    wait(full[0],0);mma(total,s.raw[0],s.packed[role],role);
    promote<true>(total,total,s.raw[0],role);
    if(threadIdx.x%32==0)arrive(empty[0]);
#pragma unroll 1
    for(int kb=1;kb<W2_K_BLOCKS;++kb) {
        int slot=kb%2,phase=(kb/2)%2;wait(full[slot],phase);
        Acc partial;mma(partial,s.raw[slot],s.packed[role],role);
        promote<false>(total,partial,s.raw[slot],role);
        if(threadIdx.x%32==0)arrive(empty[slot]);
    }
    int lane=threadIdx.x%32,warp=(threadIdx.x%128)/32;
    auto *out=g.routed_y_gl.raw_ptr;
#pragma unroll
    for(int j=0;j<8;++j) {
#pragma unroll
        for(int k=0;k<4;++k) {
            int r=warp*16+lane/4+(k%2)*8,e=r/32,er=r%32;
            int col=ng*256+role*128+j*16+(lane%4)*2+(k/2)*8;
            if(er<(e?task.r1:task.r0)) {
                size_t index=size_t((e?task.m1:task.m0)*64+er)*HIDDEN+col;
                float2 v=total.tiles[0][j].data[k];
                out[index]=__float2bfloat16_rn(v.x);out[index+1]=__float2bfloat16_rn(v.y);
            }
        }
    }
    // Generic stores by every writer must precede both ready publications.
    __threadfence();warpgroup::sync(role+1);
    if(threadIdx.x%128==0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;"::"l"(g.c.y_ready+task.m0):"memory");
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;"::"l"(g.c.y_ready+task.m1):"memory");
    }
}
} // namespace mok_sm90::warprole::paired_tail
