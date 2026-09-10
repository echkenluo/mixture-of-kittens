// Experimental two-consumer paired-tail GEMM. No production binding changes.
// Build as a shared library; the Python probe supplies the frozen c2s4 baseline.
#include "sm90_paired_sparse_tail_common.cuh"
namespace paired_tma {
using HalfA = st_fp8e4m3<32,128>;
using XGL = gl<fp8e4m3,1,1,-1,-1,HalfA>;
using WGL = gl<fp8e4m3,1,-1,-1,-1,B>;
// Scale tails would otherwise leave the next stage at a 16-byte offset,
// violating TMA's swizzled-destination alignment. Keep every ring slot aligned.
struct alignas(1024) Stage { A a; B b[2][2]; float as[64]; float bs[2][2]; };
struct Shared { Stage raw[2]; PackedB packed[2]; };
static_assert(sizeof(Stage)%1024==0 && offsetof(Shared,packed)%1024==0);
struct Globals {
    XGL x; WGL w;
    const float *as; const float *bs; bf16 *out;
    int pairs,KB;
};
constexpr int SMEM=sizeof(Shared)+1024;

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
__global__ __launch_bounds__(512,1) void kernel(const __grid_constant__ Globals g) {
    extern __shared__ int data[];
    auto &s=*reinterpret_cast<Shared*>((reinterpret_cast<uint64_t>(data)+1023)&~uint64_t(1023));
    __shared__ semaphore full[2],empty[2];
    if(threadIdx.x==0)for(int i=0;i<2;++i){init_semaphore(full[i],1,1);init_semaphore(empty[i],8,0);}
    __syncthreads();
    int role=threadIdx.x/128,pair=blockIdx.x;
    if(role==3){warpgroup::decrease_registers<64>();return;}
    if(role==2) {
        warpgroup::decrease_registers<56>();
        if(warpgroup::warpid()!=0)return;
        int lane=threadIdx.x%32;
        for(int kb=0;kb<g.KB;++kb) {
            int slot=kb%2,phase=(kb/2)%2;wait(empty[slot],phase^1);
            Stage &v=s.raw[slot];
            if(lane==0) {
                tma::expect_bytes(full[slot],sizeof(A)+4*sizeof(B));
                // A has K128 width: its two M32 halves have the same swizzle
                // mapping as two HalfA tiles placed at offsets 0 and 4096.
                auto &a0=*reinterpret_cast<HalfA*>(&v.a[{0,0}]);
                auto &a1=*reinterpret_cast<HalfA*>(&v.a[{32,0}]);
                tma::load_async(a0,g.x,{pair*4,kb},full[slot]);
                tma::load_async(a1,g.x,{pair*4+2,kb},full[slot]);
#pragma unroll
                for(int e=0;e<2;++e) {
#pragma unroll
                    for(int c=0;c<2;++c)tma::load_async(v.b[e][c],g.w,{pair*2+e,c,kb},full[slot]);
                }
            }
            v.as[lane]=g.as[((pair*2)*64+lane)*g.KB+kb];
            v.as[lane+32]=g.as[((pair*2+1)*64+lane)*g.KB+kb];
            if(lane<4)v.bs[lane/2][lane%2]=g.bs[((pair*2+lane/2)*2+lane%2)*g.KB+kb];
            __syncwarp();
            if(lane==0){asm volatile("fence.proxy.async.shared::cta;" ::: "memory");arrive(full[slot]);}
        }
        return;
    }
    warpgroup::increase_registers<192>();
    Acc total;
    wait(full[0],0);mma(total,s.raw[0],s.packed[role],role);
    promote<true>(total,total,s.raw[0],role);
    if(threadIdx.x%32==0)arrive(empty[0]);
#pragma unroll 1
    for(int kb=1;kb<g.KB;++kb) {
        int slot=kb%2,phase=(kb/2)%2;wait(full[slot],phase);
        Acc partial;mma(partial,s.raw[slot],s.packed[role],role);
        promote<false>(total,partial,s.raw[slot],role);
        if(threadIdx.x%32==0)arrive(empty[slot]);
    }
    // Preserve the production padded output layout; only valid tail rows are
    // written, with one BF16 rounding. The caller initializes padding to zero.
    int lane=threadIdx.x%32,warp=(threadIdx.x%128)/32;
#pragma unroll
    for(int j=0;j<8;++j) {
#pragma unroll
        for(int k=0;k<4;++k) {
            int r=warp*16+lane/4+(k%2)*8;
            int col=role*128+j*16+(lane%4)*2+(k/2)*8;
            size_t index=size_t((pair*2+r/32)*64+r%32)*256+col;
            float2 v=total.tiles[0][j].data[k];
            g.out[index]=__float2bfloat16_rn(v.x);g.out[index+1]=__float2bfloat16_rn(v.y);
        }
    }
}
struct Handle { Globals g; };
static thread_local std::string last_error;
extern "C" const char *paired_error(){return last_error.c_str();}
extern "C" int paired_create(uint64_t x,uint64_t w,uint64_t as,uint64_t bs,uint64_t out,
                              int pairs,int K,void **result) {
    try {
        if(pairs<1 || pairs>128 || K!=4096)throw std::runtime_error("requires pairs 1..128 and K4096");
        check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
        int occupancy=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,512,SMEM));
        if(occupancy<1)throw std::runtime_error("paired TMA kernel does not fit one CTA per SM");
        *result=new Handle{{make_gl<XGL>(x,1,1,pairs*2*64,K),make_gl<WGL>(w,1,pairs*2,256,K),
            reinterpret_cast<const float*>(as),reinterpret_cast<const float*>(bs),reinterpret_cast<bf16*>(out),pairs,K/128}};
        return 0;
    }catch(const std::exception &e){last_error=e.what();return 1;}
}
extern "C" int paired_launch(void *handle,uint64_t stream) {
    auto &g=static_cast<Handle*>(handle)->g;
    kernel<<<g.pairs,512,SMEM,reinterpret_cast<cudaStream_t>(stream)>>>(g);
    return int(cudaGetLastError());
}
extern "C" int paired_smem(){return SMEM;}
extern "C" void paired_destroy(void *handle){delete static_cast<Handle*>(handle);}
}
