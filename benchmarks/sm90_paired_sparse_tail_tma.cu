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
struct Task { int m0,m1,e0,e1,r0,r1; };
struct Globals {
    XGL x; WGL w;
    const float *as; const float *bs; bf16 *out;
    const Task *tasks;
    const unsigned int *input_ready;
    unsigned int *output_ready;
    int pairs,KB,N,groups,grid;
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

__device__ __forceinline__ Task task_at(const Globals &g,int p) {
    return g.tasks?g.tasks[p]:Task{p*2,p*2+1,p*2,p*2+1,32,32};
}
__device__ __forceinline__ void wait_input(const Globals &g,const Task &task) {
    if(!g.input_ready)return;
    for(unsigned int spin=0;;++spin) {
        unsigned int a,b;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(a):"l"(g.input_ready+task.m0):"memory");
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(b):"l"(g.input_ready+task.m1):"memory");
        if(a>=16 && b>=16)break;
        if(spin==100000000u)asm volatile("trap;");
        __nanosleep(64);
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");
}
__global__ __launch_bounds__(512,1) void kernel(const __grid_constant__ Globals g) {
    extern __shared__ int data[];
    auto &s=*reinterpret_cast<Shared*>((reinterpret_cast<uint64_t>(data)+1023)&~uint64_t(1023));
    __shared__ semaphore full[2],empty[2];
    if(threadIdx.x==0)for(int i=0;i<2;++i){init_semaphore(full[i],1,1);init_semaphore(empty[i],8,0);}
    __syncthreads();
    int role=threadIdx.x/128;
    if(role==3){warpgroup::decrease_registers<64>();return;}
    if(role==2) {
        warpgroup::decrease_registers<56>();
        if(warpgroup::warpid()!=0)return;
        int lane=threadIdx.x%32;
        for(int t=blockIdx.x;t<g.pairs*g.groups;t+=gridDim.x) {
            const Task task=task_at(g,t/g.groups);int ng=t%g.groups;
            if(lane==0)wait_input(g,task);
            __syncwarp();
            // Supported K2048/K4096 use a multiple of four K128 blocks. Both
            // ring slots therefore return to phase zero at every task boundary.
            for(int kb=0;kb<g.KB;++kb) {
                int slot=kb%2,phase=(kb/2)%2;wait(empty[slot],phase^1);
                Stage &v=s.raw[slot];
                if(lane==0) {
                    tma::expect_bytes(full[slot],sizeof(A)+4*sizeof(B));
                    auto &a0=*reinterpret_cast<HalfA*>(&v.a[{0,0}]);
                    auto &a1=*reinterpret_cast<HalfA*>(&v.a[{32,0}]);
                    tma::load_async(a0,g.x,{task.m0*2,kb},full[slot]);
                    tma::load_async(a1,g.x,{task.m1*2,kb},full[slot]);
#pragma unroll
                    for(int e=0;e<2;++e) {
#pragma unroll
                        for(int c=0;c<2;++c)tma::load_async(v.b[e][c],g.w,{e?task.e1:task.e0,ng*2+c,kb},full[slot]);
                    }
                }
                v.as[lane]=g.as[(task.m0*64+lane)*g.KB+kb];
                v.as[lane+32]=g.as[(task.m1*64+lane)*g.KB+kb];
                if(lane<4)v.bs[lane/2][lane%2]=g.bs[((lane/2?task.e1:task.e0)*(g.N/128)+ng*2+lane%2)*g.KB+kb];
                __syncwarp();
                if(lane==0){asm volatile("fence.proxy.async.shared::cta;" ::: "memory");arrive(full[slot]);}
            }
        }
        return;
    }
    warpgroup::increase_registers<192>();
    for(int t=blockIdx.x;t<g.pairs*g.groups;t+=gridDim.x) {
        const Task task=task_at(g,t/g.groups);int ng=t%g.groups;
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
        int lane=threadIdx.x%32,warp=(threadIdx.x%128)/32;
#pragma unroll
        for(int j=0;j<8;++j) {
#pragma unroll
            for(int k=0;k<4;++k) {
                int r=warp*16+lane/4+(k%2)*8,e=r/32,er=r%32;
                int col=ng*256+role*128+j*16+(lane%4)*2+(k/2)*8;
                if(er<(e?task.r1:task.r0)) {
                    size_t index=size_t((e?task.m1:task.m0)*64+er)*g.N+col;
                    float2 v=total.tiles[0][j].data[k];
                    g.out[index]=__float2bfloat16_rn(v.x);g.out[index+1]=__float2bfloat16_rn(v.y);
                }
            }
        }
        if(g.output_ready) {
            // Publish both tiles only after every consumer thread's stores.
            __threadfence();warpgroup::sync(role+1);
            if(threadIdx.x%128==0) {
                asm volatile("red.release.gpu.global.add.u32 [%0], 1;"::"l"(g.output_ready+task.m0):"memory");
                asm volatile("red.release.gpu.global.add.u32 [%0], 1;"::"l"(g.output_ready+task.m1):"memory");
            }
        }
    }
}
struct Handle { Globals g; };
static thread_local std::string last_error;
extern "C" const char *paired_error(){return last_error.c_str();}
static int create(uint64_t x,uint64_t w,uint64_t as,uint64_t bs,uint64_t out,
                  uint64_t tasks,uint64_t in_ready,uint64_t out_ready,
                  int pairs,int rows,int experts,int N,int K,void **result) {
    try {
        if(pairs<1 || pairs>128 || rows<64 || rows%64 || experts<1 || N<256 || N%256 || N>4096 || (K!=2048 && K!=4096))
            throw std::runtime_error("unsupported task geometry");
        check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
        int occupancy=0,sms=0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,512,SMEM));
        if(occupancy<1)throw std::runtime_error("paired TMA kernel does not fit one CTA per SM");
        int device=0;check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        *result=new Handle{{make_gl<XGL>(x,1,1,rows,K),make_gl<WGL>(w,1,experts,N,K),
            reinterpret_cast<const float*>(as),reinterpret_cast<const float*>(bs),reinterpret_cast<bf16*>(out),
            reinterpret_cast<const Task*>(tasks),reinterpret_cast<const unsigned int*>(in_ready),
            reinterpret_cast<unsigned int*>(out_ready),pairs,K/128,N,N/256,std::min(sms,pairs*(N/256))}};
        return 0;
    }catch(const std::exception &e){last_error=e.what();return 1;}
}
extern "C" int paired_create(uint64_t x,uint64_t w,uint64_t as,uint64_t bs,uint64_t out,
                              int pairs,int K,void **result) {
    return create(x,w,as,bs,out,0,0,0,pairs,pairs*128,pairs*2,256,K,result);
}
// Task descriptors are immutable validated int32 [pairs,6]: m0,m1,e0,e1,r0,r1.
// Valid tails are 1..32 rows. Both source tiles and experts must exist, and each
// output tile must occur once. Readiness is the W2 contract: 16 in, N/128 out.
extern "C" int paired_create_tasks(uint64_t x,uint64_t w,uint64_t as,uint64_t bs,uint64_t out,
                                    uint64_t tasks,uint64_t in_ready,uint64_t out_ready,
                                    int pairs,int rows,int experts,int N,int K,void **result) {
    if(!tasks || !in_ready || !out_ready){last_error="task and readiness pointers required";return 1;}
    return create(x,w,as,bs,out,tasks,in_ready,out_ready,pairs,rows,experts,N,K,result);
}
extern "C" int paired_launch(void *handle,uint64_t stream) {
    auto &g=static_cast<Handle*>(handle)->g;
    kernel<<<g.grid,512,SMEM,reinterpret_cast<cudaStream_t>(stream)>>>(g);
    return int(cudaGetLastError());
}
extern "C" int paired_smem(){return SMEM;}
extern "C" void paired_destroy(void *handle){delete static_cast<Handle*>(handle);}
}
