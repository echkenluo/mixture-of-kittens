// Standalone vector-load probe. See shared definitions for build flags.
#include "sm90_paired_sparse_tail_common.cuh"

// 16-byte chunks preserve the shared-memory swizzle within each vector.
// B expansion is online: no offline weight copy and no omitted packing cost.
template<bool Sparse>
__device__ __forceinline__ void block_mma(
        Acc &acc, A &a, std::conditional_t<Sparse,PackedB,B> &b,
        const fp8e4m3 *xs,const fp8e4m3 *ws,int pair,int expert,int K,int kb) {
    for(int v=threadIdx.x;v<64*8;v+=128) {
        int r=v/8,k=(v%8)*16,e=Sparse?r/32:expert,er=Sparse?r%32:r;
        const uint4 data=*reinterpret_cast<const uint4*>(xs+((pair*2+e)*64+er)*K+kb*128+k);
        *reinterpret_cast<uint4*>(&a[{r,k}])=data;
    }
    for(int v=threadIdx.x;v<128*8;v+=128) {
        int n=v/8,k=(v%8)*16;
        if constexpr(Sparse) {
            const uint4 x=*reinterpret_cast<const uint4*>(ws+(pair*2*128+n)*K+kb*128+k);
            const uint4 y=*reinterpret_cast<const uint4*>(ws+((pair*2+1)*128+n)*K+kb*128+k);
            uint4 lo=make_uint4(__byte_perm(x.x,y.x,0x5410),__byte_perm(x.x,y.x,0x7632),
                               __byte_perm(x.y,y.y,0x5410),__byte_perm(x.y,y.y,0x7632));
            uint4 hi=make_uint4(__byte_perm(x.z,y.z,0x5410),__byte_perm(x.z,y.z,0x7632),
                               __byte_perm(x.w,y.w,0x5410),__byte_perm(x.w,y.w,0x7632));
            *reinterpret_cast<uint4*>(&b[{n,2*k}])=lo;
            *reinterpret_cast<uint4*>(&b[{n,2*k+16}])=hi;
        } else {
            *reinterpret_cast<uint4*>(&b[{n,k}])=
                *reinterpret_cast<const uint4*>(ws+((pair*2+expert)*128+n)*K+kb*128+k);
        }
    }
    publish_shared();
    if constexpr(Sparse) {
        warpgroup::mma_fence(acc);
        st_descriptor<A,0> ad(a);st_descriptor<PackedB,0> bd(b);
        uint32_t meta=threadIdx.x<64?0x44444444u:0xeeeeeeeeu;
#pragma unroll
        for(int k=0;k<4;++k)sparse_mma(acc,ad.chunk_descriptor(k),bd.chunk_descriptor(2*k),meta,k!=0);
        warpgroup::mma_commit_group();
    } else warpgroup::mm_ABt(acc,a,b);
    warpgroup::mma_async_wait<0>();
    // Shared tiles may be overwritten only after all warps finish their reads.
    __syncthreads();
}
template<bool Sparse,bool First>
__device__ __forceinline__ void scale_block(Acc &total,const Acc &partial,
        const float *as,const float *bs,int pair,int expert,int KB,int kb) {
    int row=(threadIdx.x/32)*16+(threadIdx.x%32)/4;
    int e=Sparse?row/32:expert,er=Sparse?row%32:row;
    float bscale=bs[(pair*2+e)*KB+kb];
    float s0=__fmul_rn(as[((pair*2+e)*64+er)*KB+kb],bscale);
    float s1=__fmul_rn(as[((pair*2+e)*64+er+8)*KB+kb],bscale);
#pragma unroll
    for(int j=0;j<8;++j) {
#pragma unroll
        for(int k=0;k<4;++k) {
            float scale=(k%2)?s1:s0;
            float2 q=partial.tiles[0][j].data[k];float2 &t=total.tiles[0][j].data[k];
            if constexpr(First) {t.x=__fmul_rn(q.x,scale);t.y=__fmul_rn(q.y,scale);}
            else {t.x=__fmaf_rn(q.x,scale,t.x);t.y=__fmaf_rn(q.y,scale,t.y);}
        }
    }
}
template<bool Sparse>
__global__ __launch_bounds__(128) void tail_gemm(const fp8e4m3 *xs,const fp8e4m3 *ws,
        const float *as,const float *bs,float *out,int K) {
    __shared__ A a;
    __shared__ std::conditional_t<Sparse,PackedB,B> b;
    int pair=blockIdx.x,KB=K/128;
#pragma unroll 1
    for(int expert=0;expert<(Sparse?1:2);++expert) {
        // Peel block zero, matching the production FMA-promotion order and
        // avoiding a copy-or-add join that can coalesce partial/total registers.
        Acc total;
        block_mma<Sparse>(total,a,b,xs,ws,pair,expert,K,0);
        scale_block<Sparse,true>(total,total,as,bs,pair,expert,KB,0);
#pragma unroll 1
        for(int kb=1;kb<KB;++kb) {
            Acc partial;
            block_mma<Sparse>(partial,a,b,xs,ws,pair,expert,K,kb);
            scale_block<Sparse,false>(total,partial,as,bs,pair,expert,KB,kb);
        }
        store_acc(out+pair*64*128,total,Sparse?-1:expert);
        __syncthreads();
    }
}
int main(int argc,char **argv) {
    try {
        int pairs=78,K=4096;
        for(int i=1;i<argc;i+=2) {
            if(i+1==argc)throw std::runtime_error("missing option value");
            std::string arg(argv[i]);
            if(arg=="--pairs")pairs=std::stoi(argv[i+1]);
            else if(arg=="--k")K=std::stoi(argv[i+1]);
            else throw std::runtime_error("usage: paired-tail-probe [--pairs 1..128] [--k 128|4096]");
        }
        if(pairs<1 || pairs>128 || (K!=128 && K!=4096))throw std::runtime_error("unsupported shape");
        const int KB=K/128;
        int devices=0;check(cudaGetDeviceCount(&devices));
        if(devices!=1)throw std::runtime_error("exactly one explicitly selected H20 required");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
        if(prop.major!=9 || prop.minor!=0 || !std::strstr(prop.name,"H20"))throw std::runtime_error("H20 SM90 required");
        std::vector<fp8e4m3> x(size_t(pairs)*2*64*K),w(size_t(pairs)*2*128*K);
        std::vector<float> xf(x.size()),wf(w.size()),as(pairs*2*64*KB),bs(pairs*2*KB);
        uint32_t state=20260910;
        auto value=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return float(int(state%5)-2);};
        for(size_t i=0;i<x.size();++i) {float v=((i/K)%64)<32?value():0.f;xf[i]=v;x[i]=fp8e4m3(v);}
        for(size_t i=0;i<w.size();++i) {float v=value();wf[i]=v;w[i]=fp8e4m3(v);}
        for(size_t i=0;i<as.size();++i)as[i]=std::ldexp(1.f,int((i*7+i/KB)%5)-2);
        for(size_t i=0;i<bs.size();++i)bs[i]=std::ldexp(1.f,int((i*3+i/KB)%5)-2);
        fp8e4m3 *dx,*dw;float *das,*dbs,*dense,*sparse;
        size_t outputs=size_t(pairs)*64*128;
        check(cudaMalloc(&dx,x.size()));check(cudaMalloc(&dw,w.size()));
        check(cudaMalloc(&das,as.size()*sizeof(float)));check(cudaMalloc(&dbs,bs.size()*sizeof(float)));
        check(cudaMalloc(&dense,outputs*sizeof(float)));check(cudaMalloc(&sparse,outputs*sizeof(float)));
        check(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(das,as.data(),as.size()*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dbs,bs.data(),bs.size()*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemset(dense,0xff,outputs*sizeof(float)));check(cudaMemset(sparse,0xff,outputs*sizeof(float)));
        auto launch=[&](bool sp){
            if(sp)tail_gemm<true><<<pairs,128>>>(dx,dw,das,dbs,sparse,K);
            else tail_gemm<false><<<pairs,128>>>(dx,dw,das,dbs,dense,K);
        };
        launch(false);check(cudaGetLastError());launch(true);check(cudaGetLastError());check(cudaDeviceSynchronize());
        std::vector<float> hd(outputs),hs(outputs);
        check(cudaMemcpy(hd.data(),dense,outputs*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hs.data(),sparse,outputs*sizeof(float),cudaMemcpyDeviceToHost));
        size_t dense_bad=0,sparse_bad=0;
        for(int p=0;p<pairs;++p)for(int r=0;r<64;++r)for(int n=0;n<128;++n) {
            float ref=0;int e=r/32,er=r%32;
            for(int kb=0;kb<KB;++kb) {
                float partial=0;
                for(int k=0;k<128;++k)partial+=xf[((p*2+e)*64+er)*K+kb*128+k]*wf[((p*2+e)*128+n)*K+kb*128+k];
                float scale=as[((p*2+e)*64+er)*KB+kb]*bs[(p*2+e)*KB+kb];
                ref=kb?std::fma(partial,scale,ref):partial*scale;
            }
            size_t i=(size_t(p)*64+r)*128+n;
            dense_bad+=!std::isfinite(hd[i]) || hd[i]!=ref;
            sparse_bad+=!std::isfinite(hs[i]) || hs[i]!=ref;
        }
        if(dense_bad || sparse_bad) {
            std::cout<<"{\"numeric_pass\":false,\"dense_bad\":"<<dense_bad<<",\"sparse_bad\":"<<sparse_bad<<"}\n";return 6;
        }
        for(int i=0;i<10;++i){launch(false);launch(true);}check(cudaDeviceSynchronize());
        cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        std::cout<<std::setprecision(9)<<"{\"numeric_pass\":true,\"outputs_checked\":"<<outputs
            <<",\"pairs\":"<<pairs<<",\"k\":"<<K<<",\"gpu\":\""<<prop.name<<"\",\"sms\":"<<prop.multiProcessorCount
            <<",\"scope\":\"32+32 rows, N128; integer-valued FP8 with per-row/per-K128 power-of-two scales; vector loads and online B packing; serialized K-blocks, no production TMA, communication or E2E claim\",\"blocks\":[";
        for(int order=0;order<4;++order) {
            bool sp=order==1 || order==2;if(order)std::cout<<',';
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
        check(cudaFree(dx));check(cudaFree(dw));check(cudaFree(das));check(cudaFree(dbs));
        check(cudaFree(dense));check(cudaFree(sparse));return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 10;}
}
