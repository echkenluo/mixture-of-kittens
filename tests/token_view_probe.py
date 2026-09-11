"""EP4 token-view equivalence and short paired core screen; no model E2E claim."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
from unittest.mock import patch

import torch
import torch.distributed as dist
from mok import _C, functional, warprole
from mok.scratch import get_fp8_scratch_arena
from . import test_warprole_ep4 as base


def forward(ws, state, cfg, fp8, scales, weights, ids, expert_weights, actual):
    functional.acquire_workspace_lease(ws)
    schedule = functional.build_schedule(ws, cfg, ids,
        num_local_experts=base.LOCAL_EXPERTS, expert_padding=64)
    out = warprole.warprole_forward_leased(
        ws, state, schedule, fp8, scales, weights, ids, *expert_weights,
        variant='c2s4')
    result = out[:actual].clone()
    functional.release_workspace_lease(ws)
    return result


def main():
    rank = int(os.environ['RANK']); device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    out = Path(sys.argv[1]); result = {'rank':rank,'verdict':'FAIL_OR_INCOMPLETE','checks':[],'timings':[]}; rc=1
    try:
        torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
        assert dist.get_world_size()==4 and base.TOTAL_EXPERTS==256
        group=dist.group.WORLD; expert_weights=base.make_weights(device)
        def capacity(t):return ((4*t*6+base.LOCAL_EXPERTS*63+255)//256)*256
        arena=get_fp8_scratch_arena(group,device=device,capacity=capacity(8192))
        buffers={}
        for t in (8192,4096,2048,1024):
            cfg=functional.MoKConfig(schedule_capacity_rows=capacity(t),all_gather_top_experts_chunk_bytes=1024)
            ws=functional.get_fp8_route_workspace(cfg,group,device=device,num_local_tokens=t,
                hidden_size=4096,topk=6,num_local_experts=base.LOCAL_EXPERTS,scratch_arena=arena)
            state=warprole.get_warprole_state(ws,group,device=device,capacity=ws.schedule_capacity)
            buffers[t]=(cfg,ws,state)
        for invalid in (0,-256,257,True,16384):
            try:warprole.get_warprole_token_view(buffers[8192][1],group,num_local_tokens=invalid)
            except ValueError:pass
            else:raise AssertionError('invalid token view admitted')
        saved=[]
        cases=[(768,False,False),(1536,False,False),(2304,False,False),(6144,False,False),
               (768,True,False),(1536,False,True),(2304,False,False),(768,False,False)]
        for iteration,(t,empty_rank,all_padding) in enumerate(cases):
            bucket=1<<(t-1).bit_length();cfg,ws,state=buffers[bucket]
            before=torch.cuda.memory_stats(device)['allocation.all.allocated']
            def forbidden(*a,**kw):raise AssertionError('view allocated GPU storage or rendezvoused')
            with patch.object(torch,'empty',forbidden),patch.object(torch,'zeros',forbidden),\
                 patch.object(warprole.symm_mem,'empty',forbidden),patch.object(warprole.symm_mem,'rendezvous',forbidden):
                view=warprole.get_warprole_token_view(ws,group,num_local_tokens=t)
                view_state=warprole.get_warprole_state(view,group,device=device,capacity=ws.schedule_capacity)
                assert warprole.get_warprole_token_view(ws,group,num_local_tokens=t) is view
            assert torch.cuda.memory_stats(device)['allocation.all.allocated']==before
            assert view.in_use is ws.in_use is arena.in_use
            assert view.barrier_target is ws.barrier_target and view.barrier_buffer is ws.barrier_buffer
            assert view_state.push_done is state.push_done and view_state.input_expected_scratch is state.input_expected_scratch
            for name in ('x_buffer','x_scale_buffer','combine_buffer','output','all_gather_top_experts_buffer'):
                assert getattr(view,name).data_ptr()==getattr(ws,name).data_ptr()
                assert getattr(view,name).is_contiguous()
            assert view.all_gather_top_experts_buffer.stride()==(t*6,6,1)
            assert view_state.output.data_ptr()==state.output.data_ptr()
            generator=torch.Generator(device=device).manual_seed(2000+rank*100+iteration)
            hidden=torch.randn((t,4096),device=device,dtype=torch.bfloat16,generator=generator)
            fp8,scales=base.quantize_k128(hidden)
            ids,rw=base.make_routing(base.Case('views',t,1.0,skew=bool(iteration%2),empty_rank=empty_rank,all_padding=all_padding),rank,device)
            padded_fp8=torch.zeros((bucket,4096),dtype=fp8.dtype,device=device);padded_fp8[:t].copy_(fp8)
            padded_scales=torch.ones((bucket,32),dtype=scales.dtype,device=device);padded_scales[:t].copy_(scales)
            padded_ids=torch.full((bucket,6),-1,dtype=ids.dtype,device=device);padded_ids[:t].copy_(ids)
            padded_rw=torch.zeros((bucket,6),dtype=rw.dtype,device=device);padded_rw[:t].copy_(rw)
            args={'bucket':(ws,state,cfg,padded_fp8,padded_scales,padded_rw,padded_ids,expert_weights,t),
                  'view':(view,view_state,cfg,fp8,scales,rw,ids,expert_weights,t)}
            order=('bucket','view','bucket','view') if iteration%2==0 else ('view','bucket','view','bucket')
            results=[forward(*args[a]) for a in order];torch.cuda.synchronize()
            assert all(torch.equal(results[0],v) for v in results[1:]) and torch.isfinite(results[0]).all()
            for held,copy in saved:assert torch.equal(held,copy)
            saved.append((results[-1],results[-1].clone()))
            result['checks'].append({'actual_tokens':t,'bucket_tokens':bucket,'empty_rank':empty_rank,'all_padding':all_padding,
                'bitwise_equal':True,'held_outputs_preserved':True,'no_view_gpu_allocations':True,'common_lease_and_counters':True})
            if iteration<4:
                for repeat in range(2):
                    for arm in (('bucket','view') if repeat==0 else ('view','bucket')):
                        for _ in range(3):forward(*args[arm])
                        dist.barrier();torch.cuda.synchronize()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):forward(*args[arm])
                        end.record();end.synchronize();ms=start.elapsed_time(end)/10
                        slowest=torch.tensor([ms],device=device,dtype=torch.float64);dist.all_reduce(slowest,op=dist.ReduceOp.MAX)
                        result['timings'].append({'actual_tokens':t,'bucket_tokens':bucket,'repeat':repeat,'arm':arm,'local_ms':ms,'max_rank_ms':slowest.item()})
            (out/f'token-view-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        result.update(verdict='OUTPUT_AND_TOKEN_VIEW_PASS',extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
            scope='Synthetic production-size EP4 weights and routes. Complete leased schedule/input-copy/native/output-copy core; input quantization and padding setup excluded from timing. Alternating order, slowest rank. Not SGLang service performance.')
        rc=0
    except BaseException:result['failure']=traceback.format_exc()
    (out/f'token-view-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);os._exit(rc)


if __name__=='__main__':main()
