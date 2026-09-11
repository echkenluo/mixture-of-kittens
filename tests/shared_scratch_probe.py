"""EP4 exact-output/held-output check followed by fatal cross-shape reentry.

Run in an isolated owned container. The expected fatal trap exits each rank
without CUDA teardown; every per-rank receipt must pass, not just torchrun rc.
"""
import hashlib,json,os,sys,traceback
from pathlib import Path
import torch
import torch.distributed as dist
from mok import functional,warprole,_C
from mok.scratch import get_fp8_scratch_arena
from . import test_warprole_ep4 as base


def main():
    rank=int(os.environ["RANK"]);device=torch.device("cuda",int(os.environ["LOCAL_RANK"]))
    out=Path(sys.argv[1]);receipt={"rank":rank,"verdict":"FAIL_OR_INCOMPLETE"};rc=1
    try:
        torch.cuda.set_device(device);dist.init_process_group("nccl",device_id=device)
        assert dist.get_world_size()==4
        weights=base.make_weights(device);group=dist.group.WORLD
        buffers={};capacities={t:((4*t*6+base.LOCAL_EXPERTS*63+255)//256)*256 for t in (1024,256)}
        arena=get_fp8_scratch_arena(group,device=device,capacity=capacities[1024])
        for shared in (False,True):
            for t in (1024,256):
                cfg=functional.MoKConfig(schedule_capacity_rows=capacities[t],all_gather_top_experts_chunk_bytes=1024)
                ws=functional.get_fp8_route_workspace(cfg,group,device=device,num_local_tokens=t,hidden_size=4096,topk=6,num_local_experts=base.LOCAL_EXPERTS,**({"scratch_arena":arena} if shared else {}))
                state=warprole.get_warprole_state(ws,group,device=device,capacity=capacities[t])
                if shared:
                    assert ws.in_use is arena.in_use
                    assert ws.routed_x.data_ptr()==arena.routed_x.data_ptr()
                    assert ws.routed_x_scale.data_ptr()==arena.routed_x_scale.data_ptr()
                    assert state.hidden.data_ptr()==arena.hidden.data_ptr()
                    assert state.hidden_scale.data_ptr()==arena.hidden_scale.data_ptr()
                    assert state.routed_y.data_ptr()==arena.routed_y.data_ptr()
                buffers[shared,t]=(cfg,ws,state)
        saved=[];checks=[];last_args=None
        for iteration,t in enumerate((1024,256,1024,256)):
            generator=torch.Generator(device=device).manual_seed(1111+iteration+rank*100)
            x=torch.randn((t,4096),device=device,dtype=torch.bfloat16,generator=generator)
            fp8,scale=base.quantize_k128(x);ids,rw=base.make_routing(base.Case("shared",t,1.0,skew=bool(iteration%2)),rank,device)
            results=[]
            for shared in (False,True):
                cfg,ws,state=buffers[shared,t]
                args=(ws,state,cfg,fp8,scale,rw,ids,*weights)
                results.append(warprole.warprole_forward_from_topk(*args))
                if shared:last_args=args
            torch.cuda.synchronize()
            assert torch.isfinite(results[1]).all() and torch.equal(*results)
            for held,copy in saved:assert torch.equal(held,copy)
            saved.append((results[1],results[1].clone()))
            checks.append({"tokens":t,"bitwise_equal":True,"earlier_outputs_preserved":True})
        receipt.update(checks=checks,extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),shared_capacities=capacities)
        holder=torch.cuda.Stream(device=device);contender=torch.cuda.Stream(device=device)
        holder.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(holder):functional.acquire_workspace_lease(buffers[True,1024][1])
        holder.synchronize();assert arena.in_use.item()==1
        caught=None
        try:
            with torch.cuda.stream(contender):warprole.warprole_forward_from_topk(*last_args)
            contender.synchronize()
        except Exception as exc:caught=f"{type(exc).__name__}: {exc}"
        record=buffers[True,256][1].trap_record.tolist()
        assert caught is not None and record==[3,7,0,0,1,rank,0,0],(caught,record)
        receipt.update(verdict="OUTPUT_AND_CROSS_SHAPE_REENTRY_PASS",reentry_error=caught,reentry_trap=record);rc=0
    except BaseException:receipt["failure"]=traceback.format_exc()
    p=out/f"shared-scratch-rank{rank}.json";p.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True);os._exit(rc)


if __name__=="__main__":main()
