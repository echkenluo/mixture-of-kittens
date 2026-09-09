"""Fatal, isolated-process check of the real owning API's busy-workspace guard.

torchrun --standalone --nproc-per-node=4 -m tests.warprole_reentry_probe /results

A first stream acquires and retains the lease. A second stream calls the real
owning forward entry. PASS requires the host-visible REENTRANT/LEASE record,
including this rank and observed in_use=1, plus a raised CUDA error. Generic
crashes, timeouts, and missing receipts do not pass. No CUDA/NCCL cleanup is
attempted after the deliberately fatal trap; each rank exits with os._exit.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch
import torch.distributed as dist

from . import test_warprole_ep4 as base


def main():
    output_dir = Path(sys.argv[1])
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    receipt = {"rank": rank, "verdict": "FAIL_OR_INCOMPLETE"}
    result_file = output_dir / f"reentry-rank{rank}.json"
    exit_code = 1
    try:
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        base.require_warprole(device)
        binary = Path(base._C.__file__)
        receipt.update(
            extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
            world_size=dist.get_world_size(),
        )
        harness = base.build_harness(base.CASES["small_64"], rank, device, fresh=True)
        # Confirm valid metadata and a successful real forward before injecting
        # ownership conflict. This output comparison is not the negative gate.
        base.compare_arms(harness, "c2s4", "reentry-control")
        holder = torch.cuda.Stream(device=device)
        contender = torch.cuda.Stream(device=device)
        holder.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(holder):
            base.functional.acquire_workspace_lease(harness.workspace)
        holder.synchronize()
        assert int(harness.workspace.in_use.item()) == 1
        receipt["holder_acquired"] = True
        caught = None
        try:
            with torch.cuda.stream(contender):
                base.run_warprole(harness, "c2s4")
            contender.synchronize()
        except Exception as exc:
            caught = f"{type(exc).__name__}: {exc}"
        # The trap record is pinned CPU memory; do not touch CUDA after failure.
        record = harness.workspace.trap_record.tolist()
        receipt.update(error=caught, trap_record=record)
        expected_record = [3, 7, 0, 0, 1, rank, 0, 0]
        assert caught is not None and record == expected_record, receipt
        receipt["verdict"] = "EXPECTED_REENTRANT_TRAP_PASS"
        exit_code = 0
    except BaseException:
        receipt["failure"] = traceback.format_exc()
    temporary = result_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(result_file)
    print(json.dumps(receipt), flush=True)
    os._exit(exit_code)


if __name__ == "__main__":
    main()
