"""SM90 v1 BF16-forward-only benchmark harness.

Differences from bench_mok.py, per the perf-contract P0 review:
- Constructs ONLY what the SM90 port supports: no mxfp8_quantize, no
  backward. Anything unsupported raises; nothing is swallowed.
- Emits JSON with RAW per-iteration rank-max samples plus per-launch
  p50/p95 (100 timed iterations). Cross-launch aggregation happens outside,
  over >=5 launches, and is labeled as such.
- Records environment metadata: git commit, container, GPU clocks/power,
  concurrent compute processes, seed, world size, timing boundary.
- Timed region = build_schedule + forward per iteration (bench_mok
  semantics). Process/NCCL/JIT init and the correctness gate happen before
  any timed event and are reported separately as init_wall_s.
"""

import json
import os
import statistics
import subprocess
import time

import torch
import torch.distributed as dist

from benchmarks.utils import TIMED_ITERS, WARMUP_ITERS, get_num_local_experts, get_tflops, init_distributed
from mok import functional
from tests.utils import BF16_TOLERANCE, generate_inputs, run_reference_bf16

NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
COMM_SMS = int(os.environ.get("BF16_FWD_COMM_SMS", 24))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 4096))
MACROBATCH_SIZE = int(os.environ.get("MACROBATCH_SIZE", 32 * MINIBATCH_SIZE))
SEED = int(os.environ.get("BENCH_SEED", 20260813))
WARMUP = int(os.environ.get("BENCH_WARMUP", WARMUP_ITERS))
OUTPUT = os.environ.get("BENCH_OUTPUT", "/mok/bench-sm90-fwd.json")


def rank_max_samples(samples_ms, device):
    t = torch.tensor(samples_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.stack(gathered).max(dim=0).values.cpu().tolist()


def gpu_snapshot():
    out = {}
    for key, args in (
        ("gpus", "--query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu"),
        ("compute_apps", "--query-compute-apps=pid,used_memory"),
    ):
        r = subprocess.run(["nvidia-smi", args, "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        out[key] = r.stdout.strip().splitlines()
    return out


def main() -> None:
    wall0 = time.time()
    rank, world_size, device = init_distributed()
    torch.manual_seed(SEED + rank)

    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK,
                             NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    (x, topk_experts, router_weights,
     w_shared_gate, w_shared_up, w_shared_down,
     w_routed_gate, w_routed_up, w_routed_down, _d_output) = inputs

    config = functional.MoKConfig(
        fwd_num_comm_sms=COMM_SMS,
        bwd_num_comm_sms=COMM_SMS,  # unused: forward-only harness
        minibatch_size=MINIBATCH_SIZE,
        macrobatch_size=MACROBATCH_SIZE,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=x.device,
        num_local_tokens=x.shape[0], hidden_size=x.shape[1],
        topk=topk_experts.shape[1])

    def run_fwd():
        schedule = functional.build_schedule(
            workspace, config, topk_experts, num_local_experts=num_local_experts)
        output, _context = functional.forward(
            config, workspace, schedule, x, router_weights,
            w_shared_gate, w_shared_up, w_shared_down,
            w_routed_gate, w_routed_up, w_routed_down)
        return output

    # --- correctness gate (fail loud, before any timing) ---
    reference = run_reference_bf16(*inputs)
    ref_out = reference[0] if isinstance(reference, (tuple, list)) else reference
    out = run_fwd()
    torch.cuda.synchronize()
    rel = ((out.float() - ref_out.float()).abs()
           / ref_out.float().abs().clamp_min(1e-3)).max().item()
    if rel > BF16_TOLERANCE:
        raise RuntimeError(
            f"correctness gate FAILED on rank {rank}: max_rel {rel:.6f} > "
            f"tolerance {BF16_TOLERANCE}")
    del reference, ref_out, out
    init_wall_s = time.time() - wall0
    snap_start = gpu_snapshot() if rank == 0 else None

    # --- warmup (untimed) ---
    for _ in range(WARMUP):
        run_fwd()

    # --- timed region ---
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(TIMED_ITERS)]
    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    for start, end in events:
        start.record()
        run_fwd()
        end.record()
    torch.cuda.synchronize()
    dist.barrier()

    samples = rank_max_samples([s.elapsed_time(e) for s, e in events], device)
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]

    if rank == 0:
        record = {
            "schema": "bench-sm90-fwd.v1",
            "meta": {
                "shape": {"tokens_per_rank": NUM_LOCAL_TOKENS, "hidden": HIDDEN_DIM,
                          "intermediate": INTERMEDIATE_DIM, "experts": NUM_EXPERTS,
                          "topk": TOPK, "world_size": world_size},
                "comm_sms": COMM_SMS, "minibatch": MINIBATCH_SIZE,
                "macrobatch": MACROBATCH_SIZE, "seed": SEED,
                "warmup_iters": WARMUP, "timed_iters": TIMED_ITERS,
                "timing_boundary": "build_schedule + forward per iteration; "
                                   "init and correctness gate excluded (see init_wall_s)",
                "init_wall_s": round(init_wall_s, 1),
                "correctness_gate": {"max_rel": rel, "tolerance": BF16_TOLERANCE},
                "git_commit": os.environ.get("MOK_GIT_COMMIT", "unknown"),
                "container_hostname": open("/etc/hostname").read().strip(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu_snapshot_start": snap_start,
                "gpu_snapshot_end": gpu_snapshot(),
                "statistics_semantics": "p50/p95 over 100 per-launch rank-max "
                                        "samples; cross-launch aggregation is "
                                        "computed externally over >=5 launches",
            },
            "samples_ms": [round(s, 4) for s in samples],
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "tflops_p50": round(get_tflops(p50, NUM_LOCAL_TOKENS, TOPK,
                                           HIDDEN_DIM, INTERMEDIATE_DIM), 2),
        }
        with open(OUTPUT, "w") as f:
            json.dump(record, f, indent=1)
        print(f"BENCH|sm90_fwd|comm_sms={COMM_SMS}|p50={p50:.4f}ms|p95={p95:.4f}ms"
              f"|max_rel={rel:.6f}|out={OUTPUT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
