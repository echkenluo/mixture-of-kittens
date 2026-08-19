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
from tests.utils import (BF16_TOLERANCE, generate_inputs, get_error_stats,
                         run_forward_reference_bf16, run_fwd_epilogue_reference)

NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
COMM_SMS = int(os.environ.get("BF16_FWD_COMM_SMS", 24))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 4096))
MACROBATCH_SIZE = int(os.environ.get("MACROBATCH_SIZE", 32 * MINIBATCH_SIZE))
WARMUP = int(os.environ.get("BENCH_WARMUP", WARMUP_ITERS))
OUTPUT = os.environ.get("BENCH_OUTPUT", "/mok/bench-sm90-fwd.json")


def rank_max_samples(samples_ms, device):
    t = torch.tensor(samples_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.stack(gathered).max(dim=0).values.cpu().tolist()


def _provenance():
    """Self-contained identity: the tar-synced container tree has no .git, so
    git rev-parse is best-effort only. Primary identity = content hashes."""
    import glob
    import hashlib
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prov = {"frozen_commit_env": os.environ.get("MOK_FROZEN_COMMIT", "unset"),
            "manifest_sha256_env": os.environ.get("MANIFEST_SHA256", "unset (metadata_invalid)"),
            "receipt_sha256_env": os.environ.get("RECEIPT_SHA256", "unset (metadata_invalid)"),
            "bench_gpus_env": os.environ.get("BENCH_GPUS", "unset (metadata_invalid)"),
            "bench_mode_env": os.environ.get("BENCH_MODE", "unset (metadata_invalid)"),
            "frozen_commit_provenance": "external env (host git); unset if launcher omitted it"}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10, cwd=repo)
        prov["git_rev_parse"] = r.stdout.strip() if r.returncode == 0 else \
            f"unavailable ({(r.stderr or '').strip()[:60]})"
    except Exception as e:
        prov["git_rev_parse"] = f"unavailable ({e})"
    with open(os.path.abspath(__file__), "rb") as f:
        prov["harness_sha256"] = hashlib.sha256(f.read()).hexdigest()
    so = sorted(glob.glob(os.path.join(repo, "mok", "_C*.so")))
    if len(so) == 1:
        with open(so[0], "rb") as f:
            prov["so_sha256"] = hashlib.sha256(f.read()).hexdigest()
        prov["so_path"] = so[0]
    else:
        prov["so_sha256"] = f"invalid ({len(so)} .so files, metadata_invalid)"
    return prov


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


def _loadavg():
    # NOTE: read from inside the container; loadavg is not namespaced so it
    # mirrors the host kernel, but naming stays honest about the source.
    with open("/proc/loadavg") as f:
        return f.read().strip()


def main() -> None:
    wall0 = time.time()
    rank, world_size, device = init_distributed()
    # input seed is FIXED inside generate_inputs (Generator.manual_seed(1234+rank));
    # recorded as-is - there is no configurable bench seed.
    input_seed = 1234 + rank

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
    torch.cuda.synchronize()
    t_setup_end = time.time()

    def run_fwd():
        schedule = functional.build_schedule(
            workspace, config, topk_experts, num_local_experts=num_local_experts)
        output, _context = functional.forward(
            config, workspace, schedule, x, router_weights,
            w_shared_gate, w_shared_up, w_shared_down,
            w_routed_gate, w_routed_up, w_routed_down)
        return output

    # --- correctness gate: forward-only reference, same source as
    # test_forward_bf16 (no backward reference; keeps init lean and matched) ---
    (ref_combine_buffer, _rg, _ru, _rh, ref_y_shared) = run_forward_reference_bf16(
        x, topk_experts, w_shared_gate, w_shared_up, w_shared_down,
        w_routed_gate, w_routed_up, w_routed_down)
    ref_out = run_fwd_epilogue_reference(ref_y_shared, ref_combine_buffer,
                                         router_weights)
    del ref_combine_buffer, _rg, _ru, _rh, ref_y_shared
    out = run_fwd()
    torch.cuda.synchronize()
    abs_mean, abs_max, relative = get_error_stats(ref_out, out)
    absolute_tolerance, relative_tolerance = BF16_TOLERANCE
    import math as _math
    gate_pass = (all(_math.isfinite(v) for v in (abs_mean, abs_max, relative))
                 and abs_max <= absolute_tolerance
                 and relative <= relative_tolerance)
    if not gate_pass:
        raise RuntimeError(
            f"correctness gate FAILED on rank {rank}: abs_mean={abs_mean:.6f} "
            f"abs_max={abs_max:.6f} relative={relative:.6f} vs tolerance "
            f"(abs={absolute_tolerance}, rel={relative_tolerance})")
    del ref_out, out
    t_correct_end = time.time()
    snap_start = gpu_snapshot() if rank == 0 else None
    load_start = _loadavg()

    # --- warmup incl. JIT (untimed) ---
    for _ in range(WARMUP):
        run_fwd()
    torch.cuda.synchronize()
    t_warmup_end = time.time()

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
    t_timed_end = time.time()

    # PID namespace note: inside the container /proc/self/status NSpid is a
    # single (container) value - the host PID is NOT visible from here. The
    # host-side launcher captures `docker top` into the run log; acceptance
    # matches NVML host PIDs against that capture, never against these.
    nspid = "?"
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("NSpid"):
                nspid = line.split(":", 1)[1].strip()
                break
    pid_t = torch.tensor([os.getpid()], dtype=torch.int64, device=device)
    pid_g = [torch.empty_like(pid_t) for _ in range(world_size)]
    dist.all_gather(pid_g, pid_t)
    self_pids = {int(t.item()) for t in pid_g}

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
                "macrobatch": MACROBATCH_SIZE, "input_seed_rank0": 1234,  # fixed in generate_inputs (1234+rank)
                "warmup_iters": WARMUP, "timed_iters": TIMED_ITERS,
                "timing_boundary": "build_schedule + forward per iteration; "
                                   "setup/correctness/warmup(JIT) phases excluded and "
                                   "reported separately in wall_phases_s",
                "wall_phases_s": {
                    "setup": round(t_setup_end - wall0, 1),
                    "correctness_gate": round(t_correct_end - t_setup_end, 1),
                    "warmup_jit": round(t_warmup_end - t_correct_end, 1),
                    "timed_region": round(t_timed_end - t_warmup_end, 1),
                    "total": round(t_timed_end - wall0, 1),
                },
                "container_read_loadavg_start": load_start,
                "container_read_loadavg_end": _loadavg(),
                "correctness_gate": {"abs_mean": abs_mean, "abs_max": abs_max,
                                     "relative": relative,
                                     "tolerance_abs_rel": list(BF16_TOLERANCE)},
                "provenance": _provenance(),
                "container_hostname": open("/etc/hostname").read().strip(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu_snapshot_start": snap_start,
                "gpu_snapshot_end": gpu_snapshot(),
                "self_rank_pids_container_ns": sorted(self_pids),
                "self_nspid_rank0": nspid,
                "host_pid_mapping": "host sidecar in host-runs/ (docker top capture)",
                "run_id": os.environ.get("RUN_ID", "unset"),
                "sidecar_basename": f'{os.environ.get("BENCH_TAG", "?")}-{os.environ.get("RUN_ID", "?")}.host',
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
        tmp = OUTPUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=1)
        os.replace(tmp, OUTPUT)  # atomic: no partially-written artifact
        print(f"BENCH|sm90_fwd|comm_sms={COMM_SMS}|p50={p50:.4f}ms|p95={p95:.4f}ms"
              f"|abs_max={abs_max:.5f}|relative={relative:.6f}|out={OUTPUT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
