"""DeepEP + PyTorch BF16 forward-only comparator, symmetric to bench_sm90_fwd.py.

Primary comparator per preregistration v2. NOT DeepEP+DeepGEMM (that is only a
correctness reference), and NOT native MoK (no SM100 path exists on H20).

Symmetry contract - every one of these is identical to bench_sm90_fwd.py, and
benchmarks/test_symmetry_local.py asserts it mechanically rather than by
review:
  inputs        generate_inputs(rank, device, ...) with the same arguments;
                the seed is fixed inside it (1234+rank) for both sides
  shape/config  the same env var names, fed by the same manifest keys
  comm budget   BF16_FWD_COMM_SMS drives MoK's comm SMs and DeepEP's
                set_num_sms - one knob, one manifest key
  truth         run_forward_reference_bf16 + run_fwd_epilogue_reference and
                BF16_TOLERANCE - the same reference object, forward only
  measurement   TIMED_ITERS CUDA-event pairs, rank-max across ranks, per-launch
                p50/p95, raw samples in JSON
  boundary      one complete layer forward including its collective
  isolation     one process group per launch, torn down at exit; no state is
                shared between implementations or between launches

Timing-boundary equivalence: MoK times build_schedule+forward; this side times
dispatch -> expert compute -> combine. Both are "one complete layer forward
including its communication". The two are equivalent by that definition, not
by line-for-line correspondence - stated here so the claim is auditable.

DISCLOSED HANDICAP (must travel with any number produced by this file): the
image has no transformer_engine, so the expert-major permutation is done with
plain torch gather/scatter instead of TE's fused moe_permute/moe_unpermute.
That overhead is inside the timed region and is NOT present in the vendor's
own DeepEP benchmark. Any comparison made with this file understates DeepEP by
that amount, and it must never be reported as the vendor-optimal baseline.

RECV CONTRACT (established, not assumed): classic dispatch returns recv_x as
the UNIQUE token rows this rank received, and recv_topk_idx as, per row, the
local expert id for each of its topk slots or -1 where the slot routes
elsewhere. One row can therefore feed SEVERAL local experts, so the number of
(row, expert) routes is >= the number of recv_x rows, and per-expert route
counts must equal recv_num_tokens_per_expert_list. Source: DeepEP's own
classic-API test, tests/legacy/test_intranode.py in deepseek-ai/DeepEP, which
asserts exactly that per-expert equality. assert_recv_contract() re-checks it
at runtime, once, before the timed region, and fails closed.

Consequence for the permutation: index_select on recv_x is a ROUTE EXPANSION
(one row replicated per expert it feeds), not a reordering of already-grouped
rows. It cannot be removed as "overhead" even if recv_x happened to arrive in
expert order - deleting it would silently drop every token routed to more than
one local expert. This note exists to stop a future optimization from doing
exactly that.

UNVERIFIED AT AUTHORING TIME (no GPU run was permitted; each needs a bounded
probe before any matrix launch):
  U1 whether the permutation as written is the cheapest correct expansion for
     this shape (a fused kernel would be cheaper, but none is available here).
  U3 NVL buffer sizing hints for this DeepEP build.
  U4 whether torch.compile helps or hurts here on SM90.
"""

import json
import os
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils import TIMED_ITERS, WARMUP_ITERS, get_num_local_experts, get_tflops, init_distributed
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
OUTPUT = os.environ.get("BENCH_OUTPUT", "/mok/bench-deepep-fwd.json")
TORCH_COMPILE = os.environ.get("DEEPEP_TORCH_COMPILE", "off")


class UnsupportedEnvironment(RuntimeError):
    """Raised for any environment/API mismatch. Never caught, never degraded:
    a comparator that silently falls back measures something else."""


def deepep_py_tree_sha256(pkg_dir):
    """Content hash of the deep_ep PYTHON tree only. That tree is a thin
    wrapper (3 files); it is NOT the thing that executes the kernels, so this
    hash alone must never be treated as the identity of the DeepEP build."""
    import hashlib
    h = hashlib.sha256()
    files = []
    for root, _dirs, names in os.walk(pkg_dir):
        if "__pycache__" in root:
            continue
        for n in sorted(names):
            files.append(os.path.join(root, n))
    for path in sorted(files):
        h.update(path[len(pkg_dir):].encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def deepep_ext_sha256(ext_mod):
    """Content hash of the compiled extension that actually runs the kernels
    (deep_ep_cpp*.so, ~40 MB). Paired with the python-tree hash this is a
    complete content freeze; either one alone is not."""
    import hashlib
    path = os.path.abspath(ext_mod.__file__)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest(), path


def assert_recv_contract(recv_idx, num_recv_per_expert, num_local_experts):
    """One-shot runtime check of the recv contract (see module docstring),
    run before the timed region and never inside it. Fails closed: if the
    layout this comparator assumes is not what DeepEP produced, the number is
    meaningless and must not be produced at all."""
    import torch as _t
    lo = int(recv_idx.min().item())
    hi = int(recv_idx.max().item())
    if lo < -1 or hi >= num_local_experts:
        raise UnsupportedEnvironment(
            f"recv_topk_idx out of contract: range [{lo},{hi}] not within [-1,{num_local_experts - 1}]")
    valid = recv_idx >= 0
    counts = _t.bincount(recv_idx[valid].reshape(-1).to(_t.int64),
                         minlength=num_local_experts)[:num_local_experts].tolist()
    expected = [int(v) for v in num_recv_per_expert]
    if counts != expected:
        raise UnsupportedEnvironment(
            f"recv route counts {counts} != num_recv_tokens_per_expert_list {expected}")
    return {"routes_total": int(valid.sum().item()), "recv_rows": int(recv_idx.shape[0]),
            "per_expert_counts_match": True}


def assert_environment(deep_ep_mod, deep_ep_cpp_mod, torch_mod, env):
    """Pure gate: every supported-ness decision is made here so it can be
    tested without deep_ep, without CUDA and without a GPU. Returns the
    environment pin dict recorded in JSON provenance. Identity is the pair of
    content hashes; paths are recorded for humans and are never compared."""
    buffer_cls = getattr(deep_ep_mod, "Buffer", None)
    if buffer_cls is None:
        raise UnsupportedEnvironment("deep_ep.Buffer missing (this comparator targets the classic Buffer API)")
    for method in ("get_dispatch_layout", "dispatch", "combine", "set_num_sms", "destroy"):
        if not hasattr(buffer_cls, method):
            raise UnsupportedEnvironment(f"deep_ep.Buffer.{method} missing")
    is_sm90 = getattr(buffer_cls, "is_sm90_compiled", None)
    if is_sm90 is None:
        raise UnsupportedEnvironment("deep_ep.Buffer.is_sm90_compiled missing; cannot confirm an SM90 build")
    if not is_sm90():
        raise UnsupportedEnvironment("deep_ep build is not SM90-compiled")
    if not hasattr(torch_mod.nn.functional, "grouped_mm"):
        raise UnsupportedEnvironment("torch.nn.functional.grouped_mm missing")

    pin_torch = env.get("TORCH_VERSION_PIN")
    if not pin_torch:
        raise UnsupportedEnvironment("TORCH_VERSION_PIN not provided by the manifest")
    if torch_mod.__version__ != pin_torch:
        raise UnsupportedEnvironment(f"torch {torch_mod.__version__} != pinned {pin_torch}")

    pin_py = env.get("DEEPEP_PY_TREE_SHA256")
    if not pin_py:
        raise UnsupportedEnvironment("DEEPEP_PY_TREE_SHA256 not provided by the manifest")
    pin_ext = env.get("DEEPEP_EXT_SHA256")
    if not pin_ext:
        raise UnsupportedEnvironment("DEEPEP_EXT_SHA256 not provided by the manifest")
    py_dir = os.path.dirname(os.path.abspath(deep_ep_mod.__file__))
    actual_py = deepep_py_tree_sha256(py_dir)
    if actual_py != pin_py:
        raise UnsupportedEnvironment(f"deep_ep python tree {actual_py} != pinned {pin_py}")
    actual_ext, ext_path = deepep_ext_sha256(deep_ep_cpp_mod)
    if actual_ext != pin_ext:
        raise UnsupportedEnvironment(f"deep_ep extension {actual_ext} != pinned {pin_ext}")
    return {"torch": torch_mod.__version__,
            "deepep_py_tree_sha256": actual_py, "deepep_ext_sha256": actual_ext,
            "deepep_py_dir_informational": py_dir,
            "deepep_ext_path_informational": ext_path,
            "sm90_compiled": True}


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
        prov["so_role"] = "deployment identity only; this comparator does not call MoK kernels"
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


class DeepEpBf16Forward:
    """Classic-Buffer BF16 forward. No MXFP8 path and no backward exist here at
    all - not disabled by a flag, absent - so an unsupported configuration
    cannot be selected by accident."""

    def __init__(self, deep_ep_mod, inputs, num_local_experts, comm_sms, group):
        (self.x, self.topk_experts, self.router_weights,
         w_shared_gate, w_shared_up, w_shared_down,
         w_routed_gate, w_routed_up, w_routed_down, _d_output) = inputs
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts * dist.get_world_size()
        self.comm_sms = comm_sms
        # Buffer.num_sms is a process-global input to get_*_config().  Set it
        # before deriving the size hint: allocating with the default (24) and
        # switching to 32 afterwards makes DeepEP launch more channels than
        # the NVL buffer was sized for (deep_ep.cpp's capacity assertion).
        deep_ep_mod.Buffer.set_num_sms(comm_sms)
        hidden_bytes = self.x.shape[1] * max(self.x.element_size(), 2)
        num_nvl_bytes = 0
        for cfg in (deep_ep_mod.Buffer.get_dispatch_config(dist.get_world_size()),
                    deep_ep_mod.Buffer.get_combine_config(dist.get_world_size())):
            num_nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, dist.get_world_size()),
                                num_nvl_bytes)
        self.buffer = deep_ep_mod.Buffer(group, num_nvl_bytes, 0, explicitly_destroy=True)
        # expert weights transposed once, outside the timed region, exactly as
        # the vendor benchmark does - transpose cost is setup on both sides
        self.w_gate_t = w_routed_gate.detach().transpose(1, 2).contiguous()
        self.w_up_t = w_routed_up.detach().transpose(1, 2).contiguous()
        self.w_down_t = w_routed_down.detach().transpose(1, 2).contiguous()
        self.w_shared_gate = w_shared_gate.detach()
        self.w_shared_up = w_shared_up.detach()
        self.w_shared_down = w_shared_down.detach()
        # checked once, on the correctness-gate call, then never again so the
        # timed region carries no extra work
        self.recv_contract_pending = True
        self.recv_contract = None

    @torch.no_grad()
    def run_fwd(self):
        num_tokens_per_rank, _rdma, num_tokens_per_expert, is_token_in_rank, _ev = \
            self.buffer.get_dispatch_layout(self.topk_experts, self.num_experts)
        recv_x, recv_idx, recv_weights, num_recv_per_expert, handle, _ev = self.buffer.dispatch(
            self.x, num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
            topk_idx=self.topk_experts, topk_weights=self.router_weights,
            expert_alignment=1)

        if self.recv_contract_pending:
            self.recv_contract = assert_recv_contract(recv_idx, num_recv_per_expert,
                                                      self.num_local_experts)
            self.recv_contract_pending = False

        gate_shared = self.x @ self.w_shared_gate.T
        up_shared = self.x @ self.w_shared_up.T
        shared_output = (F.silu(gate_shared) * up_shared) @ self.w_shared_down.T

        # route expansion + expert-major ordering, in plain torch (no
        # transformer_engine in this image - see DISCLOSED HANDICAP above).
        # This is NOT a reordering of already-grouped rows: one recv row can
        # feed several local experts, so it is replicated once per route. See
        # the RECV CONTRACT note - removing this gather would drop tokens.
        valid = recv_idx >= 0
        safe_idx = recv_idx.clamp_min(0)
        flat_expert = torch.where(valid, safe_idx, torch.full_like(safe_idx, self.num_local_experts))
        rows = torch.arange(recv_x.shape[0], device=recv_x.device).unsqueeze(1).expand_as(flat_expert)
        sel = valid.reshape(-1)
        expert_of_pair = flat_expert.reshape(-1)[sel]
        row_of_pair = rows.reshape(-1)[sel]
        weight_of_pair = recv_weights.reshape(-1)[sel].to(torch.float32)
        order = torch.argsort(expert_of_pair, stable=True)
        row_sorted = row_of_pair[order]
        weight_sorted = weight_of_pair[order]
        counts = torch.bincount(expert_of_pair, minlength=self.num_local_experts)[:self.num_local_experts]
        offsets = torch.cumsum(counts, 0).to(torch.int32)
        expert_x = recv_x.index_select(0, row_sorted)

        gate = F.grouped_mm(expert_x, self.w_gate_t, offs=offsets)
        up = F.grouped_mm(expert_x, self.w_up_t, offs=offsets)
        hidden = F.silu(gate) * up
        expert_out = F.grouped_mm(hidden, self.w_down_t, offs=offsets)

        compact = torch.zeros_like(recv_x, dtype=torch.float32)
        compact.index_add_(0, row_sorted, expert_out.to(torch.float32) * weight_sorted.unsqueeze(1))
        routed_output, _w, _ev = self.buffer.combine(compact.to(recv_x.dtype), handle)
        return (routed_output.float() + shared_output.float()).to(torch.bfloat16)

    def destroy(self):
        self.buffer.destroy()


def main() -> None:
    wall0 = time.time()
    rank, world_size, device = init_distributed()
    input_seed = 1234 + rank  # fixed inside generate_inputs, same as the MoK side

    import deep_ep
    import deep_ep_cpp
    env_fp = assert_environment(deep_ep, deep_ep_cpp, torch, os.environ)

    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK,
                             NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    (x, topk_experts, router_weights,
     w_shared_gate, w_shared_up, w_shared_down,
     w_routed_gate, w_routed_up, w_routed_down, _d_output) = inputs

    impl = DeepEpBf16Forward(deep_ep, inputs, num_local_experts, COMM_SMS, dist.group.WORLD)
    run_fwd = impl.run_fwd
    if TORCH_COMPILE == "on":
        run_fwd = torch.compile(run_fwd, mode="max-autotune-no-cudagraphs")
    elif TORCH_COMPILE != "off":
        raise UnsupportedEnvironment(f"DEEPEP_TORCH_COMPILE must be on|off, got {TORCH_COMPILE!r}")
    torch.cuda.synchronize()
    t_setup_end = time.time()

    # --- correctness gate: the SAME forward-only reference the MoK side uses ---
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

    # --- warmup incl. JIT/compile (untimed) ---
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
            "schema": "bench-deepep-fwd.v1",
            "meta": {
                "shape": {"tokens_per_rank": NUM_LOCAL_TOKENS, "hidden": HIDDEN_DIM,
                          "intermediate": INTERMEDIATE_DIM, "experts": NUM_EXPERTS,
                          "topk": TOPK, "world_size": world_size},
                "comm_sms": COMM_SMS, "minibatch": MINIBATCH_SIZE,
                "macrobatch": MACROBATCH_SIZE, "input_seed_rank0": 1234,
                "warmup_iters": WARMUP, "timed_iters": TIMED_ITERS,
                "timing_boundary": "dispatch + expert compute + combine per iteration; "
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
                "environment_pins": env_fp,
                "recv_contract": impl.recv_contract,
                "torch_compile": TORCH_COMPILE,
                "permute_implementation": "torch gather/scatter (no transformer_engine "
                                          "in this image); inside the timed region and "
                                          "absent from the vendor benchmark - understates DeepEP",
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
        print(f"BENCH|deepep_fwd|comm_sms={COMM_SMS}|p50={p50:.4f}ms|p95={p95:.4f}ms"
              f"|abs_max={abs_max:.5f}|relative={relative:.6f}|out={OUTPUT}")

    impl.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
