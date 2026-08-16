import inspect
import os
from pathlib import Path
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fails = 0
# 1) import/registration smoke
import mok.ops as ops
assert hasattr(torch.ops.mok, "all_gather_top_experts"), "registration missing"
assert hasattr(torch.ops.mok, "fp8_block_routed_dispatch_out"), (
    "FP8 routed dispatch registration missing"
)
assert hasattr(torch.ops.mok, "fp8_block_routed_combine_out"), (
    "FP8 routed combine registration missing"
)
assert hasattr(torch.ops.mok, "fp8_block_grouped_contiguous_out"), (
    "FP8 grouped contiguous registration missing"
)
assert hasattr(torch.ops.mok, "routed_epilogue_out"), (
    "routed epilogue registration missing"
)
print("GATE|import_and_registration|OK")
# 2) fake implementations used by graph capture/compilation
with FakeTensorMode():
    x_fp8 = torch.empty((512, 256), device="cuda", dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((512, 2), device="cuda", dtype=torch.float32)
    routed_x = torch.empty((512, 256), device="cuda", dtype=torch.float8_e4m3fn)
    routed_x_scale = torch.empty((512, 2), device="cuda", dtype=torch.float32)
    m_indices = torch.empty((512,), device="cuda", dtype=torch.int32)
    schedule_rank = torch.empty((512,), device="cuda", dtype=torch.int32)
    schedule_token = torch.empty((512,), device="cuda", dtype=torch.int32)
    num_tokens = torch.empty((1,), device="cuda", dtype=torch.int32)
    tokens_per_expert = torch.empty((2,), device="cuda", dtype=torch.int32)
    routed_y = torch.empty((512, 256), device="cuda", dtype=torch.bfloat16)
    combine_buffer = torch.empty((512, 256), device="cuda", dtype=torch.bfloat16)
    grouped_weight = torch.empty(
        (2, 256, 256), device="cuda", dtype=torch.float8_e4m3fn
    )
    grouped_weight_scale = torch.empty(
        (2, 2, 2), device="cuda", dtype=torch.float32
    )
    grouped_output = torch.empty(
        (512, 256), device="cuda", dtype=torch.bfloat16
    )
    topk_weights = torch.empty((512, 1), device="cuda", dtype=torch.float32)
    pointers = [1, 1, 1, 1]
    torch.ops.mok.fp8_block_routed_dispatch_out(
        x_fp8, pointers, x_scale, pointers, routed_x, routed_x_scale,
        m_indices, schedule_rank, schedule_token, num_tokens,
        tokens_per_expert, 1,
    )
    torch.ops.mok.fp8_block_routed_combine_out(
        routed_y, combine_buffer, pointers, schedule_rank, schedule_token,
        num_tokens, 1,
    )
    torch.ops.mok.fp8_block_grouped_contiguous_out(
        routed_x, grouped_weight, routed_x_scale, grouped_weight_scale,
        m_indices, grouped_output,
    )
    torch.ops.mok.routed_epilogue_out(
        combine_buffer, topk_weights, grouped_output,
    )
print("GATE|fp8_route_fake|OK")
# 3) direct device check
try:
    ops._sm90_reject("probe"); print("GATE|_sm90_reject|MISSING"); fails += 1
except NotImplementedError:
    print("GATE|_sm90_reject|RAISES_OK")
# 4) full-signature public call through the dispatcher
try:
    ops.mxfp8_quantize(torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16), True, False)
    print("GATE|mxfp8_quantize|MISSING_RAISE"); fails += 1
except NotImplementedError:
    print("GATE|mxfp8_quantize|RAISES_OK")
# 5) sentinel reachability on the undecorated implementations
class Sentinel(Exception): pass
def boom(name): raise Sentinel(name)
ops_mod_reject = ops._sm90_reject
ops._sm90_reject = boom
for name in ("dispatch_mlp_swiglu_combine_fwd_mxfp8",
             "dispatch_mlp_swiglu_combine_bwd_mxfp8",
             "dispatch_mlp_swiglu_combine_bwd_bf16"):
    op = getattr(ops, name)
    fn = getattr(op, "_init_fn", None) or getattr(op, "_fn", None) or op
    try:
        sig = inspect.signature(fn)
        kwargs = {k: None for k in sig.parameters}
        fn(**kwargs)
        print(f"GATE|{name}|MISSING_RAISE"); fails += 1
    except Sentinel:
        print(f"GATE|{name}|SENTINEL_REACHED_OK")
    except Exception as e:
        print(f"GATE|{name}|NOT_REACHED:{type(e).__name__}"); fails += 1
ops._sm90_reject = ops_mod_reject
print("GATES_REAL_EXIT:", 1 if fails else 0)
sys.exit(1 if fails else 0)
