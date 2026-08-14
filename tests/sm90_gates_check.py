import inspect
import os
from pathlib import Path
import sys

import torch

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fails = 0
# 1) import/registration smoke
import mok.ops as ops
assert hasattr(torch.ops.mok, "all_gather_top_experts"), "registration missing"
print("GATE|import_and_registration|OK")
# 2) direct device check
try:
    ops._sm90_reject("probe"); print("GATE|_sm90_reject|MISSING"); fails += 1
except NotImplementedError:
    print("GATE|_sm90_reject|RAISES_OK")
# 3) full-signature public call through the dispatcher
try:
    ops.mxfp8_quantize(torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16), True, False)
    print("GATE|mxfp8_quantize|MISSING_RAISE"); fails += 1
except NotImplementedError:
    print("GATE|mxfp8_quantize|RAISES_OK")
# 4) sentinel reachability on the undecorated implementations
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
