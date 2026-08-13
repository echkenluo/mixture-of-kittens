import os, sys, inspect, torch
os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")
sys.path.insert(0, "/mok/mixture-of-kittens")
import mok.ops as ops
fails = 0
try:
    ops._sm90_reject("probe"); print("GATE|_sm90_reject|MISSING"); fails += 1
except NotImplementedError:
    print("GATE|_sm90_reject|RAISES_OK")
try:
    ops.mxfp8_quantize(torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16), True, False)
    print("GATE|mxfp8_quantize|MISSING_RAISE"); fails += 1
except NotImplementedError:
    print("GATE|mxfp8_quantize|RAISES_OK")
src = open("/mok/mixture-of-kittens/mok/ops.py").read()
for fn in ("dispatch_mlp_swiglu_combine_fwd_mxfp8",
           "dispatch_mlp_swiglu_combine_bwd_mxfp8",
           "dispatch_mlp_swiglu_combine_bwd_bf16"):
    seg = src[src.index("def " + fn):]
    body = seg[seg.index("\n") + 1:]
    ok = "_sm90_reject" in body[:2000] and body.index("_sm90_reject") < (body.index("_C.") if "_C." in body[:4000] else 4000)
    print(f"GATE|{fn}|{'GATE_BEFORE_LAUNCH_OK' if ok else 'GATE_MISSING'}")
    fails += 0 if ok else 1
print("GATES_REAL_EXIT:", 1 if fails else 0)
sys.exit(1 if fails else 0)
