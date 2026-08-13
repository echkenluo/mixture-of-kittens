"""Deterministic local tests for the DeepEP comparator (no GPU, no torch, no deep_ep).

Two jobs:
  A. Symmetry - the comparator and bench_sm90_fwd.py must agree on everything
     that could bias a comparison (inputs, config knobs, measurement,
     statistics, correctness truth, JSON shape). Checked by parsing both files,
     so drift fails the test instead of surviving review.
  B. Fail-closed - every unsupported-environment path raises
     UnsupportedEnvironment with an exact message. Checked by executing the
     gate function against stub modules; it is written as a pure function
     precisely so this is possible without the real stack.

Usage: python3 benchmarks/test_symmetry_local.py
"""

import ast
import hashlib
import os
import sys
import tempfile

DIR = os.path.dirname(os.path.abspath(__file__))
MOK = os.path.join(DIR, "bench_sm90_fwd.py")
DEEPEP = os.path.join(DIR, "bench_deepep_fwd.py")

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"SYM_{name}_PASS")
    else:
        FAIL += 1
        print(f"SYM_{name}_FAIL {detail}")


def load(path):
    with open(path) as f:
        src = f.read()
    return src, ast.parse(src)


MOK_SRC, MOK_AST = load(MOK)
DEEP_SRC, DEEP_AST = load(DEEPEP)


def func_src(src, tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def const_map(src, tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.get_source_segment(src, node.value)
    return out


def record_meta_keys(src, tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "record" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == "meta" and isinstance(v, ast.Dict):
                    return {kk.value for kk in v.keys if isinstance(kk, ast.Constant)}
    return set()


def block(src, start_marker, end_marker):
    lines = src.splitlines()
    try:
        i = next(n for n, l in enumerate(lines) if start_marker in l)
        j = next(n for n, l in enumerate(lines) if end_marker in l and n > i)
    except StopIteration:
        return None
    return "\n".join(l.strip() for l in lines[i:j + 1])


# ---------- A. symmetry ----------
for fn in ("rank_max_samples", "gpu_snapshot", "_loadavg"):
    a, b = func_src(MOK_SRC, MOK_AST, fn), func_src(DEEP_SRC, DEEP_AST, fn)
    report(f"A_identical_{fn}", a is not None and a == b,
           "" if a == b else "function source differs between the two harnesses")

mok_c, deep_c = const_map(MOK_SRC, MOK_AST), const_map(DEEP_SRC, DEEP_AST)
SHARED_KNOBS = ["NUM_LOCAL_TOKENS", "HIDDEN_DIM", "INTERMEDIATE_DIM", "NUM_EXPERTS",
                "TOPK", "COMM_SMS", "MINIBATCH_SIZE", "MACROBATCH_SIZE", "WARMUP"]
diff = [k for k in SHARED_KNOBS if mok_c.get(k) != deep_c.get(k)]
report("A_identical_config_knobs", not diff, f"differing: {diff}")

# the comm-SM budget must come from ONE env var on both sides
report("A_single_comm_sm_knob",
       "BF16_FWD_COMM_SMS" in (deep_c.get("COMM_SMS") or "")
       and "BF16_FWD_COMM_SMS" in (mok_c.get("COMM_SMS") or ""),
       "comparator must read the same comm-SM env var as the MoK side")

mok_stats = block(MOK_SRC, "ordered = sorted(samples)", "p95 = ordered[")
deep_stats = block(DEEP_SRC, "ordered = sorted(samples)", "p95 = ordered[")
report("A_identical_statistics", mok_stats is not None and mok_stats == deep_stats,
       "p50/p95 computation differs")

mok_gate = block(MOK_SRC, "abs_mean, abs_max, relative = get_error_stats", "del ref_out, out")
deep_gate = block(DEEP_SRC, "abs_mean, abs_max, relative = get_error_stats", "del ref_out, out")
report("A_identical_correctness_gate", mok_gate is not None and mok_gate == deep_gate,
       "correctness gate differs")

mok_ref = block(MOK_SRC, "run_forward_reference_bf16(", "run_fwd_epilogue_reference(")
deep_ref = block(DEEP_SRC, "run_forward_reference_bf16(", "run_fwd_epilogue_reference(")
report("A_identical_reference_truth", mok_ref is not None and mok_ref == deep_ref,
       "correctness reference construction differs")

mok_keys, deep_keys = record_meta_keys(MOK_SRC, MOK_AST), record_meta_keys(DEEP_SRC, DEEP_AST)
COMPARATOR_ONLY = {"environment_pins", "torch_compile", "permute_implementation",
                   "recv_contract"}
report("A_meta_keys_superset", mok_keys and mok_keys <= deep_keys,
       f"missing on comparator: {sorted(mok_keys - deep_keys)}")
report("A_meta_keys_no_extras", (deep_keys - mok_keys) <= COMPARATOR_ONLY,
       f"unexpected extras: {sorted(deep_keys - mok_keys - COMPARATOR_ONLY)}")

# same measurement size, drawn from the shared module
report("A_same_timed_iters",
       "TIMED_ITERS" in DEEP_SRC and "range(TIMED_ITERS)" in DEEP_SRC
       and "range(TIMED_ITERS)" in MOK_SRC,
       "comparator must use the shared TIMED_ITERS")

# The comparator must not contain any unsupported path in EXECUTABLE code.
# Scanning raw text would flag the disclosure prose (which names MXFP8 and the
# transformer_engine helpers precisely so the handicap is on the record), so
# docstrings are stripped first and only real code is scanned.
def code_only(tree):
    t = ast.parse(ast.unparse(tree))
    for node in ast.walk(t):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
    return ast.unparse(t)


FORBIDDEN = ["mxfp8", "MXFP8", "run_reference_bf16(", "run_bwd", "benchmark_bwd",
             "ElasticBuffer", "moe_permute", "moe_unpermute"]
DEEP_CODE = code_only(DEEP_AST)
hits = [t for t in FORBIDDEN if t in DEEP_CODE]
report("A_no_unsupported_paths", not hits, f"present in code: {hits}")

# ---------- B. fail-closed environment gates ----------
ns = {"os": os, "hashlib": hashlib}
exec(compile(ast.Module(body=[n for n in DEEP_AST.body
                              if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                              and n.name in ("UnsupportedEnvironment", "deepep_py_tree_sha256",
                                             "deepep_ext_sha256", "assert_environment")],
                        type_ignores=[]), DEEPEP, "exec"), ns)
Unsupported = ns["UnsupportedEnvironment"]
assert_environment = ns["assert_environment"]
py_tree_sha = ns["deepep_py_tree_sha256"]
ext_sha = ns["deepep_ext_sha256"]

SITE = tempfile.mkdtemp()
PKG = os.path.join(SITE, "deep_ep")
os.makedirs(PKG)
with open(os.path.join(PKG, "__init__.py"), "w") as f:
    f.write("# stub deep_ep package for gate tests\n")
# the real extension sits BESIDE the package in site-packages, not inside it -
# the stub mirrors that, which is what makes the last two checks meaningful
EXT_PATH = os.path.join(SITE, "deep_ep_cpp.stub.so")
with open(EXT_PATH, "wb") as f:
    f.write(b"stub extension bytes")
GOOD_PY = py_tree_sha(PKG)


class _Ext:
    __file__ = EXT_PATH


GOOD_EXT = ext_sha(_Ext)[0]


class _F:
    pass


def make_torch(version="2.11.0+cu130", grouped=True):
    t = _F()
    t.__version__ = version
    t.nn = _F()
    t.nn.functional = _F()
    if grouped:
        t.nn.functional.grouped_mm = lambda *a, **k: None
    return t


def make_deepep(methods=("get_dispatch_layout", "dispatch", "combine", "set_num_sms", "destroy"),
                sm90=True, has_flag=True, has_buffer=True):
    m = _F()
    m.__file__ = os.path.join(PKG, "__init__.py")
    if has_buffer:
        b = _F()
        for name in methods:
            setattr(b, name, lambda *a, **k: None)
        if has_flag:
            b.is_sm90_compiled = staticmethod(lambda: sm90)
        m.Buffer = b
    return m


GOOD_ENV = {"TORCH_VERSION_PIN": "2.11.0+cu130",
            "DEEPEP_PY_TREE_SHA256": GOOD_PY, "DEEPEP_EXT_SHA256": GOOD_EXT}


def gate_case(name, deepep, torch_mod, env, expect, ext=_Ext):
    try:
        assert_environment(deepep, ext, torch_mod, env)
        report(name, expect is None, "expected rejection, got acceptance")
    except Unsupported as e:
        report(name, expect is not None and str(e) == expect,
               f"message {str(e)!r} != expected {expect!r}")


gate_case("B_accepts_good_env", make_deepep(), make_torch(), GOOD_ENV, None)
gate_case("B_missing_buffer", make_deepep(has_buffer=False), make_torch(), GOOD_ENV,
          "deep_ep.Buffer missing (this comparator targets the classic Buffer API)")
gate_case("B_missing_method",
          make_deepep(methods=("get_dispatch_layout", "dispatch", "combine", "destroy")),
          make_torch(), GOOD_ENV, "deep_ep.Buffer.set_num_sms missing")
gate_case("B_missing_sm90_flag", make_deepep(has_flag=False), make_torch(), GOOD_ENV,
          "deep_ep.Buffer.is_sm90_compiled missing; cannot confirm an SM90 build")
gate_case("B_not_sm90_build", make_deepep(sm90=False), make_torch(), GOOD_ENV,
          "deep_ep build is not SM90-compiled")
gate_case("B_no_grouped_mm", make_deepep(), make_torch(grouped=False), GOOD_ENV,
          "torch.nn.functional.grouped_mm missing")
gate_case("B_missing_torch_pin", make_deepep(), make_torch(),
          {"DEEPEP_PY_TREE_SHA256": GOOD_PY, "DEEPEP_EXT_SHA256": GOOD_EXT},
          "TORCH_VERSION_PIN not provided by the manifest")
gate_case("B_torch_pin_mismatch", make_deepep(), make_torch(version="2.13.0+cu130"), GOOD_ENV,
          "torch 2.13.0+cu130 != pinned 2.11.0+cu130")
gate_case("B_missing_py_pin", make_deepep(), make_torch(),
          {"TORCH_VERSION_PIN": "2.11.0+cu130", "DEEPEP_EXT_SHA256": GOOD_EXT},
          "DEEPEP_PY_TREE_SHA256 not provided by the manifest")
gate_case("B_missing_ext_pin", make_deepep(), make_torch(),
          {"TORCH_VERSION_PIN": "2.11.0+cu130", "DEEPEP_PY_TREE_SHA256": GOOD_PY},
          "DEEPEP_EXT_SHA256 not provided by the manifest")
gate_case("B_py_tree_mismatch", make_deepep(), make_torch(),
          {"TORCH_VERSION_PIN": "2.11.0+cu130", "DEEPEP_PY_TREE_SHA256": "0" * 64,
           "DEEPEP_EXT_SHA256": GOOD_EXT},
          f"deep_ep python tree {GOOD_PY} != pinned {'0' * 64}")
gate_case("B_ext_mismatch", make_deepep(), make_torch(),
          {"TORCH_VERSION_PIN": "2.11.0+cu130", "DEEPEP_PY_TREE_SHA256": GOOD_PY,
           "DEEPEP_EXT_SHA256": "0" * 64},
          f"deep_ep extension {GOOD_EXT} != pinned {'0' * 64}")

# The python tree is a thin wrapper; the kernels live in the extension. If a
# swapped extension did not move any pinned hash, the pin would be decorative -
# which is exactly the hole the first version of this gate had.
with open(EXT_PATH, "wb") as f:
    f.write(b"a DIFFERENT extension binary")
report("B_ext_swap_changes_ext_hash", ext_sha(_Ext)[0] != GOOD_EXT,
       "extension hash unchanged after swapping the binary")
report("B_ext_swap_invisible_to_py_tree_hash",
       py_tree_sha(PKG) == GOOD_PY,
       "python-tree hash moved when only the extension changed (stub layout is wrong)")
with open(os.path.join(PKG, "extra.py"), "w") as f:
    f.write("x = 1\n")
report("B_py_tree_is_content_sensitive", py_tree_sha(PKG) != GOOD_PY,
       "python-tree hash unchanged after adding a file")

EXPECTED = 27
TOTAL = PASS + FAIL
if TOTAL != EXPECTED:
    print(f"SYM_COUNT_FAIL:ran {TOTAL} checks, expected {EXPECTED}")
    FAIL += 1
print(f"SYMMETRY pass={PASS} fail={FAIL}")
sys.exit(1 if FAIL else 0)
