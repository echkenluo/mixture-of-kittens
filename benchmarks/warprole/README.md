# warprole build helper

`build_sm90.sh` compiles the SM90 extension without a GPU, inside the
`harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130` container (CUDA 13.0, Torch 2.11) on node 18,
and keeps two reports per build under `$ROOT/reports/<head>-<utc-stamp>`:

- `*.ptxas.txt` — the complete `make` output including every `-Xptxas=--verbose` line
  (register counts, spill bytes, shared memory per kernel);
- `*.resources.txt` — `cuobjdump --dump-resource-usage` of the produced `.so`, one line per kernel:
  `<reg> reg <stack> stack <static smem> smem  <mangled name>`.

Usage on node 18 (`$ROOT` defaults to `/home/lenovo/luocc/mok-warprole`, source tree in `$ROOT/src`):

```bash
bash benchmarks/warprole/build_sm90.sh            # build + reports (~3 min)
bash benchmarks/warprole/build_sm90.sh --report-only   # only regenerate the cuobjdump table
```

Baseline numbers on `4e925b2` (2026-09-04): `fp8_block_test::contiguous::kernel` 101 reg / 0 stack,
`fp8_block_terminal_full::kernel` 168 reg / 392 stack, routed `dispatch_kernel` 25 reg,
`combine_kernel` 20 reg. Any new `warprole` kernel must report 0 stack and no spill line.
