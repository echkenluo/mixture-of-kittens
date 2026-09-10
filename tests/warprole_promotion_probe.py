#!/usr/bin/env python3
"""Replay frozen worst W13 tile, recording raw WGMMA partials and two promotions."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile-only', action='store_true',
                        help='build the diagnostic without launching CUDA kernels')
    parser.add_argument('--kittens-include', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'third_party/ThunderKittens/include')
    args = parser.parse_args()
    expected = '0b4af6af2c25e3d4c29aec53ee4237c04cf38891468a34779c1530893bb0fb2e'
    if digest(args.tile) != expected:
        raise ValueError('frozen input hash mismatch')
    if not args.compile_only and (torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0)):
        raise ValueError('one SM90 device required')
    here = Path(__file__).resolve().parent
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    os.environ['MAX_JOBS'] = '1'
    ext = load(name='mok_warprole_promotion_probe',
        sources=[str(here / 'warprole_promotion_probe.cu')],
        extra_include_paths=[str(args.kittens_include)],
        extra_cflags=['-std=c++20'],
        extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '--use_fast_math',
            '--expt-extended-lambda', '--expt-relaxed-constexpr',
            '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing',
            '-DKITTENS_SM90', '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__', '-Xptxas=-v', '-Xptxas=--warn-on-spills'],
        extra_ldflags=['-lcuda'], verbose=True)
    if args.compile_only:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {'compile_only': True, 'gpu_kernels_launched': False,
            'visible_cuda_devices': torch.cuda.device_count(),
            'probe_so_sha256': digest(ext.__file__), 'probe_so_path': ext.__file__,
            'cuda_source_sha256': digest(here / 'warprole_promotion_probe.cu'),
            'python_source_sha256': digest(__file__), 'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda, 'numeric_validated': False,
            'quality_go': False, 'performance_go': False}
        (args.output / 'compile-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    t = torch.load(args.tile, map_location='cpu', weights_only=True)
    x = t['x'].view(torch.uint8).repeat(128, 1).contiguous().view(torch.float8_e4m3fn).cuda()
    sx = t['x_scale'].repeat(128, 1).contiguous().cuda()
    w = t['weight'].unsqueeze(0).contiguous().cuda()
    sw = t['weight_scale'].unsqueeze(0).contiguous().cuda()
    partials, totals = [v.cpu() for v in ext.run(x, w, sx, sw)]
    torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'partials': partials, 'totals': totals}, args.output / 'promotion.pt')
    if not bool(torch.isfinite(partials).all() and torch.isfinite(totals).all()):
        raise ValueError('nonfinite probe output')

    # Exact FP8 products fit FP64 here. This is a mathematical comparator,
    # not a simulation of the Tensor Core internal accumulation order.
    exact = (t['weight'].double().reshape(256, 32, 128)
             * t['x'].double().reshape(1, 32, 128)).sum(-1).T.contiguous()
    scales = (t['x_scale'][None, :] * t['weight_scale'].repeat_interleave(128, 0)).T.contiguous()
    if not bool(((scales == 0) | (scales.abs() >= torch.finfo(torch.float32).tiny)).all()):
        raise ValueError('subnormal scale requires explicit FTZ analysis')
    actual = partials[:, 0, :]
    separate = (actual[0] * scales[0]).clone()
    fused = separate.clone()
    libm = ctypes.CDLL(ctypes.util.find_library('m'))
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float
    for kb in range(1, 32):
        separate = separate + actual[kb] * scales[kb]
        fused = torch.tensor([fmaf(float(actual[kb, c]), float(scales[kb, c]), float(fused[c]))
                              for c in range(256)], dtype=torch.float32)
    checks = {
        'partials_all_rows_equal': bool((partials == partials[:, :1]).all()),
        'totals_all_rows_equal': bool((totals == totals[:, :1]).all()),
        'separate_cpu_gpu_bitexact': torch.equal(separate.view(torch.int32), totals[0, 0].view(torch.int32)),
        'fma_cpu_gpu_bitexact': torch.equal(fused.view(torch.int32), totals[1, 0].view(torch.int32)),
        'separate_matches_archived_mok': torch.equal(totals[0].bfloat16(), t['mok_up'].repeat(128, 1)),
    }
    target = 224
    result = {'tile_sha256': expected, 'cuda_source_sha256': digest(here / 'warprole_promotion_probe.cu'),
        'python_source_sha256': digest(__file__), 'probe_so_sha256': digest(ext.__file__),
        'probe_so_path': ext.__file__, 'outputs_sha256': digest(args.output / 'promotion.pt'),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'shape': {'M': 128, 'N': 256, 'K': 4096}, 'checks': checks,
        'target': {'column': target, 'up_channel': 1504, 'bf16_midpoint': -3.2578125,
            'archived_mok': float(t['mok_up'][target]), 'archived_deepgemm': float(t['deepgemm_up'][target]),
            'separate_fp32': float(totals[0, 0, target]), 'fma_fp32': float(totals[1, 0, target]),
            'separate_bf16': float(totals[0, 0, target].bfloat16()), 'fma_bf16': float(totals[1, 0, target].bfloat16()),
            'partial_actual': actual[:, target].tolist(), 'partial_exact_fp64': exact[:, target].tolist(),
            'scale': scales[:, target].tolist()},
        'partials_vs_exact_fp64': {'max_absolute': float((actual.double() - exact).abs().max()),
            'different_count': int((actual.double() != exact).sum())},
        'fma_matches_archived_deepgemm_columns': int((totals[1, 0].bfloat16() == t['deepgemm_up']).sum()),
        'separate_matches_archived_deepgemm_columns': int((totals[0, 0].bfloat16() == t['deepgemm_up']).sum()),
        'valid_for_promotion_attribution': all(checks.values()),
        'scope': 'single captured route, standalone arithmetic diagnostic; no performance or quality claim',
        'quality_go': False, 'performance_go': False}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise ValueError('probe failed reconstruction checks; attribution invalid')


if __name__ == '__main__':
    main()
