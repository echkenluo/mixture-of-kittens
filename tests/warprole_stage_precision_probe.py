"""Compare all frozen dense stages with the actual DeepGEMM/SGLang stack.

Uses original inputs and sampled rows from the independent six-case suite.
Never changes that suite's thresholds or emits a numeric PASS.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from mok import _C
from .test_warprole_ep4 import sglang_activation
from .test_warprole_independent_numeric import SAMPLES, inputs, oracle_rows
from .warprole_numeric_reference import metrics, swiglu_quant
from .warprole_precision_probe import torch_gpu_reference


def sglang_quant(gate):
    function, kind = sglang_activation()
    gate = gate.cuda().contiguous()
    hidden = torch.empty(gate.shape[0], 2048, dtype=torch.float8_e4m3fn, device='cuda')
    scale = torch.empty(gate.shape[0], 16, dtype=torch.float32, device='cuda')
    kwargs = dict(input=gate, output=hidden, output_scale=scale, quant_group_size=128,
                  scale_ue8m0=False, transposed=False, swiglu_limit=10.0, swizzle=False)
    if kind == 'dynamic':
        kwargs['active_tokens'] = torch.tensor([gate.shape[0]], dtype=torch.int32, device='cuda')
    function(**kwargs)
    return hidden.cpu(), scale.cpu()


def main(destination):
    import deep_gemm
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    package = Path(deep_gemm.__file__).parent
    activation, kind = sglang_activation()
    activation_source = Path(inspect.getsourcefile(activation))
    receipt = {
        'extension_sha256': hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'samples': SAMPLES,
        'deepgemm': {'version': deep_gemm.__version__,
                     'python_sha256': hashlib.sha256((package / '__init__.py').read_bytes()).hexdigest(),
                     'binary_sha256': hashlib.sha256((package / '_C.so').read_bytes()).hexdigest(),
                     'entry': 'fp8_gemm_nt', 'recipe': 'runtime default'},
        'sglang_activation': {'kind': kind, 'source': str(activation_source),
                             'sha256': hashlib.sha256(activation_source.read_bytes()).hexdigest()},
        'cases': [], 'verdict': 'DIAGNOSTIC_ONLY_NO_NUMERIC_GO',
    }

    def save(name, snapshots, extra=None):
        reference = snapshots['cpu_reference']
        arms = {arm: metrics(snapshots[arm], reference)
                for arm in ('split', 'c1s6', 'c2s4', 'deepgemm')}
        case = {'name': name, 'arms': arms,
                'deepgemm_vs_mok': metrics(snapshots['deepgemm'], snapshots['split']),
                'split_vs_c1s6': metrics(snapshots['split'], snapshots['c1s6']),
                'split_vs_c2s4': metrics(snapshots['split'], snapshots['c2s4'])}
        if extra:
            case.update(extra)
        path = destination / f'{name}.pt'
        torch.save(snapshots, path)
        case['tensors_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt['cases'].append(case)
        (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('STAGE_PRECISION ' + json.dumps(case), flush=True)

    w13_data = None
    w13_snapshots = None
    for stage, k in (('w13', 4096), ('w2', 2048)):
        data = inputs(k)
        snapshots = {'cpu_reference': oracle_rows(data),
                     'torch_fp32_reference': torch_gpu_reference(data)}
        a, b, a_scale, b_scale, indices = [x.cuda() for x in data]
        active = torch.tensor([128], dtype=torch.int32, device='cuda')
        for arm, symbol in (('split', 'fp8_block_grouped_contiguous_dynamic_out'),
                            ('c1s6', 'fp8_block_warprole_gemm_c1s6_out'),
                            ('c2s4', 'fp8_block_warprole_gemm_c2s4_out')):
            out = torch.full((192, 4096), 12345.0, dtype=torch.bfloat16, device='cuda')
            getattr(_C, symbol)(a, b, a_scale, b_scale, indices, active, out)
            cpu = out.cpu()
            assert bool((cpu[128:] == 12345.0).all())
            snapshots[arm] = cpu[list(SAMPLES)]
        pieces = []
        for start, expert in ((0, 1), (64, 0)):
            out = torch.empty((64, 4096), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_gemm_nt((a[start:start + 64], a_scale[start:start + 64]),
                                 (b[expert], b_scale[expert]), out)
            pieces.append(out.cpu()[[i - start for i in SAMPLES if start <= i < start + 64]])
        snapshots['deepgemm'] = torch.cat(pieces)
        save(stage, snapshots, {'torch_fp32_vs_cpu': metrics(snapshots['torch_fp32_reference'], snapshots['cpu_reference'])})
        if stage == 'w13':
            w13_data, w13_snapshots = data, snapshots

    cpu_hidden, cpu_scale = swiglu_quant(w13_snapshots['cpu_reference'])
    snapshots = {'cpu_reference': cpu_hidden, 'cpu_scale': cpu_scale}
    for arm in ('split', 'deepgemm'):
        snapshots[arm], snapshots[arm + '_scale'] = sglang_quant(w13_snapshots[arm])
    # Separate activation's own CPU/GPU difference from preceding GEMM rounding.
    snapshots['cpu_activation_from_split_gemm'], snapshots['cpu_scale_from_split_gemm'] = swiglu_quant(w13_snapshots['split'])
    a, b, a_scale, b_scale, indices = [x.cuda() for x in w13_data]
    for arm in ('c1s6', 'c2s4'):
        out = torch.ones((192, 2048), dtype=torch.float8_e4m3fn, device='cuda')
        scale = torch.full((192, 16), -1.0, device='cuda')
        getattr(_C, f'fp8_block_warprole_w13_{arm}_out')(
            a, a_scale, b, b_scale, indices, active, out, scale, 10.0)
        out_cpu, scale_cpu = out.cpu(), scale.cpu()
        assert bool((out_cpu[128:].float() == 1).all()) and bool((scale_cpu[128:] == -1).all())
        snapshots[arm] = out_cpu.float()[list(SAMPLES)].to(torch.float8_e4m3fn)
        snapshots[arm + '_scale'] = scale_cpu[list(SAMPLES)]
    for arm in ('split', 'deepgemm', 'c1s6', 'c2s4'):
        assert bool(torch.isfinite(snapshots[arm + '_scale']).all())
        assert bool((snapshots[arm + '_scale'] > 0).all())
    save('activation', snapshots, {
        'scales_vs_cpu': {arm: metrics(snapshots[arm + '_scale'], cpu_scale)
                          for arm in ('split', 'c1s6', 'c2s4', 'deepgemm')},
        'split_activation_vs_cpu_from_same_gemm': metrics(snapshots['split'], snapshots['cpu_activation_from_split_gemm']),
    })
    receipt['complete'] = True
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    main(parser.parse_args().destination)
