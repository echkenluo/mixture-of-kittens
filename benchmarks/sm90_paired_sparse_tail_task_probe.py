"""One historical variable-tail subset with full W2 N4096/K2048 task traversal."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

os.environ['MOK_SM90_EXPERIMENTAL'] = '1'
import torch
from mok import _C


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    root = Path('/work')
    torch.set_num_threads(1)
    assert sha(_C.__file__) == '57f884418459de5ee300d326c42899c4ccfec31fc0fd6682ffd387931c0548b3'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    assert 'H20' in torch.cuda.get_device_name()
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == '21080c38-1b6d-6cb7-0a91-5181e1a5383d'
    cfg = json.loads((root/'sm90_paired_sparse_tail_route_case.json').read_text())
    eligible = cfg['eligible']
    MT, E, K, N, KB = len(eligible), len(cfg['counts']), 2048, 4096, 16
    assert MT % 2 == 0 and all(1 <= r <= 32 for _, r in eligible)
    pairs = MT // 2
    expert_ids = torch.tensor([e for e, _ in eligible], dtype=torch.long)
    lengths = torch.tensor([r for _, r in eligible], dtype=torch.long)
    valid = torch.arange(32)[None, :] < lengths[:, None]
    task_rows = [[i, MT-1-i, eligible[i][0], eligible[MT-1-i][0], eligible[i][1], eligible[MT-1-i][1]] for i in range(pairs)]
    assert sorted(v for row in task_rows for v in row[:2]) == list(range(MT))
    gen = torch.Generator().manual_seed(20260910)
    xf = torch.randint(-2, 3, (MT, 64, K), generator=gen, dtype=torch.int8).float()
    xf[torch.arange(64)[None, :] >= lengths[:, None]] = 0
    wf = torch.randint(-2, 3, (E, N, K), generator=gen, dtype=torch.int8).float()
    ai = torch.arange(MT * 64 * KB).reshape(MT, 64, KB)
    bi = torch.arange(E * (N//128) * KB).reshape(E, N//128, KB)
    ac = torch.pow(2., ((ai * 7 + ai // KB) % 5 - 2).float())
    bc = torch.pow(2., ((bi * 3 + bi // KB) % 5 - 2).float())
    # Integer FP8 and dyadic scales make all FP32 partials and promotions exact;
    # this checks layout, metadata, scaling and output rounding, not model quality.
    reference = torch.zeros(MT, 32, N)
    for kb in range(KB):
        q = torch.bmm(xf[:, :32, kb*128:(kb+1)*128].contiguous(),
                      wf[expert_ids, :, kb*128:(kb+1)*128].transpose(1, 2).contiguous())
        scale = ac[:, :32, kb, None] * bc[expert_ids, :, kb].repeat_interleave(128, dim=1)[:, None, :]
        reference += q * scale
    reference = reference.bfloat16()
    xc, wc = xf.reshape(MT*64, K).to(torch.float8_e4m3fn), wf.to(torch.float8_e4m3fn)
    identities = {name: tensor_sha(t) for name, t in [('x', xc), ('w', wc), ('as', ac), ('bs', bc)]}
    x, w, sa, sb = xc.cuda(), wc.cuda(), ac.reshape(MT*64, KB).cuda(), bc.cuda()
    indices = expert_ids.to(device='cuda', dtype=torch.int32).repeat_interleave(64)
    active = torch.tensor([MT*64], dtype=torch.int32, device='cuda')
    dense = torch.full((MT*64, N), float('nan'), dtype=torch.bfloat16, device='cuda')
    paired = torch.zeros_like(dense)
    paired.view(MT, 64, N)[:, :32] = float('nan')
    lib = ctypes.CDLL(str(root/'paired-tail-probe.so'))
    lib.paired_create_tasks.argtypes = [ctypes.c_uint64]*8 + [ctypes.c_int]*5 + [ctypes.POINTER(ctypes.c_void_p)]
    lib.paired_create_tasks.restype = ctypes.c_int
    lib.paired_launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.paired_launch.restype = ctypes.c_int
    lib.paired_error.restype = ctypes.c_char_p
    lib.paired_destroy.argtypes = [ctypes.c_void_p]
    lib.paired_smem.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    tasks = torch.tensor(task_rows, dtype=torch.int32, device='cuda')
    input_ready = torch.full((MT,), 16, dtype=torch.int32, device='cuda')
    output_ready = torch.zeros(MT, dtype=torch.int32, device='cuda')
    rc = lib.paired_create_tasks(x.data_ptr(), w.data_ptr(), sa.data_ptr(), sb.data_ptr(), paired.data_ptr(),
                                 tasks.data_ptr(), input_ready.data_ptr(), output_ready.data_ptr(),
                                 pairs, MT*64, E, N, K, ctypes.byref(handle))
    assert rc == 0, lib.paired_error().decode()
    record = {'scope': 'Historical variable tails <=32, full W2 N4096/K2048, nonadjacent tile/expert task descriptors, persistent N traversal and dual output-ready publication. Integer FP8/dyadic scales, graph timing. Input counters pre-ready; no real W13 producer, pair construction, full-tile fallback, communication or E2E claim',
              'pairs': pairs, 'k': K, 'n': N, 'input_sha256': identities, 'route_case': cfg, 'tasks': task_rows, 'valid_rows': int(lengths.sum()), 'graph_tasks': pairs*(N//256),
              'baseline_commit': 'fc14526e45b32f08b4fee1a95e1cd6a785d4d6b1',
              'baseline_so_sha256': sha(_C.__file__), 'candidate_so_sha256': sha(root/'paired-tail-probe.so'),
              'script_sha256': sha(__file__), 'dynamic_smem_bytes': lib.paired_smem(), 'blocks': []}

    def save():
        (root/'result.json').write_text(json.dumps(record, indent=2)+'\n')

    def baseline():
        _C.fp8_block_warprole_gemm_c2s4_out(x, w, sa, sb, indices, active, dense)

    def candidate():
        status = lib.paired_launch(handle, torch.cuda.current_stream().cuda_stream)
        assert status == 0, status

    try:
        baseline(); torch.cuda.synchronize()
        record['baseline_completed'] = True
        save()
        candidate(); torch.cuda.synchronize()
        for name, out in [('c2s4', dense), ('paired_tma', paired)]:
            actual = out.cpu().view(MT, 64, N)[:, :32][valid].contiguous()
            record[name+'_numeric'] = {'finite': bool(torch.isfinite(actual).all()),
                                      'mismatches': int((actual.view(torch.int16) != reference[valid].contiguous().view(torch.int16)).sum()),
                                      'output_sha256': tensor_sha(actual)}
        record['reference_sha256'] = tensor_sha(reference[valid])
        record['outputs_checked'] = int(lengths.sum())*N
        record['numeric_pass'] = all(record[n+'_numeric']['finite'] and record[n+'_numeric']['mismatches'] == 0
                                     for n in ('c2s4', 'paired_tma'))
        record['initial_output_ready'] = output_ready.cpu().tolist()
        assert record['initial_output_ready'] == [N//128]*MT
        save()
        assert record['numeric_pass'], 'numeric mismatch; stop before timing'
        for _ in range(10): baseline(); candidate()
        torch.cuda.synchronize()
        graphs = {}
        for name, fn in [('c2s4', baseline), ('paired_tma', candidate)]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20): fn()
            graphs[name] = graph
        torch.cuda.synchronize()
        for name in ('c2s4', 'paired_tma', 'paired_tma', 'c2s4'):
            samples = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); graphs[name].replay(); end.record(); end.synchronize()
                samples.append(start.elapsed_time(end)/20)
            record['blocks'].append({'arm': name, 'ms_per_call': samples})
            save()
        record['final_output_ready'] = output_ready.cpu().tolist()
        record['expected_final_ready'] = (1+10+2*20*20)*(N//128)
        assert record['final_output_ready'] == [record['expected_final_ready']]*MT
        record['complete'] = True
        save()
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            lib.paired_destroy(handle)


if __name__ == '__main__':
    main()
