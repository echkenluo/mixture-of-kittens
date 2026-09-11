"""CPU storage-lifetime checks; actual CUDA lease is checked in the GPU probe."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch


@pytest.fixture
def arena_module(monkeypatch):
    name = "_mok_scratch_host_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / "mok/scratch.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    empty, zeros = torch.empty, torch.zeros
    monkeypatch.setattr(module.torch, "empty", lambda *a, **k: empty(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(module.torch, "zeros", lambda *a, **k: zeros(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: False)
    return module


def test_reuse_does_not_reset_guard_or_invalidate_old_views(arena_module):
    m = arena_module
    group = SimpleNamespace(group_name="ep4")
    device = torch.device("cuda", 1)
    first = m.get_fp8_scratch_arena(group, device=device, capacity=512)
    view = first.routed_y[:256]
    view.fill_(3)
    first.in_use.fill_(1)
    assert m.get_fp8_scratch_arena(group, device=device, capacity=256) is first
    assert first.in_use.item() == 1
    larger = m.get_fp8_scratch_arena(group, device=device, capacity=1024)
    assert larger is not first and larger.in_use.data_ptr() != first.in_use.data_ptr()
    assert m.get_fp8_scratch_arena(group, device=device, capacity=256) is first
    m.clear_scratch_arena_cache()
    assert torch.all(view == 3) and first.in_use.item() == 1
    assert larger.routed_y.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()


def test_capacity_and_identity_are_checked_before_sharing(arena_module):
    m = arena_module
    group = SimpleNamespace(group_name="ep4")
    device = torch.device("cuda", 0)
    for invalid in (True, 0, -256, 257):
        with pytest.raises(ValueError):
            m.get_fp8_scratch_arena(group, device=device, capacity=invalid)
    arena = m.get_fp8_scratch_arena(group, device=device, capacity=256)
    for kw in ({"group_name":"other","device":device,"capacity":256},
               {"group_name":"ep4","device":torch.device("cuda",1),"capacity":256},
               {"group_name":"ep4","device":device,"capacity":512}):
        with pytest.raises(ValueError):arena.validate(**kw)
