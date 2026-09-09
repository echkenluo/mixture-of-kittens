"""Execute the owning Python API with a model of mutable workspace storage.

This checks orchestration and exception ownership, not CUDA ordering/liveness.
"""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def model(monkeypatch):
    package = types.ModuleType("_warprole_test_package")
    package.__path__ = []
    functional = types.ModuleType(package.__name__ + ".functional")
    extension = types.SimpleNamespace()
    for name in ("MoKConfig", "MoKFP8RouteWorkspace", "MoKSchedule"):
        setattr(functional, name, type(name, (), {}))
    state = types.SimpleNamespace(held=False, value=0, output=None, fail=False,
                                  acquisitions=0, releases=0)

    def acquire(workspace):
        assert not state.held
        state.held = True
        state.acquisitions += 1

    def release(workspace):
        assert state.held
        state.held = False
        state.releases += 1
        # A subsequent owner may immediately overwrite workspace output.
        state.output.value = -99

    def build(*args, **kwargs):
        assert state.held, "schedule modified outside lease"
        state.value += 1
        return functional.MoKSchedule()

    functional.acquire_workspace_lease = acquire
    functional.release_workspace_lease = release
    functional._validate_build_schedule_inputs = lambda *args, **kwargs: args[2]
    functional._build_schedule_validated = build
    functional.format_trap_record = lambda workspace: None
    package._C = extension
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, functional.__name__, functional)
    name = package.__name__ + ".warprole"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[2] / "mok/warprole.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_validate_forward_inputs", lambda *a, **k: None)

    class Output:
        def __init__(self, value):
            self.value = value

        def clone(self, **kwargs):
            assert state.held, "output read after releasing lease"
            return Output(self.value)

    def kernel(*args, **kwargs):
        assert state.held, "input publication/launch outside lease"
        if state.fail:
            raise RuntimeError("launch failed")
        state.output = Output(state.value)
        return state.output

    monkeypatch.setattr(module, "warprole_forward_leased", kernel)
    workspace = types.SimpleNamespace(num_local_experts=32)
    args = (workspace, object(), object(), *([object()] * 8))
    return module, state, args


def test_schedule_through_output_is_one_transaction(model):
    module, state, args = model
    first = module.warprole_forward_from_topk(*args)
    second = module.warprole_forward_from_topk(*args)
    assert (first.value, second.value) == (1, 2)
    assert first is not second
    assert state.output.value == -99
    assert not state.held
    assert state.acquisitions == state.releases == 2


def test_launch_failure_does_not_release_partially_written_workspace(model):
    module, state, args = model
    state.fail = True
    with pytest.raises(RuntimeError, match="launch failed"):
        module.warprole_forward_from_topk(*args)
    assert state.held and state.releases == 0


def test_bad_metadata_does_not_acquire_workspace(model, monkeypatch):
    module, state, args = model

    def reject(*args, **kwargs):
        raise ValueError("bad metadata")

    monkeypatch.setattr(module, "_validate_forward_inputs", reject)
    with pytest.raises(ValueError, match="bad metadata"):
        module.warprole_forward_from_topk(*args)
    assert state.acquisitions == state.releases == 0
