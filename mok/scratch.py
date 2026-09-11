"""Shared SM90 capacity storage; all borrowers use the same production lease."""
from dataclasses import dataclass
import threading

import torch


@dataclass(eq=False, frozen=True, slots=True)
class FP8ScratchArena:
    group_name: str
    device: torch.device
    capacity: int
    routed_x: torch.Tensor
    routed_x_scale: torch.Tensor
    hidden: torch.Tensor
    hidden_scale: torch.Tensor
    routed_y: torch.Tensor
    in_use: torch.Tensor

    def validate(self, *, group_name, device, capacity):
        if (self.group_name != group_name or self.device != device
                or type(capacity) is not int or capacity <= 0
                or capacity % 256 or capacity > self.capacity):
            raise ValueError("scratch arena must match group/device and cover the aligned capacity")


_ARENAS: dict[tuple[str, int], list[FP8ScratchArena]] = {}
_ARENA_LOCK = threading.Lock()


def get_fp8_scratch_arena(group, *, device, capacity):
    """Reuse the smallest covering arena, never resize a live allocation.

    Create all geometries in collective order on every EP rank, outside graph
    capture. Largest-first warmup uses one arena. A later larger shape gets a
    new generation: old views/graphs keep their own storage and lease alive.
    This is cross-shape storage sharing, not a bounded macrobatch ring.
    """
    if type(capacity) is not int or capacity <= 0 or capacity % 256:
        raise ValueError("scratch capacity must be a positive multiple of 256")
    index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", index)
    key = (group.group_name, index)
    with _ARENA_LOCK:
        candidates = [a for a in _ARENAS.get(key, ()) if a.capacity >= capacity]
        if candidates:
            return min(candidates, key=lambda a: a.capacity)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create shared scratch before CUDA graph capture")
        arena = FP8ScratchArena(
            group_name=group.group_name, device=device, capacity=capacity,
            routed_x=torch.empty((capacity, 4096), dtype=torch.float8_e4m3fn, device=device),
            routed_x_scale=torch.empty((capacity, 32), dtype=torch.float32, device=device),
            hidden=torch.empty((capacity, 2048), dtype=torch.float8_e4m3fn, device=device),
            hidden_scale=torch.empty((capacity, 16), dtype=torch.float32, device=device),
            routed_y=torch.empty((capacity, 4096), dtype=torch.bfloat16, device=device),
            in_use=torch.zeros(1, dtype=torch.int32, device=device),
        )
        _ARENAS.setdefault(key, []).append(arena)
        return arena


def clear_scratch_arena_cache():
    """Drop registry references only, after the normal collective cache drain."""
    with _ARENA_LOCK:
        _ARENAS.clear()
