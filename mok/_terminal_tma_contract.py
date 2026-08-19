"""Pure-Python raw bulk-TMA pointer and interval contract."""

from __future__ import annotations


RAW_BULK_ALIGNMENT = 16
MAX_UINTPTR = (1 << 64) - 1


def _checked_interval(name: str, pointer: int, size: int) -> tuple[int, int]:
    if type(pointer) is not int or pointer <= 0 or pointer > MAX_UINTPTR:
        raise ValueError(f"{name} pointer must be a positive uintptr")
    if pointer % RAW_BULK_ALIGNMENT != 0:
        raise ValueError(f"{name} pointer must be 16-byte aligned")
    if type(size) is not int or size <= 0 or size > MAX_UINTPTR - pointer:
        raise ValueError(f"{name} byte interval is invalid or overflows uintptr")
    return pointer, pointer + size


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def validate_terminal_tma_dispatch_layout(
    x_ptrs: list[int],
    x_bytes_per_rank: list[int],
    x_scale_ptrs: list[int],
    x_scale_bytes_per_rank: list[int],
    *,
    required_x_bytes: int,
    required_x_scale_bytes: int,
    routed_x_pointer: int,
    routed_x_bytes: int,
    routed_x_scale_pointer: int,
    routed_x_scale_bytes: int,
) -> None:
    """Reject unaligned, undersized, overflowing, or overlapping TMA storage."""

    for name, pointers, sizes, required in (
        ("x_ptrs", x_ptrs, x_bytes_per_rank, required_x_bytes),
        (
            "x_scale_ptrs",
            x_scale_ptrs,
            x_scale_bytes_per_rank,
            required_x_scale_bytes,
        ),
    ):
        if not isinstance(pointers, list) or len(pointers) != 4:
            raise ValueError(f"{name} must contain exactly four pointers")
        if not isinstance(sizes, list) or len(sizes) != 4:
            raise ValueError(f"{name} sizes must contain exactly four entries")
        if type(required) is not int or required <= 0:
            raise ValueError(f"{name} required byte count must be positive")
        if any(type(size) is not int or size < required for size in sizes):
            raise ValueError(
                f"{name} per-rank storage must cover the required byte count"
            )

    destinations = (
        (
            "routed_x",
            _checked_interval("routed_x", routed_x_pointer, routed_x_bytes),
        ),
        (
            "routed_x_scale",
            _checked_interval(
                "routed_x_scale",
                routed_x_scale_pointer,
                routed_x_scale_bytes,
            ),
        ),
    )

    sources: list[tuple[str, tuple[int, int]]] = []
    for rank, (pointer, size) in enumerate(zip(x_ptrs, x_bytes_per_rank)):
        name = f"x_ptrs[{rank}]"
        sources.append((name, _checked_interval(name, pointer, size)))
    for rank, (pointer, size) in enumerate(
        zip(x_scale_ptrs, x_scale_bytes_per_rank)
    ):
        name = f"x_scale_ptrs[{rank}]"
        sources.append((name, _checked_interval(name, pointer, size)))

    for source_name, source in sources:
        for destination_name, destination in destinations:
            if _overlaps(source, destination):
                raise ValueError(
                    "terminal TMA source and destination intervals overlap: "
                    f"{source_name} vs {destination_name}"
                )
