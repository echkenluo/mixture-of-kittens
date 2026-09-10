"""CPU layout planning for the optional SM90 symmetric route arena."""

def fp8_route_arena_layout(num_local_tokens, hidden_size, topk, ep_size):
    """Byte offsets for non-overlapping, 256-byte-aligned route storage."""
    sizes = (
        num_local_tokens * hidden_size,
        num_local_tokens * (hidden_size // 128) * 4,
        num_local_tokens * topk * hidden_size * 2,
        ep_size * num_local_tokens * topk * 4,
        4,
    )
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += (size + 255) // 256 * 256
    return tuple(zip(offsets, sizes)), total


