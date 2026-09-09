"""CPU-only geometry for distributed warp-role fixtures."""

import math


def schedule_multiplier(tokens, topk, ep_size, local_experts, minimum=0.5):
    """Cover all routes plus expert padding with an M256-aligned capacity.

    The workspace capacity is exactly T * topk * ceil(EP * multiplier).
    For M64 token buckets and top-6, an odd factor can violate M256 even
    though every expert segment is M64 aligned.
    """
    routes = tokens * topk
    factor = max(ep_size + math.ceil(local_experts * 63 / routes),
                 math.ceil(ep_size * minimum))
    quantum = 256 // math.gcd(routes, 256)
    factor = math.ceil(factor / quantum) * quantum
    return factor / ep_size
