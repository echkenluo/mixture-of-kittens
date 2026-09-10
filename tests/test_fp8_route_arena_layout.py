"""CPU byte-layout checks; remote pointers and multicast need an EP4 GPU run."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "route_arena", Path(__file__).resolve().parents[1] / "mok/route_arena.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
layout_for = module.fp8_route_arena_layout


class TestRouteArenaLayout(unittest.TestCase):
    def test_alignment_and_disjoint_ranges(self):
        for tokens in (256, 768, 1024, 3072, 4096):
            for ep in (4, 8):
                with self.subTest(tokens=tokens, ep=ep):
                    layout, total = layout_for(tokens, 4096, 6, ep)
                    self.assertEqual(len(layout), 5)
                    self.assertTrue(all(offset % 256 == 0 for offset, _ in layout))
                    self.assertTrue(all(o + n <= p for (o, n), (p, _) in zip(layout, layout[1:])))
                    self.assertLessEqual(layout[-1][0] + layout[-1][1], total)
                    self.assertEqual(total % 256, 0)
                    self.assertLess(total - sum(n for _, n in layout), 5 * 256)

    def test_typed_views_share_storage_without_overlap(self):
        tokens, hidden, topk, ep = 256, 4096, 6, 4
        layout, total = layout_for(tokens, hidden, topk, ep)
        arena = torch.zeros(total, dtype=torch.uint8)
        shapes = [(tokens, hidden), (tokens, hidden // 128),
                  (tokens * topk, hidden), (ep, tokens, topk), (1,)]
        dtypes = [torch.float8_e4m3fn, torch.float32, torch.bfloat16, torch.int32, torch.int32]
        views = []
        for index, ((offset, size), shape, dtype) in enumerate(zip(layout, shapes, dtypes)):
            view = arena.narrow(0, offset, size).view(dtype).view(shape)
            self.assertTrue(view.is_contiguous())
            self.assertEqual(view.data_ptr(), arena.data_ptr() + offset)
            self.assertEqual(view.numel() * view.element_size(), size)
            view.view(torch.uint8).fill_(index + 1)
            views.append(view)
        del arena
        for index, view in enumerate(views):
            self.assertTrue(torch.all(view.view(torch.uint8) == index + 1).item())


if __name__ == "__main__":
    unittest.main()
