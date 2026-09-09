"""CPU control-flow regression for the communicating benchmark warmup."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class BurnInTest(unittest.TestCase):
    def run_rank(self, rank, decisions, clocks):
        path = Path(__file__).resolve().parents[2] / "benchmarks/bench_warprole_sched.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "burn_in")
        events = []
        values = iter(decisions)
        ticks = iter(clocks)

        class Decision:
            value = None

            def fill_(self, value):
                self.value = value

            def item(self):
                return self.value

        def broadcast(decision, src):
            self.assertEqual(src, 0)
            expected = next(values)
            if rank == 0:
                self.assertEqual(decision.value, expected)
            else:
                decision.value = expected
            events.append("broadcast")

        namespace = {
            "torch": SimpleNamespace(empty=lambda *a, **k: Decision(), int32=None,
                                     cuda=SimpleNamespace(synchronize=lambda: events.append("sync"))),
            "dist": SimpleNamespace(get_rank=lambda: rank, broadcast=broadcast),
            "time": SimpleNamespace(monotonic=lambda: next(ticks)),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        namespace["burn_in"](lambda: events.append("call"), 2, collective=True)
        self.assertEqual(events, ["broadcast"] + ["call"] * 10 + ["sync", "broadcast"])

    def test_leader_decides_stop(self):
        self.run_rank(0, [1, 0], [0, 1, 3])

    def test_peer_obeys_leader_without_local_deadline_poll(self):
        self.run_rank(1, [1, 0], [10000])


if __name__ == "__main__":
    unittest.main()
