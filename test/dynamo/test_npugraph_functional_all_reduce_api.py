# Owner(s): ["module: dynamo"]
"""Core npugraph_ex checks; algorithm selection needs separate device evidence."""
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.distributed import _functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests

import torch_npu
from torch_npu.distributed import functional_all_reduce

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distributed.functional_all_reduce_test_utils import (
    NpuCollectiveTestCase, assert_close, premul_reference,
)


@contextmanager
def observe_graph():
    counts = {"capture": 0, "replay": 0}
    cls = torch_npu.npu.NPUGraph
    capture, replay = cls.capture_end, cls.replay

    def capture_end(graph, *args, **kwargs):
        result = capture(graph, *args, **kwargs)
        counts["capture"] += 1
        return result

    def replay_graph(graph, *args, **kwargs):
        result = replay(graph, *args, **kwargs)
        counts["replay"] += 1
        return result

    with patch.object(cls, "capture_end", capture_end), patch.object(cls, "replay", replay_graph):
        yield counts


class TestFunctionalAllReduceGraph(NpuCollectiveTestCase):
    def _check_graph(self, group, op, backward=False, dtype=torch.float32):
        def model(x):
            return funcol.wait_tensor(functional_all_reduce(x.clone(), op, group))

        torch.compiler.reset()
        compiled = torch.compile(model, backend="npugraph_ex", fullgraph=True)
        counts = {phase: {"capture": 0, "replay": 0} for phase in ("forward", "backward")}
        for step in range(5):
            value = self.rank + step + 1.0
            if dtype == torch.bool:
                value = step % 2 == 0 and self.rank == 0
            elif dtype == torch.complex64:
                value = complex(value, value + 1)
            x = torch.full((8,), value, dtype=dtype, device=self.device, requires_grad=backward)
            reference = funcol.wait_tensor(funcol.all_reduce(x.detach().clone(), op, group))
            with observe_graph() as observed:
                y = compiled(x)
            for key in observed:
                counts["forward"][key] += observed[key]
            assert_close(y, reference)
            if backward:
                with observe_graph() as observed:
                    (y * (self.rank + 1)).sum().backward()
                for key in observed:
                    counts["backward"][key] += observed[key]
                expected = torch.full_like(x, self.world_size * (self.world_size + 1) / 2)
                assert_close(x.grad, expected)
        for phase in (("forward", "backward") if backward else ("forward",)):
            self.assertGreater(counts[phase]["capture"], 0, (phase, counts))
            self.assertGreater(counts[phase]["replay"], 0, (phase, counts))

    def test_forward_capture_replay(self):
        with self.communication() as group:
            for op in ("sum", "avg", "min", "max"):
                self._check_graph(group, op)

    def test_conversion_capture_replay(self):
        with self.communication() as group:
            for dtype, op in (
                (torch.bool, "sum"), (torch.uint8, "avg"),
                (torch.complex64, "sum"), (torch.complex64, "avg"),
            ):
                self._check_graph(group, op, dtype=dtype)

    def test_sum_backward_capture_replay(self):
        with self.communication() as group:
            self._check_graph(group, "sum", backward=True)

    def test_premul_scalar_changes_capture_replay(self):
        with self.communication() as group:
            def model(x, factor):
                return funcol.wait_tensor(functional_all_reduce(
                    x, "premul_sum", group, premul_factor=factor))

            torch.compiler.reset()
            compiled = torch.compile(model, backend="npugraph_ex", fullgraph=True)
            captures = 0
            for factor in (0.5, 2.0, 0.5):
                counts = {"capture": 0, "replay": 0}
                for step in range(5):
                    x = torch.full((8,), self.rank + step + 1.0, device=self.device)
                    reference = premul_reference(x, factor, group)
                    with observe_graph() as observed:
                        result = compiled(x, factor)
                    for key in counts:
                        counts[key] += observed[key]
                    captures += observed["capture"]
                    assert_close(result, reference)
                # The third stage may reuse the first stage's captured graph.
                self.assertGreater(counts["replay"], 0, (factor, counts))
            self.assertGreater(captures, 0)


if __name__ == "__main__":
    run_tests()
