#!/usr/bin/env python3
# Owner(s): ["module: fx"]

import inspect

import torch
import torch.fx as fx
from torch.fx import _graph_pickler
from torch.testing._internal.common_utils import run_tests, TestCase


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _build_test_graph(device: str = "cpu") -> tuple[fx.GraphModule, torch.Tensor]:
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(8)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)))

    model = SimpleModel().eval().to(device)
    input_tensor = torch.randn(2, 3, 8, 8, device=device)
    traced = fx.symbolic_trace(model)
    return traced, input_tensor


def _node_kinds(graph: fx.Graph) -> list[tuple[str, str]]:
    return [(node.op, str(node.target)) for node in graph.nodes]


def _extract_graph(loaded_obj: object) -> fx.Graph:
    if isinstance(loaded_obj, fx.Graph):
        return loaded_obj
    return loaded_obj.graph  # type: ignore[union-attr]


def _build_options():
    options_cls = getattr(_graph_pickler, "Options", None)
    if options_cls is None:
        return None

    signature = inspect.signature(options_cls)
    has_required_parameter = any(
        name != "self" and param.default is inspect._empty
        for name, param in signature.parameters.items()
    )
    if has_required_parameter:
        return None
    return options_cls()


def _loads_payload(payload: bytes):
    loads_signature = inspect.signature(_graph_pickler.GraphPickler.loads)
    if "fake_mode" in loads_signature.parameters:
        from torch._subclasses.fake_tensor import FakeTensorMode

        return _graph_pickler.GraphPickler.loads(payload, fake_mode=FakeTensorMode())
    return _graph_pickler.GraphPickler.loads(payload)


class TestFxGraphPickler(TestCase):
    def test_graphpickler_dumps_and_loads_cpu(self):
        self.assertTrue(hasattr(_graph_pickler, "GraphPickler"))
        self.assertTrue(hasattr(_graph_pickler.GraphPickler, "dumps"))
        self.assertTrue(hasattr(_graph_pickler.GraphPickler, "loads"))

        traced, _ = _build_test_graph("cpu")
        payload = _graph_pickler.GraphPickler.dumps(traced)
        loaded_obj = _loads_payload(payload)
        loaded_graph = _extract_graph(loaded_obj)
        self.assertEqual(_node_kinds(traced.graph), _node_kinds(loaded_graph))

    def test_graphpickler_dumps_and_loads_npu(self):
        if not _npu_available():
            self.skipTest("NPU not available")

        traced, _ = _build_test_graph("npu")
        payload = _graph_pickler.GraphPickler.dumps(traced)
        loaded_obj = _loads_payload(payload)
        loaded_graph = _extract_graph(loaded_obj)
        self.assertEqual(_node_kinds(traced.graph), _node_kinds(loaded_graph))

    def test_graphpickler_options(self):
        options = _build_options()
        if options is None:
            self.skipTest(
                "torch.fx._graph_pickler.Options unavailable or requires args"
            )

        traced, _ = _build_test_graph("cpu")
        dumps_signature = inspect.signature(_graph_pickler.GraphPickler.dumps)
        if "options" in dumps_signature.parameters:
            payload = _graph_pickler.GraphPickler.dumps(traced, options=options)
        else:
            payload = _graph_pickler.GraphPickler.dumps(traced, options)

        self.assertIsInstance(payload, (bytes, bytearray))
        loaded_obj = _loads_payload(payload)
        self.assertTrue(isinstance(loaded_obj, (fx.Graph, fx.GraphModule)))


if __name__ == "__main__":
    run_tests()
