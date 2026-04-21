"""
Add validation cases for torch.fx.Tracer/Transformer APIs on NPU:

1. test/test_fx.py from PyTorch community lacks direct test cases for these APIs:
   - torch.fx.Tracer.trace
   - torch.fx.Tracer.path_of_module
   - torch.fx.Tracer.iter
   - torch.fx.Tracer.keys
   - torch.fx.Tracer.proxy
   - torch.fx.Tracer.to_bool
   - torch.fx.Transformer.call_function
   - torch.fx.Transformer.call_module
   - torch.fx.Transformer.get_attr
   - torch.fx.Transformer.placeholder

2. This file validates the core functionality of these APIs on NPU environment.
"""

import torch
import torch_npu

# Disable NPU JIT compilation to reduce CI time
torch_npu.npu.set_compile_mode(jit_compile=False)

from torch.fx import Tracer, symbolic_trace, Transformer, GraphModule
from torch.fx.proxy import Proxy, TraceError
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTracerTrace(TestCase):
    """Test torch.fx.Tracer.trace method."""

    def test_trace_function(self):
        """Verify trace can symbolically trace a function."""
        def fn(x, y):
            return x + y

        tracer = Tracer()
        graph = tracer.trace(fn)

        graph.lint()
        self.assertIsNotNone(graph)

    def test_trace_module(self):
        """Verify trace can symbolically trace a Module."""
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        tracer = Tracer()
        graph = tracer.trace(MyModule())

        graph.lint()
        self.assertIsNotNone(graph)


class TestTracerPathOfModule(TestCase):
    """Test torch.fx.Tracer.path_of_module method."""

    def test_path_of_module(self):
        """Verify path_of_module can get submodule qualified path."""
        class SubModule(torch.nn.Module):
            def forward(self, x):
                return x

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                return self.sub(x)

        tracer = Tracer()
        mod = MyModule()
        tracer.root = mod

        tracer.submodule_paths = {}
        for name, submod in mod.named_modules():
            if submod is not mod:
                tracer.submodule_paths[submod] = name

        path = tracer.path_of_module(mod.sub)
        self.assertEqual(path, "sub")


class TestTracerIter(TestCase):
    """Test torch.fx.Tracer.iter method."""

    def test_iter_raises_trace_error(self):
        """Verify iter raises TraceError for Proxy objects."""
        graph = torch.fx.Graph()
        node = graph.placeholder("x")

        tracer = Tracer()
        tracer.graph = graph
        proxy = tracer.proxy(node)

        self.assertRaises(TraceError, tracer.iter, proxy)


class TestTracerKeys(TestCase):
    """Test torch.fx.Tracer.keys method."""

    def test_keys_method_callable(self):
        """Verify keys method exists and is callable on Tracer."""
        tracer = Tracer()
        self.assertTrue(hasattr(tracer, 'keys'))
        self.assertTrue(callable(tracer.keys))


class TestTracerProxy(TestCase):
    """Test torch.fx.Tracer.proxy method."""

    def test_proxy_creates_proxy_object(self):
        """Verify proxy wraps Node into a Proxy object."""
        graph = torch.fx.Graph()
        node = graph.placeholder("x")

        tracer = Tracer()
        tracer.graph = graph

        proxy = tracer.proxy(node)
        self.assertIsInstance(proxy, Proxy)
        self.assertEqual(proxy.node, node)


class TestTracerToBool(TestCase):
    """Test torch.fx.Tracer.to_bool method."""

    def test_to_bool_raises_trace_error(self):
        """Verify to_bool raises TraceError for Proxy objects."""
        graph = torch.fx.Graph()
        node = graph.placeholder("x")

        tracer = Tracer()
        tracer.graph = graph
        proxy = tracer.proxy(node)

        self.assertRaises(TraceError, tracer.to_bool, proxy)


class TestTransformerCallFunction(TestCase):
    """Test torch.fx.Transformer.call_function method."""

    def test_call_function(self):
        """Verify Transformer handles call_function nodes correctly."""
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = symbolic_trace(MyModule())
        transformed = Transformer(gm).transform()

        input = torch.randn(4, 4, device="npu")
        result = transformed(input)
        self.assertEqual(result.shape, input.shape)


class TestTransformerCallModule(TestCase):
    """Test torch.fx.Transformer.call_module method."""

    def test_call_module(self):
        """Verify Transformer handles call_module nodes correctly."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        mod = MyModule().to("npu")
        gm = symbolic_trace(mod)
        transformed = Transformer(gm).transform()

        input = torch.randn(2, 4, device="npu")
        self.assertEqual(transformed(input).shape, input.shape)


class TestTransformerGetAttr(TestCase):
    """Test torch.fx.Transformer.get_attr method."""

    def test_get_attr(self):
        """Verify Transformer handles get_attr nodes correctly."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, 4, device="npu"))

            def forward(self, x):
                return x + self.weight

        tracer = Tracer()
        graph = tracer.trace(MyModule())
        gm = GraphModule(tracer.root, graph)
        transformed = Transformer(gm).transform()

        input = torch.randn(4, 4, device="npu")
        result = transformed(input)
        self.assertEqual(result.shape, input.shape)


class TestTransformerPlaceholder(TestCase):
    """Test torch.fx.Transformer.placeholder method."""

    def test_placeholder(self):
        """Verify Transformer handles placeholder nodes correctly."""
        class MyModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        gm = symbolic_trace(MyModule())
        transformed = Transformer(gm).transform()

        x = torch.randn(4, 4, device="npu")
        y = torch.randn(4, 4, device="npu")
        result = transformed(x, y)
        self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    run_tests()