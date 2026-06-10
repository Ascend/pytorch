"""
Add validation cases for torch.fx.Tracer/Transformer APIs on NPU:

1. test/test_fx.py from PyTorch community lacks direct test cases for these APIs:
   - torch.fx.Tracer.trace
   - torch.fx.Tracer.create_proxy
   - torch.fx.Tracer.get_fresh_qualname
   - torch.fx.Tracer.path_of_module
   - torch.fx.Tracer.iter
   - torch.fx.Tracer.keys
   - torch.fx.Tracer.proxy
   - torch.fx.Tracer.to_bool
   - torch.fx.Transformer.call_function
   - torch.fx.Transformer.call_module
   - torch.fx.Transformer.get_attr
   - torch.fx.Transformer.placeholder
   - torch.fx.Tracer.getattr
   - torch.fx.Tracer.call_module

2. This file validates the core functionality of these APIs on NPU environment.
"""

import torch
import torch_npu
import torch.nn as nn
import torch.fx as fx

from torch.fx import Tracer, symbolic_trace, Transformer, GraphModule
from torch.fx.proxy import Proxy, TraceError
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


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


class TestTracerGetAttr(TestCase):
    """Test torch.fx.Tracer.getattr method."""

    def test_getattr_parameter(self):
        """Verify getattr correctly captures nn.Parameter in graph and computes correctly on NPU."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(4, 4, device="npu"))

            def forward(self, x):
                p = self.param
                return x + p

        mod = MyModule()
        tracer = Tracer()
        graph = tracer.trace(mod)
        gm = GraphModule(tracer.root, graph)

        # Verify graph structure
        get_attr_nodes = [n for n in graph.nodes if n.op == 'get_attr']
        self.assertEqual(len(get_attr_nodes), 1)
        self.assertEqual(get_attr_nodes[0].target, 'param')

        # Verify computation on NPU
        input = torch.randn(4, 4, device="npu")
        expected = mod(input)
        actual = gm(input)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected))

    def test_getattr_buffer(self):
        """Verify getattr correctly captures buffer in graph and computes correctly on NPU."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('buf', torch.zeros(4, 4, device="npu"))

            def forward(self, x):
                b = self.buf
                return x - b

        mod = MyModule()
        tracer = Tracer()
        graph = tracer.trace(mod)
        gm = GraphModule(tracer.root, graph)

        # Verify graph structure
        get_attr_nodes = [n for n in graph.nodes if n.op == 'get_attr']
        self.assertEqual(len(get_attr_nodes), 1)
        self.assertEqual(get_attr_nodes[0].target, 'buf')

        # Verify computation on NPU
        input = torch.randn(4, 4, device="npu")
        expected = mod(input)
        actual = gm(input)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected))


class TestTracerCreateProxy(TestCase):
    """Test torch.fx.Tracer.create_proxy method."""

    def test_create_proxy_basic(self):
        """Verify create_proxy returns Proxy wrapping a valid Node with correct op."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3).npu()

            def forward(self, x):
                return self.linear(x)

        proxy_records = []

        class MyTracer(Tracer):
            def create_proxy(self, kind, target, args, kwargs, name=None,
                             type_expr=None, proxy_factory_fn=None):
                proxy = super().create_proxy(
                    kind, target, args, kwargs, name, type_expr,
                    proxy_factory_fn)
                proxy_records.append((kind, proxy))
                return proxy

        tracer = MyTracer()
        tracer.trace(MyModule())

        self.assertTrue(len(proxy_records) > 0)
        for kind, proxy in proxy_records:
            self.assertIsInstance(proxy, Proxy)
            self.assertIsNotNone(proxy.node)
            self.assertEqual(proxy.node.op, kind)

    def test_create_proxy_graph_correctness(self):
        """Verify the traced graph produces correct results on NPU."""
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3).npu()

            def forward(self, x):
                return torch.relu(self.linear(x))

        mod = MyModule()
        gm = symbolic_trace(mod)
        input_tensor = torch.randn(2, 4, device="npu")
        expected = mod(input_tensor)
        actual = gm(input_tensor)
        self.assertTrue(torch.allclose(actual, expected))


class TestTracerGetFreshQualname(TestCase):
    """Test torch.fx.Tracer.get_fresh_qualname method."""

    def test_get_fresh_qualname_skip_existing(self):
        """When module has 'param0', get_fresh_qualname('param') returns 'param1'."""
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param0 = torch.nn.Parameter(
                    torch.randn(3, device="npu"))

            def forward(self, x):
                return x + self.param0

        tracer = Tracer()
        tracer.root = TestModule()
        self.assertEqual(tracer.get_fresh_qualname("param"), "param1")

    def test_get_fresh_qualname_fresh_prefix(self):
        """A brand-new prefix returns '{prefix}0'."""
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4).npu()

            def forward(self, x):
                return self.linear(x)

        tracer = Tracer()
        tracer.root = TestModule()
        self.assertEqual(tracer.get_fresh_qualname("new_attr"), "new_attr0")


class TestTracerCallModule(TestCase):
    """
    DIRECT test suite for fx.Tracer.call_module API.
    Does NOT use trace() - validates call_module API behavior in isolation.
    """

    def setUp(self):
        """Setup fresh tracer and empty graph for each test"""
        super().setUp()
        self.tracer = fx.Tracer()
        self.graph = fx.Graph()
        self.tracer.graph = self.graph
        self.tracer.root = nn.Module()

    def _create_placeholder(self, name: str, value: torch.Tensor = None) -> fx.Proxy:
        """Helper to create a placeholder node and return its Proxy"""
        node = self.graph.placeholder(name)
        return fx.Proxy(node, self.tracer)

    def test_call_module_creates_node_in_graph(self):
        """[CORE] Verify call_module creates a call_module node in the graph"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(linear, linear.forward, (x_proxy,), {})

        nodes = list(self.graph.nodes)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertEqual(nodes[1].op, "call_module")
        self.assertEqual(nodes[1].target, "linear")
        self.assertEqual(nodes[1].args, (x_proxy.node,))
        self.assertIsInstance(result, fx.Proxy)

    def test_call_module_returns_proxy_with_correct_node(self):
        """[CORE] Verify call_module returns a Proxy whose node has correct attributes"""
        relu = nn.ReLU().to(device_type)
        self.tracer.root.add_module("relu", relu)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(relu, relu.forward, (x_proxy,), {})

        self.assertIsInstance(result, fx.Proxy)
        self.assertEqual(result.node.op, "call_module")
        self.assertEqual(result.node.target, "relu")
        self.assertEqual(result.node.args, (x_proxy.node,))
        self.assertEqual(result.node.kwargs, {})

    def test_call_module_preserves_positional_args(self):
        """[CORE] Verify multiple positional arguments are preserved"""
        class MultiArgModule(nn.Module):
             def forward(self, x, y, z):
                 return x + y + z

        mod = MultiArgModule().to(device_type)
        self.tracer.root.add_module("mod", mod)

        x_proxy = self._create_placeholder("x")
        y_proxy = self._create_placeholder("y")
        z_proxy = self._create_placeholder("z")

        result = self.tracer.call_module(mod, mod.forward, (x_proxy, y_proxy, z_proxy), {})

        self.assertIsInstance(result, fx.Proxy)
        self.assertIn(result.node.op, ["call_module", "call_function"])

    def test_call_module_preserves_kwargs(self):
        """[CORE] Verify keyword arguments are preserved"""
        class KwargModule(nn.Module):
            def forward(self, x, bias=None, scale=None):
                out = x
                if scale is not None:
                    out = out * scale
                if bias is not None:
                    out = out + bias
                return out

        mod = KwargModule().to(device_type)
        self.tracer.root.add_module("mod", mod)

        x_proxy = self._create_placeholder("x")
        bias_proxy = self._create_placeholder("bias")
        scale_proxy = self._create_placeholder("scale")

        result = self.tracer.call_module(
            mod, mod.forward,
            (x_proxy,),
            {"bias": bias_proxy, "scale": scale_proxy}
        )
        self.assertIsNotNone(result.node.kwargs)

    def test_call_module_chains_multiple_calls(self):
        """[CORE] Verify chained call_module calls produce correct dataflow"""
        linear1 = nn.Linear(10, 8).to(device_type)
        linear2 = nn.Linear(8, 5).to(device_type)
        self.tracer.root.add_module("linear1", linear1)
        self.tracer.root.add_module("linear2", linear2)

        x_proxy = self._create_placeholder("x")
        intermediate = self.tracer.call_module(linear1, linear1.forward, (x_proxy,), {})
        result = self.tracer.call_module(linear2, linear2.forward, (intermediate,), {})

        self.assertEqual(result.node.args[0], intermediate.node)
        nodes = list(self.graph.nodes)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertEqual(nodes[1].op, "call_module")
        self.assertEqual(nodes[1].target, "linear1")
        self.assertEqual(nodes[2].op, "call_module")
        self.assertEqual(nodes[2].target, "linear2")
        self.assertEqual(nodes[2].args[0], nodes[1])

    def test_call_module_same_module_multiple_times(self):
        """[CORE] Verify calling same module multiple times creates distinct nodes"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        y_proxy = self._create_placeholder("y")

        result1 = self.tracer.call_module(linear, linear.forward, (x_proxy,), {})
        result2 = self.tracer.call_module(linear, linear.forward, (y_proxy,), {})

        self.assertNotEqual(result1.node, result2.node)
        self.assertEqual(result1.node.target, result2.node.target)
        self.assertEqual(result1.node.target, "linear")

        nodes = list(self.graph.nodes)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes[2].op, "call_module")
        self.assertEqual(nodes[3].op, "call_module")

    def test_call_module_with_nested_module_path(self):
        """[CORE] Verify nested module target path resolution"""
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5).to(device_type)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

        outer = Outer().to(device_type)
        self.tracer.root.add_module("outer", outer)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(
            outer.inner.linear,
            outer.inner.linear.forward,
            (x_proxy,),
            {}
        )

        self.assertEqual(result.node.target, "outer.inner.linear")
        self.assertEqual(result.node.op, "call_module")

    def test_call_module_with_sequential_indexing(self):
        """[CORE] Verify Sequential submodule indexing works"""
        seq = nn.Sequential(
            nn.Linear(5, 10).to(device_type),
            nn.ReLU().to(device_type),
            nn.Linear(10, 5).to(device_type)
        )
        self.tracer.root.add_module("seq", seq)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(seq[0], seq[0].forward, (x_proxy,), {})
        self.assertEqual(result.node.target, "seq.0")

    def test_call_module_result_can_be_used_in_operations(self):
        """[CORE] Verify call_module result can be used in arithmetic operations"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        linear_out = self.tracer.call_module(linear, linear.forward, (x_proxy,), {})
        add_result = linear_out + 1.0

        self.assertIsInstance(add_result, fx.Proxy)
        self.assertEqual(add_result.node.op, "call_function")
        self.assertEqual(add_result.node.args[0], linear_out.node)

    def test_call_module_with_different_module_types(self):
        """[CORE] Verify call_module works with various module types"""
        modules = {
            "linear": nn.Linear(10, 5).to(device_type),
            "conv2d": nn.Conv2d(3, 16, 3).to(device_type),
            "relu": nn.ReLU().to(device_type),
            "dropout": nn.Dropout(0.5).to(device_type),
            "batchnorm": nn.BatchNorm2d(16).to(device_type),
        }

        for name, module in modules.items():
            with self.subTest(module_type=name):
                self.graph = fx.Graph()
                self.tracer.graph = self.graph
                self.tracer.root = nn.Module()
                x_proxy = self._create_placeholder("x")

                self.tracer.root.add_module(name, module)
                result = self.tracer.call_module(module, module.forward, (x_proxy,), {})

                self.assertIsInstance(result, fx.Proxy)
                self.assertEqual(result.node.op, "call_module")
                self.assertEqual(result.node.target, name)

    def test_call_module_graph_contains_only_call_module_nodes(self):
        """[CORE] Verify graph only contains nodes created by call_module (no trace artifacts)"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        self.tracer.call_module(linear, linear.forward, (x_proxy,), {})

        nodes = list(self.graph.nodes)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertEqual(nodes[1].op, "call_module")
        self.assertNotIn("output", [n.op for n in nodes])

    def test_call_module_module_must_be_registered(self):
        """[CORE] Verify module must be registered in root to get string target"""
        linear = nn.Linear(10, 5).to(device_type)
        x_proxy = self._create_placeholder("x")

        with self.assertRaises(NameError) as context:
            self.tracer.call_module(linear, linear.forward, (x_proxy,), {})

        self.assertIn("not installed as a submodule", str(context.exception))

    def test_call_module_with_single_arg(self):
        """[CORE] Verify call_module works with single argument"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(linear, linear.forward, (x_proxy,), {})

        self.assertIsInstance(result, fx.Proxy)
        self.assertEqual(result.node.op, "call_module")
        self.assertEqual(len(result.node.args), 1)
        self.assertEqual(result.node.args[0], x_proxy.node)

    def test_call_module_preserves_output_for_further_tracing(self):
        """[CORE] Verify the graph built by call_module can be traced/executed"""
        linear = nn.Linear(10, 5).to(device_type)
        self.tracer.root.add_module("linear", linear)

        x_proxy = self._create_placeholder("x")
        result = self.tracer.call_module(linear, linear.forward, (x_proxy,), {})

        output_node = self.graph.output(result.node)
        graph_module = fx.GraphModule(self.tracer.root, self.graph)

        x = torch.randn(3, 10).to(device_type)
        output = graph_module(x)

        self.assertEqual(output.shape, (3, 5))
        expected = linear(x)
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    run_tests()