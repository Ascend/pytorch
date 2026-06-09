"""
Add validation cases for torch.fx.Tracer APIs on NPU:

1. PyTorch community lacks sufficient and direct API validations for
   fx.Tracer.create_arg and fx.Tracer.create_args_for_root, so this file is added.

2. This file validates the following APIs:
   - torch.fx.Tracer.create_arg
   - torch.fx.Tracer.create_args_for_root
   (extendable for other tracer APIs such as call_module, getattr, etc.)
"""

import torch
import torch.fx as fx
import torch.nn as nn
import torch_npu
from torch.testing._internal.common_utils import run_tests, TestCase


class TwoArgModule(nn.Module):
    def forward(self, x, y):
        return x + y


class DefaultArgModule(nn.Module):
    def forward(self, x, scale=2.0):
        return x * scale


class TestTracerCreateArg(TestCase):
    """
    Test suite for fx.Tracer.create_arg and fx.Tracer.create_args_for_root methods.
    Validates that create_arg correctly processes tensors, containers,
    basic types, and NPU tensors during symbolic tracing;
    and that create_args_for_root correctly introspects function/module
    signatures and creates corresponding placeholder proxy nodes.
    """

    def _get_placeholder_nodes(self, graph):
        """Helper to extract placeholder nodes from the graph"""
        return [node for node in graph.nodes if node.op == "placeholder"]

    def _get_constant_nodes(self, graph):
        """Helper to extract constant nodes from the graph"""
        return [
            node
            for node in graph.nodes
            if node.op == "get_attr" or "tensor_constant" in node.name
        ]

    def test_create_arg_tensor_to_constant_node(self):
        """Test that a torch.Tensor is converted to a constant graph node (get_attr)"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        tensor = torch.randn(2, 3).npu()
        result = tracer.create_arg(tensor)

        self.assertIsInstance(result, fx.Node)
        self.assertEqual(result.op, "get_attr")
        self.assertIn("_tensor_constant", result.name)

    def test_create_arg_tensor_node_in_graph(self):
        """Test that the created tensor node is added to the computation graph"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModule()
        tracer = fx.Tracer()
        graph = tracer.trace(model)

        tensor = torch.randn(2, 3).npu()
        result = tracer.create_arg(tensor)

        node_names = [node.name for node in graph.nodes]
        self.assertIn(result.name, node_names)

    def test_create_arg_tensor_list_to_node_list(self):
        """Test that a list of Tensors is converted to a list of graph nodes"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        tensor_list = [torch.randn(2, 2).npu(), torch.randn(2, 2).npu()]
        result = tracer.create_arg(tensor_list)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        for elem in result:
            self.assertIsInstance(elem, fx.Node)
            self.assertEqual(elem.op, "get_attr")

    def test_create_arg_dict_to_node_dict(self):
        """Test that a dict of Tensors is converted to a dict of graph nodes"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        tensor_dict = {
            "first": torch.randn(2, 2).npu(),
            "second": torch.randn(2, 2).npu()
        }
        result = tracer.create_arg(tensor_dict)

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"first", "second"})

        for value in result.values():
            self.assertIsInstance(value, fx.Node)
            self.assertEqual(value.op, "get_attr")

    def test_create_arg_tuple_to_node_tuple(self):
        """Test that a tuple of Tensors is converted to a tuple of graph nodes"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        tensor_tuple = (torch.randn(2, 2).npu(), torch.randn(2, 2).npu())
        result = tracer.create_arg(tensor_tuple)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        for elem in result:
            self.assertIsInstance(elem, fx.Node)
            self.assertEqual(elem.op, "get_attr")

    def test_create_arg_deeply_nested_container(self):
        """Test recursive processing of deeply nested Tensor containers"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        nested_input = [
            {"data": [torch.randn(2, 3).npu(), torch.randn(2, 3).npu()]},
            torch.randn(2, 3).npu(),
        ]
        result = tracer.create_arg(nested_input)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        dict_elem = result[0]
        self.assertIsInstance(dict_elem, dict)
        self.assertIn("data", dict_elem)

        list_elem = dict_elem["data"]
        self.assertIsInstance(list_elem, list)
        self.assertEqual(len(list_elem), 2)

        for node_elem in list_elem:
            self.assertIsInstance(node_elem, fx.Node)

        self.assertIsInstance(result[1], fx.Node)

    def test_create_arg_with_none(self):
        """Test that None is passed through without modification"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        result = tracer.create_arg(None)
        self.assertIsNone(result)

    def test_create_arg_with_basic_types(self):
        """Test that basic types (int, float, str, bool) are preserved"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        self.assertEqual(tracer.create_arg(42), 42)
        self.assertEqual(tracer.create_arg(3.14), 3.14)
        self.assertEqual(tracer.create_arg("hello"), "hello")
        self.assertTrue(tracer.create_arg(True))

    def test_create_arg_npu_tensor_consistency(self):
        """Test that NPU tensors are processed the same as CPU tensors"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        cpu_tensor = torch.randn(2, 3)
        npu_tensor = torch.randn(2, 3).npu()

        cpu_result = tracer.create_arg(cpu_tensor)
        npu_result = tracer.create_arg(npu_tensor)

        self.assertIsInstance(cpu_result, fx.Node)
        self.assertIsInstance(npu_result, fx.Node)
        self.assertEqual(cpu_result.op, npu_result.op)

    def test_create_arg_with_custom_tracer_override(self):
        """Test that custom Tracer can override create_arg for custom types"""

        class CustomType:
            def __init__(self, value):
                self.value = value

        class CustomTracer(fx.Tracer):
            def create_arg(self, a):
                if isinstance(a, CustomType):
                    return super().create_arg(a.value)
                return super().create_arg(a)

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModule()
        tracer = CustomTracer()
        tracer.trace(model)

        custom_input = CustomType(torch.randn(2, 3).npu())
        result = tracer.create_arg(custom_input)
        self.assertIsInstance(result, fx.Node)

    def test_create_arg_list_mixed_types(self):
        """Test mixed-type list processing (Tensor + basic types)"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        mixed_list = [
            torch.randn(2, 2).npu(),
            42,
            "hello",
            torch.randn(2, 2).npu()
        ]
        result = tracer.create_arg(mixed_list)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

        self.assertIsInstance(result[0], fx.Node)
        self.assertEqual(result[1], 42)
        self.assertEqual(result[2], "hello")
        self.assertIsInstance(result[3], fx.Node)

    def test_create_arg_dict_mixed_values(self):
        """Test mixed-type dict processing (Tensor + basic types + nested list)"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x

        model = SimpleModule()
        tracer = fx.Tracer()
        tracer.trace(model)

        mixed_dict = {
            "tensor": torch.randn(2, 2).npu(),
            "int": 42,
            "str": "hello",
            "list": [torch.randn(2, 2).npu(), torch.randn(2, 2).npu()],
        }
        result = tracer.create_arg(mixed_dict)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["tensor"], fx.Node)
        self.assertEqual(result["int"], 42)
        self.assertEqual(result["str"], "hello")
        self.assertIsInstance(result["list"], list)
        self.assertEqual(len(result["list"]), 2)
        self.assertIsInstance(result["list"][0], fx.Node)

    # ===================================================================
    # Tests for torch.fx.Tracer.create_args_for_root
    # ===================================================================

    def test_create_args_for_root_basic_module(self):
        """is_module=True with single-arg Module: args = [root, placeholder('x')]"""

        class SimpleModule(nn.Module):
            def forward(self, x):
                return x + 1

        mod = SimpleModule()
        tracer = fx.Tracer()
        tracer.root = mod
        tracer.graph = fx.Graph()
        fn, args = tracer.create_args_for_root(mod.forward, is_module=True)
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], mod)
        self.assertEqual(args[1].node.op, "placeholder")
        self.assertEqual(args[1].node.name, "x")

    def test_create_args_for_root_two_arg_module(self):
        """is_module=True with multi-arg Module: args = [root, ph('x'), ph('y')]"""
        mod = TwoArgModule()
        tracer = fx.Tracer()
        tracer.root = mod
        tracer.graph = fx.Graph()
        fn, args = tracer.create_args_for_root(mod.forward, is_module=True)
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], mod)
        self.assertEqual(args[1].node.name, "x")
        self.assertEqual(args[2].node.name, "y")

    def test_create_args_for_root_default_arg_module(self):
        """is_module=True with default-value param: placeholder for 'scale' created"""
        mod = DefaultArgModule()
        tracer = fx.Tracer()
        tracer.root = mod
        tracer.graph = fx.Graph()
        fn, args = tracer.create_args_for_root(mod.forward, is_module=True)
        self.assertEqual(len(args), 3)
        self.assertEqual(args[1].node.name, "x")
        self.assertEqual(args[2].node.name, "scale")

    def test_create_args_for_root_concrete_args_dict(self):
        """concrete_args as dict: specialise 'y' to a concrete tensor value"""
        mod = TwoArgModule()
        tracer = fx.Tracer()
        tracer.root = mod
        tracer.graph = fx.Graph()
        concrete = {"y": torch.tensor(3.0)}
        fn, args = tracer.create_args_for_root(
            mod.forward, is_module=True, concrete_args=concrete
        )
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], mod)

    def test_create_args_for_root_concrete_args_tuple(self):
        """concrete_args as tuple with PH: non-specialised params keep placeholder"""
        mod = TwoArgModule()
        tracer = fx.Tracer()
        tracer.root = mod
        tracer.graph = fx.Graph()
        concrete = (fx.PH, torch.tensor(5.0))
        fn, args = tracer.create_args_for_root(
            mod.forward, is_module=True, concrete_args=concrete
        )
        self.assertEqual(len(args), 3)

    def test_create_args_for_root_plain_function(self):
        """is_module=False: all params become placeholders, no self skip"""
        def my_fn(a, b):
            return a + b

        tracer = fx.Tracer()
        tracer.root = None
        tracer.graph = fx.Graph()
        fn, args = tracer.create_args_for_root(my_fn, is_module=False)
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0].node.name, "a")
        self.assertEqual(args[1].node.name, "b")

    def test_create_args_for_root_function_with_defaults(self):
        """is_module=False with default-value param: placeholder created for 'b'"""
        def my_fn(a, b=10):
            return a * b

        tracer = fx.Tracer()
        tracer.root = None
        tracer.graph = fx.Graph()
        fn, args = tracer.create_args_for_root(my_fn, is_module=False)
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0].node.name, "a")
        self.assertEqual(args[1].node.name, "b")


if __name__ == "__main__":
    run_tests()
