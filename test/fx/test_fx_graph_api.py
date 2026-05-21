"""
Add validation cases for torch.fx Graph APIs on NPU:

1. This file adds lightweight direct validations for torch.fx Graph APIs on NPU.
2. This file validates torch.fx.Graph.inserting_after,
   torch.fx.Graph.inserting_before, torch.fx.graph.magic_methods.format,
   and torch.fx.graph.inplace_methods.format.
"""

import operator

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.fx import GraphModule, symbolic_trace

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFxGraphApiNPU(TestCase):
    def test_graph_inserting_after(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(torch.neg, (x,))
        with graph.inserting_after(neg):
            relu = graph.call_function(torch.relu, (neg,))
        graph.output(relu)
        graph.lint()

        nodes = list(graph.nodes)
        # Validate both insertion order and data dependency.
        self.assertEqual(nodes.index(relu), nodes.index(neg) + 1)
        self.assertEqual(neg.args, (x,))
        self.assertEqual(relu.args, (neg,))
        self.assertEqual(relu.target, torch.relu)

        gm = GraphModule(torch.nn.Module(), graph)
        input_tensor = torch.randn(2, 3).to(device_type)
        self.assertEqual(gm(input_tensor), torch.relu(torch.neg(input_tensor)))

    def test_graph_inserting_before(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, (x,))
        graph.output(relu)
        with graph.inserting_before(relu):
            neg = graph.call_function(torch.neg, (x,))
            relu.args = (neg,)
        graph.lint()

        nodes = list(graph.nodes)
        # Validate both insertion order and rewritten input dependency.
        self.assertLess(nodes.index(neg), nodes.index(relu))
        self.assertEqual(nodes.index(neg), nodes.index(relu) - 1)
        self.assertEqual(neg.args, (x,))
        self.assertEqual(relu.args, (neg,))
        self.assertEqual(neg.target, torch.neg)

        gm = GraphModule(torch.nn.Module(), graph)
        input_tensor = torch.randn(2, 3).to(device_type)
        self.assertEqual(gm(input_tensor), torch.relu(torch.neg(input_tensor)))

    def test_magic_methods_format_codegen(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x << 3, x >> 3

        input_tensor = torch.LongTensor(10).random_(0, 1024).to(device_type)
        gm = symbolic_trace(MyModule())
        gm.graph.lint()

        # Symbolic tracing should create proxy nodes for shift operators.
        nodes = list(gm.graph.nodes)
        x_node = next(node for node in nodes if node.op == "placeholder")
        lshift_node = next(node for node in nodes if node.target == operator.lshift)
        rshift_node = next(node for node in nodes if node.target == operator.rshift)
        self.assertEqual(lshift_node.args, (x_node, 3))
        self.assertEqual(rshift_node.args, (x_node, 3))

        self.assertIn("x << 3", gm.code)
        self.assertIn("x >> 3", gm.code)

        expected = MyModule()(input_tensor)
        self.assertEqual(gm(input_tensor), expected)

    def test_inplace_methods_format_codegen(self):
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        # Build imul directly because tracing "a *= b" lowers to operator.mul.
        imul = graph.call_function(operator.imul, (a, b), {})
        graph.output(a)
        graph.lint()

        self.assertEqual(imul.args, (a, b))
        self.assertEqual(imul.target, operator.imul)

        gm = GraphModule(torch.nn.Module(), graph)
        gm.recompile()
        self.assertIn("a *= b", gm.code)

        input_tensor = torch.ones(2, 3).to(device_type)
        scale = torch.full((2, 3), 3.0).to(device_type)
        output = gm(input_tensor, scale)
        self.assertEqual(output, torch.full((2, 3), 3.0).to(device_type))
        self.assertEqual(input_tensor, torch.full((2, 3), 3.0).to(device_type))


if __name__ == "__main__":
    run_tests()
