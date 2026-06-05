"""
Add validation cases for torch.fx Graph APIs on NPU:

1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.fx.Graph.inserting_after,
   torch.fx.Graph.inserting_before, torch.fx.graph.magic_methods.format,
   torch.fx.graph.inplace_methods.format,
   torch.fx.Graph.find_nodes, torch.fx.Graph.graph_copy,
   torch.fx.Graph.erase_node, and torch.fx.Graph.get_attr.
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

    # Test find_nodes: query nodes by op type, target, and sort parameter.
    def test_find_nodes(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.tensor([1.0]))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return torch.relu(x) + self.linear(x) * self.p

        m = M().to(device_type)
        gm = symbolic_trace(m)
        # call_function + target
        relu_nodes = gm.graph.find_nodes(op="call_function", target=torch.relu)
        self.assertEqual(len(relu_nodes), 1)
        self.assertEqual(relu_nodes[0].op, "call_function")
        # call_module
        module_nodes = gm.graph.find_nodes(op="call_module")
        self.assertEqual(len(module_nodes), 1)
        self.assertEqual(module_nodes[0].target, "linear")
        # get_attr
        attr_nodes = gm.graph.find_nodes(op="get_attr")
        self.assertEqual(len(attr_nodes), 1)
        self.assertEqual(attr_nodes[0].target, "p")
        # sort requires target
        sorted_relu = gm.graph.find_nodes(op="call_function", target=torch.relu, sort=True)
        unsorted_relu = gm.graph.find_nodes(op="call_function", target=torch.relu, sort=False)
        self.assertEqual(set(sorted_relu), set(unsorted_relu))
        # empty result
        self.assertEqual(gm.graph.find_nodes(op="call_function", target=torch.sigmoid), [])

        input_tensor = torch.randn(2, 3).to(device_type)
        self.assertEqual(gm(input_tensor), m(input_tensor))

    # Test graph_copy: copy nodes between graphs with val_map and return_output_node.
    def test_graph_copy(self):
        g_src = torch.fx.Graph()
        x = g_src.placeholder("x")
        neg = g_src.call_function(torch.neg, (x,))
        relu = g_src.call_function(torch.relu, (neg,))
        g_src.output((relu,))

        # Basic copy with val_map populated.
        g_dst = torch.fx.Graph()
        val_map = {}
        g_dst.graph_copy(g_src, val_map)
        self.assertEqual(len(val_map), 3)
        x_dst, neg_dst, relu_dst = val_map[x], val_map[neg], val_map[relu]
        self.assertEqual(neg_dst.args, (x_dst,))
        self.assertEqual(relu_dst.args, (neg_dst,))

        # return_output_node returns the source output node.
        g_dst2 = torch.fx.Graph()
        rv = g_dst2.graph_copy(g_src, {}, return_output_node=True)
        self.assertIsNotNone(rv)
        _, src_output = rv
        self.assertEqual(src_output.op, "output")

        # val_map reuse: existing node replaces source node.
        g_dst3 = torch.fx.Graph()
        existing = g_dst3.placeholder("existing")
        val_map3 = {x: existing}
        g_dst3.graph_copy(g_src, val_map3)
        self.assertEqual(val_map3[neg].args[0], existing)

    # Test erase_node: remove a node and rewire its users.
    def test_erase_node(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(torch.neg, (x,))
        relu = graph.call_function(torch.relu, (neg,))
        graph.output(relu)

        neg.replace_all_uses_with(x)
        graph.erase_node(neg)

        self.assertNotIn(neg, list(graph.nodes))
        self.assertEqual(relu.args, (x,))
        graph.lint()

        gm = GraphModule(torch.nn.Module(), graph)
        input_tensor = torch.randn(2, 3).to(device_type)
        self.assertEqual(gm(input_tensor), torch.relu(input_tensor))

    # Test get_attr: retrieve module parameters as graph nodes via symbolic tracing.
    def test_get_attr(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0]))
                self.bias = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return x * self.weight + self.bias

        gm = symbolic_trace(M())
        attr_nodes = gm.graph.find_nodes(op="get_attr")
        self.assertEqual(len(attr_nodes), 2)
        targets = {n.target for n in attr_nodes}
        self.assertIn("weight", targets)
        self.assertIn("bias", targets)

        gm = gm.to(device_type)
        input_tensor = torch.randn(2, 3).to(device_type)
        self.assertEqual(gm(input_tensor), M().to(device_type)(input_tensor))


if __name__ == "__main__":
    run_tests()
