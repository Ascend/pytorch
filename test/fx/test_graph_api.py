# Owner(s): ["module: fx"]

import unittest

import torch
import torch_npu
from torch.fx import Graph, GraphModule
from torch.fx.graph import map_arg
from torch.testing._internal.common_utils import TestCase, run_tests


class TestGraphAPI(TestCase):
    def _build_add_graph(self):
        graph = Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
        output = graph.output(add)
        return graph, x, y, add, output

    def test_map_arg_preserves_nested_structure(self):
        _, x, y, _, _ = self._build_add_graph()

        mapped = map_arg(
            ((x,), {"rhs": [y], "tag": "graph"}),
            lambda node: node.name,
        )

        self.assertEqual(mapped[0], ("x",))
        self.assertEqual(mapped[1]["rhs"], ["y"])
        self.assertEqual(mapped[1]["tag"], "graph")

    def test_node_copy_rebuilds_graph(self):
        graph, _, _, _, _ = self._build_add_graph()
        copied = Graph()
        value_map = {}

        for node in graph.nodes:
            if node.op == "placeholder":
                value_map[node] = copied.placeholder(node.target)
            else:
                value_map[node] = copied.node_copy(node, lambda n: value_map[n])

        self.assertEqual(
            [node.op for node in copied.nodes],
            ["placeholder", "placeholder", "call_function", "output"],
        )

    def test_nodes_iterates_in_topological_order(self):
        graph, _, _, add, output = self._build_add_graph()

        self.assertEqual(
            [node.op for node in graph.nodes],
            ["placeholder", "placeholder", "call_function", "output"],
        )
        self.assertEqual(list(graph.nodes)[-2:], [add, output])

    def test_on_generate_code_updates_graph_module_code(self):
        graph, _, _, _, _ = self._build_add_graph()
        gm = GraphModule(torch.nn.Module(), graph)

        def transform_code(code):
            return ['print("fx-hook")\n', *code]

        gm.graph.on_generate_code(lambda _: transform_code)
        gm.recompile()

        self.assertIn('print("fx-hook")', gm.code)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_output_runs_with_npu_inputs(self):
        graph, _, _, _, output = self._build_add_graph()
        gm = GraphModule(torch.nn.Module(), graph)
        x = torch.randn(2, 2).npu()
        y = torch.randn(2, 2).npu()

        result = gm(x, y).cpu()

        self.assertEqual(output.op, "output")
        self.assertEqual(result.shape, torch.Size([2, 2]))
        self.assertEqual(result, (x + y).cpu())


if __name__ == "__main__":
    run_tests()
