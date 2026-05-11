import contextlib
import io
import unittest

import torch
import torch_npu
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import TestCase, run_tests


class TestFxGraphApi(TestCase):

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_graph_placeholder_with_npu_tensor(self):
        graph = Graph()

        x = graph.placeholder("x")
        y = graph.placeholder("y")

        self.assertEqual(x.op, "placeholder")
        self.assertEqual(x.target, "x")
        self.assertEqual(y.op, "placeholder")
        self.assertEqual(y.target, "y")

        add_node = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
        graph.output(add_node)

        gm = GraphModule({}, graph)

        cpu_x = torch.randn(2, 3)
        cpu_y = torch.randn(2, 3)

        npu_x = cpu_x.npu()
        npu_y = cpu_y.npu()

        cpu_out = cpu_x + cpu_y
        npu_out = gm(npu_x, npu_y).cpu()

        self.assertTrue(torch.allclose(cpu_out, npu_out, rtol=1e-3, atol=1e-3))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_graph_output_node_with_npu_tensor(self):
        graph = Graph()

        x = graph.placeholder("x")
        neg_node = graph.call_function(torch.ops.aten.neg.default, args=(x,))
        graph.output(neg_node)

        output_node = graph.output_node()

        self.assertIsNotNone(output_node)
        self.assertEqual(output_node.op, "output")
        self.assertEqual(output_node.args[0], neg_node)

        gm = GraphModule({}, graph)

        cpu_x = torch.randn(4, 4)
        npu_x = cpu_x.npu()

        cpu_out = torch.neg(cpu_x)
        npu_out = gm(npu_x).cpu()

        self.assertTrue(torch.allclose(cpu_out, npu_out, rtol=1e-3, atol=1e-3))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_graph_print_tabular_with_npu_meta(self):
        try:
            import tabulate  # noqa: F401
        except ImportError:
            self.skipTest("tabulate is not installed")

        graph = Graph()

        x = graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3).npu()

        relu_node = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        graph.output(relu_node)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            graph.print_tabular()

        output = buffer.getvalue()

        self.assertIn("placeholder", output)
        self.assertIn("call_function", output)
        self.assertIn("output", output)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_graph_process_inputs_with_npu_tensor(self):
        graph = Graph()

        cpu_x = torch.randn(2, 3)
        cpu_y = torch.randn(2, 3)

        npu_x = cpu_x.npu()
        npu_y = cpu_y.npu()

        processed_inputs = graph.process_inputs(npu_x, npu_y)

        self.assertEqual(len(processed_inputs), 2)
        self.assertTrue(processed_inputs[0].is_npu)
        self.assertTrue(processed_inputs[1].is_npu)
        self.assertTrue(torch.allclose(processed_inputs[0].cpu(), cpu_x, rtol=1e-3, atol=1e-3))
        self.assertTrue(torch.allclose(processed_inputs[1].cpu(), cpu_y, rtol=1e-3, atol=1e-3))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_graph_process_outputs_with_npu_tensor(self):
        graph = Graph()

        cpu_out = torch.randn(2, 3)
        npu_out = cpu_out.npu()

        processed_output = graph.process_outputs(npu_out)

        self.assertTrue(processed_output.is_npu)
        self.assertTrue(torch.allclose(processed_output.cpu(), cpu_out, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    run_tests()
