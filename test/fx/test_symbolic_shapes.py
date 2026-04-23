import unittest

import torch
import torch_npu
from torch.fx import Graph
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv,
    is_accessor_node,
    is_concrete_bool,
    is_concrete_float,
    is_concrete_int,
    is_symbolic,
)
from torch.testing._internal.common_utils import TestCase, run_tests


class TestSymbolicShapes(TestCase):
    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_is_accessor_node_with_call_method(self):
        graph = Graph()
        x = graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3).npu()

        size_node = graph.call_method("size", args=(x, 0))
        self.assertTrue(is_accessor_node(size_node))

    def test_is_accessor_node_with_call_function(self):
        graph = Graph()
        x = graph.placeholder("x")

        size_node = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        add_node = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))

        self.assertTrue(is_accessor_node(size_node))
        self.assertFalse(is_accessor_node(add_node))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_is_concrete_int_with_literal_and_npu_shape(self):
        x = torch.randn(2, 3).npu()
        sym_int = ShapeEnv().create_unbacked_symint()

        self.assertTrue(is_concrete_int(3))
        self.assertTrue(is_concrete_int(x.size(0)))
        self.assertFalse(is_concrete_int(sym_int))
        self.assertFalse(is_symbolic(x.size(0)))
        self.assertTrue(is_symbolic(sym_int))

    def test_is_concrete_float_with_literal_and_symbolic_value(self):
        sym_float = ShapeEnv().create_unbacked_symfloat()

        self.assertTrue(is_concrete_float(1.5))
        self.assertFalse(is_concrete_float(sym_float))
        self.assertFalse(is_symbolic(1.5))
        self.assertTrue(is_symbolic(sym_float))

    def test_is_concrete_bool_with_literal_and_symbolic_value(self):
        sym_bool = ShapeEnv().create_unbacked_symbool()

        self.assertTrue(is_concrete_bool(True))
        self.assertFalse(is_concrete_bool(sym_bool))
        self.assertFalse(is_symbolic(True))
        self.assertTrue(is_symbolic(sym_bool))


if __name__ == "__main__":
    run_tests()
