"""
Add validation cases for torch.fx.experimental.symbolic_shapes.ShapeEnv APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for these ShapeEnv
   symbolic expression methods, so this file is added.
2. This file validates ShapeEnv.deserialize_symexpr, ShapeEnv.evaluate_symexpr,
   ShapeEnv.evaluate_guards_expression, ShapeEnv.evaluate_guards_for_args,
   ShapeEnv.evaluate_sym_node (extendable).
"""

import torch
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestShapeEnvAPIs(TestCase):

    def test_deserialize_symexpr_constant(self):
        env = ShapeEnv()
        self.assertEqual(int(env.deserialize_symexpr("5")), 5)

    def test_deserialize_symexpr_expression(self):
        env = ShapeEnv()
        self.assertEqual(int(env.deserialize_symexpr("2 + 3")), 5)

    def test_deserialize_symexpr_with_symbol(self):
        env = ShapeEnv()
        source = TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0)
        s = env.create_symbol(5, source=source, dynamic_dim=DimDynamic.DYNAMIC)
        self.assertEqual(str(env.deserialize_symexpr(str(s))), str(s))

    def test_evaluate_symexpr_constant(self):
        env = ShapeEnv()
        self.assertEqual(env.evaluate_symexpr("10"), 10)

    def test_evaluate_symexpr_addition(self):
        env = ShapeEnv()
        self.assertEqual(env.evaluate_symexpr("3 + 4"), 7)

    def test_evaluate_symexpr_multiplication(self):
        env = ShapeEnv()
        self.assertEqual(env.evaluate_symexpr("6 * 7"), 42)

    def test_evaluate_guards_expression_true(self):
        env = ShapeEnv()
        self.assertTrue(env.evaluate_guards_expression("True", []))

    def test_evaluate_guards_expression_false(self):
        env = ShapeEnv()
        self.assertFalse(env.evaluate_guards_expression("False", []))

    def test_evaluate_guards_expression_returns_bool(self):
        env = ShapeEnv()
        self.assertIsInstance(env.evaluate_guards_expression("True", []), bool)

    def test_evaluate_guards_for_args_basic(self):
        env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=env, allow_non_fake_inputs=True)
        with fake_mode:
            placeholder = torch.empty(3, 4)
        real_tensor = torch.randn(3, 4).to(device_type)
        result = env.evaluate_guards_for_args([placeholder], [real_tensor])
        self.assertIsInstance(result, bool)

    def test_evaluate_guards_for_args_multi_placeholders(self):
        env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=env, allow_non_fake_inputs=True)
        with fake_mode:
            p0 = torch.empty(3, 4)
            p1 = torch.empty(5, 6)
        real0 = torch.randn(3, 4).to(device_type)
        real1 = torch.randn(5, 6).to(device_type)
        result = env.evaluate_guards_for_args([p0, p1], [real0, real1])
        self.assertIsInstance(result, bool)

    def test_evaluate_sym_node_basic(self):
        env = ShapeEnv()
        source = TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0)
        s = env.create_symbol(5, source=source, dynamic_dim=DimDynamic.DYNAMIC)
        sym = env.create_symintnode(s, hint=5, source=source)
        self.assertEqual(env.evaluate_sym_node(sym.node, size_oblivious=False), 5)

    def test_evaluate_sym_node_size_oblivious(self):
        env = ShapeEnv()
        source = TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0)
        s = env.create_symbol(5, source=source, dynamic_dim=DimDynamic.DYNAMIC)
        sym = env.create_symintnode(s, hint=5, source=source)
        self.assertEqual(env.evaluate_sym_node(sym.node, size_oblivious=True), 5)

    def test_evaluate_sym_node_different_hint(self):
        env = ShapeEnv()
        source = TensorPropertySource(LocalSource("y"), TensorProperty.SIZE, 0)
        s = env.create_symbol(7, source=source, dynamic_dim=DimDynamic.DYNAMIC)
        sym = env.create_symintnode(s, hint=7, source=source)
        self.assertEqual(env.evaluate_sym_node(sym.node, size_oblivious=False), 7)


if __name__ == "__main__":
    run_tests()
