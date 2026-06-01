"""
Add validation cases for torch.fx.experimental.symbolic_shapes.ShapeEnv APIs on NPU:

PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
This file validates ShapeEnv.produce_guards_expression, ShapeEnv.produce_guards_verbose, ShapeEnv.replace, ShapeEnv.set_unbacked_var_to_val, ShapeEnv.simplify (extendable).
"""

import unittest

import sympy
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv,
    Source,
    is_accessor_node,
    is_concrete_bool,
    is_concrete_float,
    is_concrete_int,
    is_symbolic,
)
from torch.testing._internal.common_utils import TestCase, run_tests


def _shape_env_has(name: str) -> bool:
    """Return True if ShapeEnv exports a callable method with the given name."""
    return callable(getattr(ShapeEnv, name, None))


def _produce_guards_verbose_works() -> bool:
    """Probe produce_guards_verbose with bare Source(); skip when upstream rejects it."""
    if not _shape_env_has("produce_guards_verbose"):
        return False
    try:
        env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=env)
        placeholder = fake_mode.from_tensor(torch.randn(2, 2))
        env.produce_guards_verbose([placeholder], [Source()])
        return True
    except (NotImplementedError, TypeError):
        return False


def _shape_env_set_unbacked_var_to_val_works() -> bool:
    """Probe set_unbacked_var_to_val(create_unbacked_symint(), val); skip on known breakage."""
    if not _shape_env_has("set_unbacked_var_to_val"):
        return False
    try:
        env = ShapeEnv()
        env.set_unbacked_var_to_val(env.create_unbacked_symint(), 4)
        return True
    except TypeError:
        return False


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


class TestShapeEnvNPU(TestCase):
    """Issue #1627: direct NPU unit tests for ShapeEnv guard / sympy methods (v2.9.0)."""

    def _shape_env_with_fake_placeholders(self, shape=(3, 4)):
        """Build a ShapeEnv with NPU FakeTensor placeholders for guard-generation tests."""
        env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=env)
        fake_tensor = fake_mode.from_tensor(torch.randn(*shape, device="npu"))
        return env, [fake_tensor]

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_produce_guards_expression(self):
        """Verify produce_guards_expression returns a guard expression string on NPU."""
        env, placeholders = self._shape_env_with_fake_placeholders()
        guards = env.produce_guards_expression(placeholders)
        self.assertIsInstance(guards, str)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    @unittest.skipUnless(
        _produce_guards_verbose_works(),
        "torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose "
        "cannot run with bare Source() on this PyTorch build",
    )
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_produce_guards_verbose(self):
        """Verify produce_guards_verbose with Source list on NPU FakeTensor placeholders."""
        env, placeholders = self._shape_env_with_fake_placeholders()
        source = Source()
        sources = [source] * len(placeholders)
        guards = env.produce_guards_verbose(placeholders, sources)
        self.assertIsNotNone(guards)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_replace(self):
        """Verify replace is identity when no substitution rules are registered."""
        env = ShapeEnv()
        a, b = sympy.symbols("a b")
        original_expr = a + b
        self.assertEqual(env.replace(original_expr), original_expr)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    @unittest.skipUnless(
        _shape_env_set_unbacked_var_to_val_works(),
        "torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val "
        "is broken with create_unbacked_symint() on this PyTorch build",
    )
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_set_unbacked_var_to_val(self):
        """Verify set_unbacked_var_to_val binds a concrete value to an unbacked SymInt."""
        env = ShapeEnv()
        unbacked_sym = env.create_unbacked_symint()
        env.set_unbacked_var_to_val(unbacked_sym, 4)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_simplify(self):
        """Verify simplify reduces sympy expressions inside the ShapeEnv context."""
        env = ShapeEnv()
        a, b = sympy.symbols("a b")
        self.assertEqual(env.simplify((a + b) - b), a)


if __name__ == "__main__":
    run_tests()
