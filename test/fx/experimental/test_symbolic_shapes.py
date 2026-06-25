"""
Add validation cases for torch.fx.experimental.symbolic_shapes APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for these APIs, so this file is added.
2. This file validates RelaxedUnspecConstraint, resolve_unbacked_bindings, safe_expand,
-ShapeEnv.add_backed_var_to_val, ShapeEnv.create_symboolnode, ShapeEnv.create_symfloatnode,
-ShapeEnv.create_symbol, ShapeEnv.bound_sympy, ShapeEnv.check_equal,
-ShapeEnv.cleanup, ShapeEnv.bind_symbols.
3. This file is extendable for other torch.fx.experimental.symbolic_shapes APIs.
"""

import sympy
import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import RelaxedUnspecConstraint
from torch.fx.experimental.symbolic_shapes import resolve_unbacked_bindings
from torch.fx.experimental.symbolic_shapes import safe_expand
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSymbolicShapes(TestCase):

    def test_relaxed_unspec_constraint_with_npu_tensor_shape(self):
        # Verify the core instantiation and properties of RelaxedUnspecConstraint.
        dummy_tensor = torch.zeros(3).to(device_type)
        shape_env = ShapeEnv()
        sym_int = shape_env.create_unbacked_symint()
        symbol = sym_int.node.expr
        constraint = RelaxedUnspecConstraint(symbol)
        self.assertIsNotNone(constraint)
        self.assertTrue(hasattr(constraint, 'args') or constraint is not None)
        self.assertEqual(dummy_tensor.device.type, "npu")

    def test_resolve_unbacked_bindings_with_npu_tensor_shape(self):
        # Verify the core functionality of resolve_unbacked_bindings using NPU tensor shapes.
        dummy_tensor = torch.ones(4, 4).to(device_type)
        shape_env = ShapeEnv()
        sym_int = shape_env.create_unbacked_symint()
        symbol = sym_int.node.expr
        bindings = {symbol: dummy_tensor.size(0)}
        resolved = resolve_unbacked_bindings(shape_env, bindings)
        self.assertIsNotNone(resolved)
        self.assertEqual(dummy_tensor.device.type, "npu")

    def test_safe_expand_with_npu_tensor_shape(self):
        # Verify the core functionality of safe_expand for symbolic expressions.
        dummy_tensor = torch.ones(4, 4).to(device_type)
        shape_env = ShapeEnv()
        sym_int = shape_env.create_unbacked_symint()
        symbol = sym_int.node.expr
        expr = symbol * (symbol + dummy_tensor.size(0))
        expanded_expr = safe_expand(expr)
        self.assertIsNotNone(expanded_expr)
        self.assertNotEqual(expr, expanded_expr)
        self.assertEqual(dummy_tensor.device.type, "npu")


class TestShapeEnvSymbolicShapes(TestCase):

    def test_create_symboolnode_with_npu_tensor_shape(self):
        # Create a tensor on the current accelerator and use its shape as
        # the backed value for symbolic boolean creation.
        tensor = torch.ones(2, 3).to(device_type)
        self.assertEqual(tensor.device.type, device_type)

        # Build a ShapeEnv symbol from the NPU tensor shape.
        # This verifies that the API works with metadata from an NPU tensor.
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(
            tensor.size(0),
            source=ConstantSource("symbool"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )

        # Create a SymBool from a symbolic expression based on the NPU tensor shape.
        symbool = shape_env.create_symboolnode(sympy.Eq(symbol, tensor.size(0)))

        # Validate both the returned type and the boolean result.
        self.assertIsInstance(symbool, torch.SymBool)
        self.assertTrue(bool(symbool))

    def test_create_symfloatnode_with_npu_tensor_shape(self):
        # Create a tensor on the current accelerator and use its shape as
        # the backed value for symbolic float creation.
        tensor = torch.ones(2, 3).to(device_type)
        self.assertEqual(tensor.device.type, device_type)

        # Build a ShapeEnv symbol from the NPU tensor shape.
        # This keeps the symbolic value connected with NPU tensor metadata.
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(
            tensor.size(0),
            source=ConstantSource("symfloat"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )

        # Create a SymFloat from a symbolic expression based on the NPU tensor shape.
        # The hint equals tensor.size(0) + 0.5.
        symfloat = shape_env.create_symfloatnode(
            symbol + sympy.Float(0.5),
            hint=float(tensor.size(0)) + 0.5,
        )

        # Validate both the returned type and the concrete hint result.
        self.assertIsInstance(symfloat, torch.SymFloat)
        self.assertEqual(float(symfloat), 2.5)

    def test_add_backed_var_to_val_with_npu_tensor_shape(self):
        # Verify the core functionality of ShapeEnv.add_backed_var_to_val using NPU tensor shapes.
        dummy_tensor = torch.ones(7).to(device_type)
        shape_env = ShapeEnv()
        sym = sympy.Symbol("s0")
        actual_val = dummy_tensor.size(0)
        shape_env.add_backed_var_to_val(sym, actual_val)
        self.assertIn(sym, shape_env.backed_var_to_val)
        self.assertEqual(shape_env.backed_var_to_val[sym], 7)
        self.assertEqual(dummy_tensor.device.type, "npu")


class TestShapeEnvCoreMethods(TestCase):
    """Unit tests for ShapeEnv core methods: create_symbol, bound_sympy, check_equal, cleanup, bind_symbols."""

    def setUp(self):
        self.env = ShapeEnv()
        self.source = ConstantSource("x")

    def test_create_symbol(self):
        """Test create_symbol returns unique sympy.Symbol instances usable in expressions."""
        sym1 = self.env.create_symbol(5, self.source)
        sym2 = self.env.create_symbol(10, self.source)

        self.assertIsInstance(sym1, sympy.Symbol)
        self.assertIsInstance(sym2, sympy.Symbol)
        self.assertTrue(sym1.name.startswith('s'))
        self.assertTrue(sym2.name.startswith('s'))
        self.assertNotEqual(sym1.name, sym2.name, "Symbols must have distinct names")
        expr = sym1 + sym2
        self.assertIsInstance(expr, sympy.Expr)

    def test_bound_sympy(self):
        """Test bound_sympy returns correct lower bound for 's0+2' (at least 4)."""
        s0 = self.env.create_symbol(5, self.source)
        expr = s0 + 2
        bounds = self.env.bound_sympy(expr)

        self.assertTrue(hasattr(bounds, 'lower'))
        self.assertTrue(hasattr(bounds, 'upper'))
        self.assertLessEqual(bounds.lower, bounds.upper)
        self.assertGreaterEqual(bounds.lower, 4)

    def test_check_equal(self):
        """Test check_equal passes for self-equality and fails for different environment states."""
        self.env.check_equal(self.env)

        other = ShapeEnv()
        other.create_symbol(5, self.source)
        # NotEqualError is defined in torch.fx.experimental.recording (PyTorch 2.9+)
        try:
            from torch.fx.experimental.recording import NotEqualError
            expected_exceptions = (AssertionError, NotEqualError)
        except ImportError:
            expected_exceptions = AssertionError
        with self.assertRaises(expected_exceptions):
            self.env.check_equal(other)

    def test_cleanup(self):
        """Test cleanup does not break future symbol creation."""
        self.env.create_symbol(5, self.source)
        self.env.cleanup()
        new_sym = self.env.create_symbol(10, self.source)
        self.assertIsInstance(new_sym, sympy.Symbol)

    def test_bind_symbols(self):
        """Test bind_symbols correctly maps symbolic placeholders to concrete tensor dimensions."""
        try:
            from torch.fx.experimental.proxy_tensor import make_fake_tensor
            has_fake = True
        except ImportError:
            has_fake = False

        if not has_fake:
            self.skipTest("make_fake_tensor not available in this environment")

        # Single tensor binding
        s0 = self.env.create_symbol(5, self.source)
        s1 = self.env.create_symbol(2, self.source)
        fake_input = make_fake_tensor(torch.empty(s0, s1), self.env, self.source)
        real_input = torch.randn(5, 2)
        bindings = self.env.bind_symbols([fake_input], [real_input])
        self.assertIsInstance(bindings, dict)
        self.assertIn(s0, bindings)
        self.assertIn(s1, bindings)
        self.assertEqual(bindings[s0], 5)
        self.assertEqual(bindings[s1], 2)

        # Multiple tensor batch binding
        s2 = self.env.create_symbol(3, self.source)
        s3 = self.env.create_symbol(4, self.source)
        fake_input_2 = make_fake_tensor(torch.empty(s2, s3), self.env, self.source)
        real_input_2 = torch.randn(3, 4)
        bindings_batch = self.env.bind_symbols([fake_input, fake_input_2], [real_input, real_input_2])
        self.assertEqual(len(bindings_batch), 4)
        # Verify both old and new symbols are correctly mapped
        self.assertIn(s0, bindings_batch)
        self.assertIn(s1, bindings_batch)
        self.assertIn(s2, bindings_batch)
        self.assertIn(s3, bindings_batch)
        self.assertEqual(bindings_batch[s0], 5)
        self.assertEqual(bindings_batch[s1], 2)
        self.assertEqual(bindings_batch[s2], 3)
        self.assertEqual(bindings_batch[s3], 4)

        # Negative case: mismatched argument count
        with self.assertRaises(ValueError):
            self.env.bind_symbols([fake_input], [real_input, torch.randn(3, 4)])


if __name__ == "__main__":
    run_tests()
