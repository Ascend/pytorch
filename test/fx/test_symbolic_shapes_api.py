# Owner(s): ["module: fx"]
"""
Add validation cases for torch.fx symbolic_shapes related APIs on NPU:

1. This file groups symbolic_shapes compatibility and constraint-behavior tests
   in one place, so the same-category APIs can continue to be extended here.
2. This file validates the core functionality of symbolic_shapes related APIs
   on NPU environment.
3. Current covered APIs / behaviors include:
   - symbolic_shapes.definitely_true
   - symbolic_shapes.definitely_false
   - symbolic_shapes.DimConstraints.add
   - symbolic_shapes.DimConstraints.add_equality
"""

import sympy

import torch_npu
from torch._dynamo.source import ConstantSource
from torch.fx.experimental import symbolic_shapes
from torch.testing._internal.common_utils import TestCase, run_tests


class TestSymbolicShapesAPI(TestCase):
    def test_definitely_true_compatibility_behavior(self):
        # Verify that the compatibility patch exposes definitely_true.
        self.assertTrue(hasattr(symbolic_shapes, "definitely_true"))
        self.assertTrue(symbolic_shapes.definitely_true(True))
        self.assertFalse(symbolic_shapes.definitely_true(False))

    def test_definitely_false_compatibility_behavior(self):
        # Verify that the compatibility patch exposes definitely_false.
        self.assertTrue(hasattr(symbolic_shapes, "definitely_false"))
        self.assertFalse(symbolic_shapes.definitely_false(True))
        self.assertTrue(symbolic_shapes.definitely_false(False))

    def test_dim_constraints_add_true_returns_trivial(self):
        # Verify that a trivial `true` constraint is accepted without
        # introducing any tracked inequality state.
        symbol = sympy.Symbol("s0", integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {symbol: [ConstantSource("x")]},
            {symbol: sympy.Integer(4)},
            set(),
            {},
        )

        self.assertTrue(constraints.add(sympy.true))
        self.assertEqual(dict(constraints._univariate_inequalities), {})

    def test_dim_constraints_add_tracks_univariate_and_multivariate_constraints(self):
        # Verify that `add` records both single-symbol and multi-symbol
        # constraints in the expected internal collections.
        s0 = sympy.Symbol("s0", integer=True)
        s1 = sympy.Symbol("s1", integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {s0: [ConstantSource("x")], s1: [ConstantSource("y")]},
            {s0: sympy.Integer(4), s1: sympy.Integer(5)},
            set(),
            {},
        )

        # A single-symbol equality should be tracked as a univariate constraint.
        self.assertFalse(constraints.add(sympy.Eq(s0, 4)))
        self.assertIn(s0, constraints._symbols_with_equalities)
        self.assertEqual(constraints._univariate_inequalities[s0], {sympy.Eq(s0, 4)})

        # A mixed-symbol expression should be tracked as a multivariate constraint.
        multivariate_expr = s0 + s1 > 1
        self.assertFalse(constraints.add(multivariate_expr))
        self.assertEqual(constraints._multivariate_inequalities, {multivariate_expr})

    def test_dim_constraints_add_equality_records_static_and_symbolic_results(self):
        # Verify that `add_equality` distinguishes static equalities from
        # symbolic expressions and stores both results correctly.
        symbol = sympy.Symbol("s0", integer=True)
        source = ConstantSource("z")
        constraints = symbolic_shapes.DimConstraints(
            {symbol: [source]},
            {symbol: sympy.Integer(4)},
            set(),
            {},
        )

        constraints.add_equality(source, sympy.Integer(6))
        self.assertEqual(constraints._static_results, {"z == 6"})

        symbolic_expr = symbol + 1
        constraints.add_equality(source, symbolic_expr)
        self.assertEqual(constraints._symbolic_equivalences, [(source, symbolic_expr)])


if __name__ == "__main__":
    run_tests()
