# Owner(s): ["module: fx"]
"""
Add validation cases for torch.fx symbolic_shapes related APIs on NPU:

1. This file groups symbolic_shapes compatibility and constraint-behavior tests
   in one place, so the same-category APIs can continue to be extended here.
2. This file validates the core functionality of symbolic_shapes related APIs
   on NPU environment.
3. Current covered APIs / behaviors include:
   - symbolic_shapes.DimConstraints.add
   - symbolic_shapes.DimConstraints.add_equality
   - symbolic_shapes.DimConstraints.rewrite_with_congruences
   - symbolic_shapes.DimConstraints.solve
   - symbolic_shapes.DimConstraints.forced_specializations
   - symbolic_shapes.DimConstraints.prettify_results
   - torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint
   - torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards
   - torch.fx.experimental.symbolic_shapes.ShapeEnvSettings
   - torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext
   - torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext
   - symbolic_shapes._lru_cache
   - symbolic_shapes.CallMethodKey
   - symbolic_shapes.CallMethodKey.get
   - symbolic_shapes.canonicalize_bool_expr
   - symbolic_shapes.check_consistent
   - symbolic_shapes.StrictMinMaxConstraint
   - symbolic_shapes.StrictMinMaxConstraint.render
   - symbolic_shapes.SubclassSymbolicContext
   - symbolic_shapes.sym_eq
"""

import dataclasses
import inspect

import sympy
import torch

import torch_npu
from torch._dynamo.source import ConstantSource
from torch.export import Dim
from torch.fx.experimental import symbolic_shapes
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._sympy.functions import FloorDiv, Mod
from torch.utils._sympy.value_ranges import ValueRanges


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSymbolicShapesAPI(TestCase):
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

    def test_dim_constraints_rewrite_with_congruences_records_mod_guard(self):
        # Verify that congruence guards rewrite floor division and record
        # the corresponding modular relationship for the symbol.
        symbol = sympy.Symbol("s0", positive=True, integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {},
            {symbol: sympy.Integer(5)},
            set(),
            {},
        )

        rewritten = constraints.rewrite_with_congruences(symbol, FloorDiv(symbol, 2))

        self.assertEqual(rewritten, symbol / 2 - sympy.Rational(1, 2))
        self.assertEqual(
            {str(expr) for expr in constraints._congruences[symbol]},
            {"Mod(s0 + 1, 2)"},
        )

    def test_dim_constraints_solve_records_dynamic_results(self):
        # Verify that `solve` classifies a satisfiable lower-bound guard
        # as a dynamic result rather than a static specialization.
        symbol = sympy.Symbol("s0", positive=True, integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {symbol: [ConstantSource("x")]},
            {symbol: sympy.Integer(4)},
            {symbol},
            {},
        )

        constraints.add(symbol >= 2)
        constraints.solve()

        self.assertEqual(constraints._static_results, set())
        self.assertEqual(constraints._dynamic_results, {"2 <= x"})

    def test_dim_constraints_forced_specializations_reports_marked_dynamic_equalities(self):
        # Verify that marked dynamic equalities are reported as forced
        # specializations using the configured debug name.
        symbol = sympy.Symbol("s0", positive=True, integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {symbol: [ConstantSource("x")]},
            {symbol: sympy.Integer(4)},
            {symbol},
            {"x": "dx"},
        )

        constraints.add(sympy.Eq(symbol, 4))
        constraints.solve()

        self.assertEqual(constraints.forced_specializations(), {"dx = x": 4})

    def test_dim_constraints_prettify_results_reports_forced_specialization(self):
        # Verify that `prettify_results` explains a forced specialization
        # and includes the suggested concrete dimension value.
        def fn(x):
            return x

        symbol = sympy.Symbol("s0", positive=True, integer=True)
        constraints = symbolic_shapes.DimConstraints(
            {symbol: [ConstantSource("x")]},
            {symbol: sympy.Integer(4)},
            {symbol},
            {"x": "dx"},
        )

        constraints.add(sympy.Eq(symbol, 4))
        constraints.solve()
        message = constraints.prettify_results(
            inspect.signature(fn),
            {"x": Dim("dx")},
            ValueError("dummy constraint violation"),
            constraints.forced_specializations(),
        )

        self.assertIn("Specializations unexpectedly required (dx)!", message)
        self.assertIn("dx = x", message)
        self.assertIn("dx = 4", message)

    def test_strict_min_max_constraint_records_warn_only_and_value_range(self):
        # Verify that StrictMinMaxConstraint stores warn_only and ValueRanges
        # according to its actual constructor signature.
        constraint = symbolic_shapes.StrictMinMaxConstraint(
            warn_only=False,
            vr=ValueRanges(2, 10),
        )

        self.assertFalse(constraint.warn_only)
        self.assertEqual(constraint.vr.lower, 2)
        self.assertEqual(constraint.vr.upper, 10)

    def test_strict_min_max_constraint_render(self):
        # Verify that render converts the min/max range constraint into
        # a readable constraint expression for the given source.
        constraint = symbolic_shapes.StrictMinMaxConstraint(
            warn_only=False,
            vr=ValueRanges(2, 10),
        )

        self.assertEqual(constraint.render(ConstantSource("x")), "2 <= x <= 10")

    def test_subclass_symbolic_context_records_dynamic_sizes_and_tensor_source(self):
        # Verify that SubclassSymbolicContext can be constructed with the
        # required dynamic_sizes, tensor_source, and inner_contexts arguments.
        source = ConstantSource("x")
        context = symbolic_shapes.SubclassSymbolicContext(
            dynamic_sizes=[],
            tensor_source=source,
            inner_contexts={},
        )

        self.assertEqual(context.dynamic_sizes, [])
        self.assertIs(context.tensor_source, source)
        self.assertEqual(context.inner_contexts, {})

    def test_sym_eq_for_python_values(self):
        # Verify that sym_eq returns the expected equality result for
        # ordinary Python values.
        self.assertTrue(symbolic_shapes.sym_eq(1, 1))
        self.assertFalse(symbolic_shapes.sym_eq(1, 2))


    def test_shape_env_size_hint(self):
        shape_env = symbolic_shapes.ShapeEnv()
        self.assertEqual(shape_env.size_hint(sympy.Integer(8)), 8)

        signature = inspect.signature(shape_env.size_hint)
        self.assertIn("expr", signature.parameters)
        self.assertIn("allow_none", signature.parameters)
        self.assertEqual(signature.parameters["allow_none"].default, False)

    def test_shape_env_suppress_guards(self):
        shape_env = symbolic_shapes.ShapeEnv()
        with shape_env.suppress_guards():
            self.assertEqual(shape_env.size_hint(sympy.Integer(4)), 4)

    def test_shape_env_settings(self):
        field_names = {
            field.name for field in dataclasses.fields(symbolic_shapes.ShapeEnvSettings)
        }
        setting_values = {
            "allow_scalar_outputs": True,
            "allow_dynamic_output_shape_ops": True,
            "assume_static_by_default": False,
            "specialize_zero_one": True,
            "duck_shape": True,
            "prefer_deferred_runtime_asserts_over_guards": False,
            "allow_complex_guards_as_runtime_asserts": False,
            "trace_asserts": False,
        }

        settings = symbolic_shapes.ShapeEnvSettings(
            **{
                name: value
                for name, value in setting_values.items()
                if name in field_names
            }
        )

        self.assertIn("allow_scalar_outputs", field_names)
        self.assertIn("duck_shape", field_names)
        for name, value in setting_values.items():
            if name in field_names:
                self.assertEqual(getattr(settings, name), value)

    def test_stateless_symbolic_context(self):
        context = symbolic_shapes.StatelessSymbolicContext(
            dynamic_sizes=[symbolic_shapes.DimDynamic.DUCK, symbolic_shapes.DimDynamic.DUCK],
        )

        self.assertEqual(
            context.dynamic_sizes,
            [symbolic_shapes.DimDynamic.DUCK, symbolic_shapes.DimDynamic.DUCK],
        )
        self.assertEqual(
            context.dynamic_strides,
            [symbolic_shapes.DimDynamic.INFER_STRIDE, symbolic_shapes.DimDynamic.INFER_STRIDE],
        )
        self.assertEqual(context.constraint_sizes, [None, None])
        self.assertEqual(context.constraint_strides, [None, None])

    def test_stateful_symbolic_context(self):
        tensor_source = ConstantSource("x")
        context = symbolic_shapes.StatefulSymbolicContext(
            dynamic_sizes=[symbolic_shapes.DimDynamic.DUCK, symbolic_shapes.DimDynamic.DUCK],
            tensor_source=tensor_source,
        )

        self.assertEqual(context.tensor_source, tensor_source)
        self.assertEqual(context.shape_env_to_source_to_symbol_cache, {})
        self.assertEqual(
            context.dynamic_sizes,
            [symbolic_shapes.DimDynamic.DUCK, symbolic_shapes.DimDynamic.DUCK],
        )



class TestSymbolicShapesTargetApiNPU(TestCase):
    def test_lru_cache_handles_hits_clears_and_maxsize(self):
        class DummyShapeEnv:
            def __init__(self):
                self._version_counter = 0

            def _get_key(self):
                return ("dummy", self._version_counter)

        calls = {"count": 0}

        @symbolic_shapes._lru_cache
        def cached(self, value):
            calls["count"] += 1
            return self._version_counter, value, calls["count"]

        env = DummyShapeEnv()
        first = cached(env, 7)
        self.assertEqual(first, (0, 7, 1))
        self.assertEqual(cached.cache_info().misses, 1)

        # Same ShapeEnv version and same arguments should hit the cache.
        self.assertEqual(cached(env, 7), first)
        self.assertEqual(calls["count"], 1)
        cache_info = cached.cache_info()
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.currsize, 1)

        # A ShapeEnv version update must invalidate old cached values.
        env._version_counter += 1
        self.assertEqual(cached(env, 7), (1, 7, 2))
        self.assertEqual(calls["count"], 2)

        # cache_clear is exposed by the wrapper and removes current entries.
        cached.cache_clear()
        self.assertEqual(cached.cache_info().currsize, 0)
        self.assertEqual(cached(env, 7), (1, 7, 3))
        self.assertEqual(calls["count"], 3)

        limited_calls = {"count": 0}

        def limited_cached(self, value):
            limited_calls["count"] += 1
            return value, limited_calls["count"]

        limited = symbolic_shapes._lru_cache(limited_cached, maxsize=1)
        limited(env, 1)
        limited(env, 2)
        limited(env, 1)

        # maxsize=1 should evict the older argument entry.
        self.assertEqual(limited.cache_info().maxsize, 1)
        self.assertEqual(limited_calls["count"], 3)

    def test_call_method_key_get_on_npu_tensor(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(device_type)
        size_key = symbolic_shapes.CallMethodKey("size")
        same_size_key = symbolic_shapes.CallMethodKey("size")
        stride_key = symbolic_shapes.CallMethodKey("stride")
        storage_offset_key = symbolic_shapes.CallMethodKey("storage_offset")
        dim_key = symbolic_shapes.CallMethodKey("dim")

        # CallMethodKey is a frozen dataclass with a stable path string.
        self.assertEqual(size_key, same_size_key)
        self.assertEqual(hash(size_key), hash(same_size_key))
        self.assertEqual(str(size_key), ".size()")

        # Validate zero-argument Tensor method calls on the current accelerator.
        self.assertEqual(tuple(size_key.get(tensor)), (3, 4))
        self.assertEqual(tuple(stride_key.get(tensor)), (4, 1))
        self.assertEqual(storage_offset_key.get(tensor), 0)
        self.assertEqual(dim_key.get(tensor), 2)

    def test_canonicalize_bool_expr(self):
        a, b = sympy.symbols("a b", integer=True)

        # Ge and Gt are rewritten into Le and Lt with non-constant terms on rhs.
        self.assertEqual(
            symbolic_shapes.canonicalize_bool_expr(sympy.Ge(a + 3, b)),
            sympy.Le(b, a + 3),
        )
        self.assertEqual(
            symbolic_shapes.canonicalize_bool_expr(sympy.Gt(2 * a + 4, 2 * b)),
            sympy.Lt(b, a + 2),
        )

        # Eq and Ne keep their relation type while reducing integer factors.
        self.assertEqual(
            symbolic_shapes.canonicalize_bool_expr(sympy.Eq(2 * a + 4, 2 * b + 2)),
            sympy.Eq(a + 1, b),
        )
        self.assertEqual(
            symbolic_shapes.canonicalize_bool_expr(sympy.Ne(2 * a, 2 * b)),
            sympy.Ne(a, b),
        )

        canonical_and = symbolic_shapes.canonicalize_bool_expr(
            sympy.And(sympy.Ge(a + 3, b), sympy.Ne(a, b))
        )
        self.assertIsInstance(canonical_and, sympy.And)
        self.assertIn(sympy.Le(b, a + 3), canonical_and.args)
        self.assertIn(sympy.Ne(a, b), canonical_and.args)

        canonical_or = symbolic_shapes.canonicalize_bool_expr(
            sympy.Or(sympy.Gt(a, b), sympy.Eq(2 * a + 4, 2 * b + 2))
        )
        self.assertIsInstance(canonical_or, sympy.Or)
        self.assertIn(sympy.Lt(b, a), canonical_or.args)
        self.assertIn(sympy.Eq(a + 1, b), canonical_or.args)

        canonical_not = symbolic_shapes.canonicalize_bool_expr(
            sympy.Not(sympy.And(sympy.Ge(a, b), sympy.Ne(a, b)))
        )
        self.assertIsInstance(canonical_not, sympy.Or)
        self.assertIn(sympy.Lt(a, b), canonical_not.args)
        self.assertIn(sympy.Eq(a, b), canonical_not.args)

        plain_expr = sympy.Integer(5)
        self.assertEqual(symbolic_shapes.canonicalize_bool_expr(plain_expr), plain_expr)

    def test_check_consistent_on_npu_tensor_and_scalars(self):
        old_tensor = torch.zeros(2, 3, device=device_type)
        new_tensor = torch.ones(2, 3, device=device_type)

        # Equal tensor metadata and equal scalar values should pass.
        symbolic_shapes.check_consistent(new_tensor, old_tensor)
        symbolic_shapes.check_consistent(4, 4)
        symbolic_shapes.check_consistent(2.5, 2.5)

        with self.assertRaisesRegex(RuntimeError, "old != new"):
            symbolic_shapes.check_consistent(
                torch.zeros(2, 4, device=device_type), old_tensor
            )
        with self.assertRaisesRegex(RuntimeError, "old != new"):
            symbolic_shapes.check_consistent(
                torch.zeros(6, device=device_type), old_tensor
            )
        with self.assertRaisesRegex(RuntimeError, "old != new"):
            symbolic_shapes.check_consistent(4, 5)
        with self.assertRaisesRegex(RuntimeError, "old != new"):
            symbolic_shapes.check_consistent(2.5, 3.5)

        # Bool and unknown types are intentionally skipped by check_consistent.
        symbolic_shapes.check_consistent(True, False)
        symbolic_shapes.check_consistent({"shape": (2,)}, {"shape": (3,)})


if __name__ == "__main__":
    run_tests()
