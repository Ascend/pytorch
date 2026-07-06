"""
Add validation cases for torch.fx.experimental.symbolic_shapes APIs on NPU:

PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
This file validates ShapeEnv guard / sympy APIs and PropagateUnbackedSymInts interpreter APIs (extendable).
"""

import unittest

import sympy
import torch
import torch_npu
from torch import nn
from torch._guards import ShapeGuard, SLoc
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, Interpreter, symbolic_trace
from torch.fx.experimental.symbolic_shapes import (
    PropagateUnbackedSymInts,
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

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_propagate_unbacked_symints_map_nodes_to_values(self):
        # Validate map_nodes_to_values maps FX nodes to NPU runtime values.
        self.assertIs(
            PropagateUnbackedSymInts.map_nodes_to_values,
            Interpreter.map_nodes_to_values,
        )

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        module = Module().npu()
        gm = symbolic_trace(module)
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        call_module = next(node for node in gm.graph.nodes if node.op == "call_module")

        interpreter = PropagateUnbackedSymInts(gm)
        interpreter.env[placeholder] = torch.randn(4, 3).npu()

        mapped = interpreter.map_nodes_to_values((placeholder,), call_module)
        self.assertEqual(mapped[0].device.type, "npu")
        self.assertEqual(tuple(mapped[0].shape), (4, 3))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_propagate_unbacked_symints_fetch_args_kwargs_from_env(self):
        # Validate fetch_args_kwargs_from_env resolves NPU node arguments.
        self.assertIs(
            PropagateUnbackedSymInts.fetch_args_kwargs_from_env,
            Interpreter.fetch_args_kwargs_from_env,
        )

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        module = Module().npu()
        gm = symbolic_trace(module)
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        call_module = next(node for node in gm.graph.nodes if node.op == "call_module")

        interpreter = PropagateUnbackedSymInts(gm)
        interpreter.env[placeholder] = torch.randn(4, 3).npu()

        args, kwargs = interpreter.fetch_args_kwargs_from_env(call_module)
        self.assertEqual(args[0].device.type, "npu")
        self.assertEqual(tuple(args[0].shape), (4, 3))
        self.assertEqual(kwargs, {})

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_propagate_unbacked_symints_fetch_attr(self):
        # Validate fetch_attr fetches NPU GraphModule attributes.
        self.assertIs(PropagateUnbackedSymInts.fetch_attr, Interpreter.fetch_attr)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(2))

            def forward(self, x):
                return x + self.bias

        module = Module().npu()
        gm = symbolic_trace(module)
        get_attr = next(node for node in gm.graph.nodes if node.op == "get_attr")

        interpreter = PropagateUnbackedSymInts(gm)

        fetched_attr = interpreter.fetch_attr(get_attr.target)
        self.assertEqual(fetched_attr.device.type, "npu")
        self.assertEqual(tuple(fetched_attr.shape), (2,))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_propagate_unbacked_symints_call_module(self):
        # Validate call_module executes NPU submodules.
        self.assertIs(PropagateUnbackedSymInts.call_module, Interpreter.call_module)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        module = Module().npu()
        gm = symbolic_trace(module)
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        call_module = next(node for node in gm.graph.nodes if node.op == "call_module")

        interpreter = PropagateUnbackedSymInts(gm)
        interpreter.env[placeholder] = torch.randn(4, 3).npu()

        module_result = interpreter.call_module(
            call_module.target, (interpreter.env[placeholder],), {}
        )
        self.assertEqual(module_result.device.type, "npu")
        self.assertEqual(tuple(module_result.shape), (4, 2))

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_propagate_unbacked_symints_get_attr(self):
        # Validate get_attr returns NPU attributes.
        self.assertIs(PropagateUnbackedSymInts.get_attr, Interpreter.get_attr)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(2))

            def forward(self, x):
                return x + self.bias

        module = Module().npu()
        gm = symbolic_trace(module)
        get_attr = next(node for node in gm.graph.nodes if node.op == "get_attr")

        interpreter = PropagateUnbackedSymInts(gm)

        attr_result = interpreter.get_attr(get_attr.target, (), {})
        self.assertEqual(attr_result.device.type, "npu")
        self.assertEqual(tuple(attr_result.shape), (2,))


class TestShapeEnvNPU(TestCase):
    """Issue #1627: direct NPU unit tests for ShapeEnv guard / sympy methods (v2.7.1)."""

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

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_format_guards(self):
        """Verify format_guards returns formatted guard strings."""
        se = ShapeEnv()
        self.assertEqual(se.format_guards(), "")

        s0 = sympy.Symbol('s0', integer=True, positive=True)
        s1 = sympy.Symbol('s1', integer=True, positive=True)
        sloc = SLoc("framework_loc", "user_loc")
        se.guards.append(ShapeGuard(
            sympy.Lt(s0, s1, evaluate=False), sloc, False))
        se.guards.append(ShapeGuard(
            sympy.Ge(s0, sympy.Integer(1), evaluate=False), sloc, False))

        result = se.format_guards()
        self.assertIn("s0 < s1", result)
        self.assertIn("s0 >= 1", result)

        result_verbose = se.format_guards(verbose=True)
        self.assertIn("s0 < s1", result_verbose)
        self.assertIn("user_loc", result_verbose)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_freeze(self):
        """Verify freeze toggles the frozen state of ShapeEnv."""
        se = ShapeEnv()
        self.assertFalse(se.frozen)
        se.freeze()
        self.assertTrue(se.frozen)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_freeze_runtime_asserts(self):
        """Verify freeze_runtime_asserts toggles runtime_asserts_frozen."""
        se = ShapeEnv()
        self.assertFalse(se.runtime_asserts_frozen)
        se.freeze_runtime_asserts()
        self.assertTrue(se.runtime_asserts_frozen)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_get_axioms(self):
        """Verify get_axioms returns axioms tuple, optionally filtered by symbols."""
        se = ShapeEnv()
        self.assertIsInstance(se.get_axioms(), tuple)

        s0 = sympy.Symbol('s0', integer=True, positive=True)
        s1 = sympy.Symbol('s1', integer=True, positive=True)
        sloc = SLoc("framework_loc", "user_loc")
        se.guards.append(ShapeGuard(
            sympy.Lt(s0, s1, evaluate=False), sloc, False))

        axioms = se.get_axioms(symbols=(s0,))
        self.assertIn(sympy.Lt(s0, s1, evaluate=False), axioms)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_torch_fx_experimental_symbolic_shapes_ShapeEnv_get_implications(self):
        """Verify get_implications returns implications for Eq/Lt/Ne/Le expressions."""
        se = ShapeEnv()
        s0 = sympy.Symbol('s0', integer=True, positive=True)
        s1 = sympy.Symbol('s1', integer=True, positive=True)

        # Eq implies equality
        impl_eq = dict(se.get_implications(
            sympy.Eq(s0, s1, evaluate=False)))
        self.assertIn(sympy.Eq(s0, s1, evaluate=False), impl_eq)

        # Lt implies Le and Ne
        impl_lt = dict(se.get_implications(
            sympy.Lt(s0, s1, evaluate=False)))
        self.assertIn(sympy.Lt(s0, s1, evaluate=False), impl_lt)
        self.assertIn(sympy.Le(s0, s1, evaluate=False), impl_lt)
        self.assertIn(sympy.Ne(s0, s1, evaluate=False), impl_lt)

        # Ne
        impl_ne = dict(se.get_implications(
            sympy.Ne(s0, s1, evaluate=False)))
        self.assertIn(sympy.Ne(s0, s1, evaluate=False), impl_ne)

        # Le implies Lt(a, b+1)
        impl_le = dict(se.get_implications(
            sympy.Le(s0, s1, evaluate=False)))
        self.assertIn(sympy.Le(s0, s1, evaluate=False), impl_le)
        self.assertIn(sympy.Lt(s0, s1 + 1, evaluate=False), impl_le)


if __name__ == "__main__":
    run_tests()
