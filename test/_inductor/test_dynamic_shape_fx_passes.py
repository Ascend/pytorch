# Owner(s): ["module: inductor"]
"""Device-independent unit tests for dynamic-shape graph optimization.

These cases validate the symbolic-shape utilities and passes purely at the FX
level (build graph + run pass + assert transform); they run no NPU/CUDA kernel and
thus work on CPU-only machines. Coverage focuses on:
1. symbolic_shape_util: three-valued checks / normalization / materialization / ranges;
2. per-pass "optimize only when provable" behavior on dynamic shapes;
3. boundary safety: undecidable, rank mismatch, distinct symbols, switch-off, etc.
   must not mis-fold;
4. static-shape regression: existing optimizations still fire on static inputs.
"""

import collections.abc
import importlib
import logging
import os
import sys
import types
import unittest


def _shim_missing_torch_internals():
    """Shim an internal util present in the target torch (2.10) but missing on older local torch (e.g. 2.2).

    Injected only when genuinely absent; the real target environment is untouched."""
    try:
        importlib.import_module("torch.utils._ordered_set")
        return
    except Exception:
        pass

    module = types.ModuleType("torch.utils._ordered_set")

    class OrderedSet(collections.abc.MutableSet):
        def __init__(self, iterable=()):
            self._data = dict.fromkeys(iterable)

        def __contains__(self, value):
            return value in self._data

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def add(self, value):
            self._data[value] = None

        def discard(self, value):
            self._data.pop(value, None)

        def pop(self):
            key = next(iter(self._data))
            del self._data[key]
            return key

    module.OrderedSet = OrderedSet
    sys.modules["torch.utils._ordered_set"] = module


_NPU_TEST_LIB = None


def _shim_missing_npu_ops():
    """Register a schema stub for the npu custom op referenced at import time when the torch_npu C++ ext is absent."""
    global _NPU_TEST_LIB
    try:
        torch.ops.npu._npu_dtype_cast.default
        return
    except Exception:
        pass
    _NPU_TEST_LIB = torch.library.Library("npu", "FRAGMENT")
    try:
        _NPU_TEST_LIB.define("_npu_dtype_cast(Tensor self, ScalarType dtype) -> Tensor")
    except Exception:
        pass


def _ensure_torch_npu_importable():
    """Stub the torch_npu package root on NPU-less machines so the pure graph-pass modules import.

    On a real NPU box ``import torch_npu`` works and we return early; otherwise we
    only stub the necessary package nodes and ``torch_npu._inductor.config.log``,
    while other submodules still load from the real source files.
    """
    try:
        import torch_npu  # noqa: F401

        return
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _pkg(name, rel):
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(repo_root, *rel)]
        sys.modules[name] = module

    _pkg("torch_npu", ["torch_npu"])
    _pkg("torch_npu._inductor", ["torch_npu", "_inductor"])
    config_stub = types.ModuleType("torch_npu._inductor.config")
    config_stub.log = logging.getLogger("dynamic_fx_test")
    sys.modules["torch_npu._inductor.config"] = config_stub
    _pkg("torch_npu._inductor.fx_passes", ["torch_npu", "_inductor", "fx_passes"])
    _pkg(
        "torch_npu._inductor.fx_passes.utils",
        ["torch_npu", "_inductor", "fx_passes", "utils"],
    )
    _pkg(
        "torch_npu._inductor.fx_passes.ascend_custom_passes",
        ["torch_npu", "_inductor", "fx_passes", "ascend_custom_passes"],
    )


import torch

_shim_missing_torch_internals()
_shim_missing_npu_ops()
_ensure_torch_npu_importable()

from torch.fx.experimental.proxy_tensor import make_fx

from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass as agp
from torch_npu._inductor.fx_passes.utils import symbolic_shape_util as ssu


def _symbolic_gm(fn, *example_inputs):
    """Trace an aten-level fx graph in symbolic mode; nodes carry symbolic FakeTensor meta."""
    return make_fx(fn, tracing_mode="symbolic")(*example_inputs)


def _first_symint(gm):
    """Grab one symbolic dim (SymInt) from the graph's first placeholder, for util tests."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if val is not None and hasattr(val, "shape"):
                for dim in val.shape:
                    if isinstance(dim, torch.SymInt):
                        return dim
    raise AssertionError("no symbolic dim found in traced graph")


def _count_target(gm, target):
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is target
    )


class TestSymbolicShapeUtil(unittest.TestCase):
    """Three-valued logic, normalization, materialization and ranges of symbolic_shape_util."""

    def setUp(self):
        os.environ.pop("NPU_INDUCTOR_DYNAMIC_FX_PASS", None)
        ssu.reset_stats()
        # Build two independent symbols s0, s1 plus derived expressions
        self.gm = _symbolic_gm(lambda a, b: a + b.sum(), torch.randn(6, 5), torch.randn(7))
        self.s0 = None
        self.s1 = None
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if val is None or not hasattr(val, "shape") or not val.shape:
                    continue
                dim0 = val.shape[0]
                if isinstance(dim0, torch.SymInt):
                    if self.s0 is None:
                        self.s0 = dim0
                    elif self.s1 is None:
                        self.s1 = dim0
        self.assertIsNotNone(self.s0)
        self.assertIsNotNone(self.s1)

    # ---- three-valued checks -----------------------------------------
    def test_statically_known_eq_int(self):
        self.assertTrue(ssu.statically_known_eq(3, 3))
        self.assertFalse(ssu.statically_known_eq(3, 4))

    def test_statically_known_eq_same_symbol(self):
        self.assertTrue(ssu.statically_known_eq(self.s0, self.s0))
        self.assertTrue(ssu.statically_known_eq(self.s0 + 1, self.s0 + 1))

    def test_statically_known_eq_distinct_symbols_unprovable(self):
        # s0 == s1 is undecidable -> False (neither mis-proven true nor adds a guard)
        self.assertFalse(ssu.statically_known_eq(self.s0, self.s1))

    def test_statically_known_eq_symbol_vs_const_unprovable(self):
        self.assertFalse(ssu.statically_known_eq(self.s0, 5))

    def test_statically_known_geq_gt_leq(self):
        self.assertTrue(ssu.statically_known_geq(self.s0 + 1, self.s0))
        self.assertTrue(ssu.statically_known_gt(self.s0 + 1, self.s0))
        self.assertTrue(ssu.statically_known_leq(self.s0, self.s0 + 1))
        self.assertFalse(ssu.statically_known_gt(self.s0, self.s0 + 1))

    def test_is_statically_one(self):
        self.assertTrue(ssu.is_statically_one(1))
        self.assertFalse(ssu.is_statically_one(2))
        self.assertFalse(ssu.is_statically_one(self.s0))

    def test_shapes_statically_equal(self):
        self.assertTrue(ssu.shapes_statically_equal([self.s0, 3], [self.s0, 3]))
        self.assertFalse(ssu.shapes_statically_equal([self.s0, 3], [self.s1, 3]))
        self.assertFalse(ssu.shapes_statically_equal([self.s0], [self.s0, 1]))
        self.assertFalse(ssu.shapes_statically_equal(None, [1]))

    def test_has_free_symbols(self):
        self.assertTrue(ssu.has_free_symbols([self.s0, 2]))
        self.assertFalse(ssu.has_free_symbols([2, 3]))
        self.assertFalse(ssu.has_free_symbols(None))

    # ---- normalization -----------------------------------------------
    def test_resolve_size_arg_scalar(self):
        self.assertEqual(ssu.resolve_size_arg(4), 4)
        self.assertIs(ssu.resolve_size_arg(self.s0), self.s0)
        self.assertIsNone(ssu.resolve_size_arg(True))
        self.assertIsNone(ssu.resolve_size_arg(1.5))

    def test_resolve_size_arg_node(self):
        # sym_size node: meta['val'] is a SymInt
        sym_size_node = None
        for n in self.gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.meta.get("val"), torch.SymInt):
                sym_size_node = n
                break
        if sym_size_node is not None:
            self.assertIsInstance(ssu.resolve_size_arg(sym_size_node), torch.SymInt)

    def test_resolve_size_list(self):
        self.assertEqual(ssu.resolve_size_list([1, 2, 3]), [1, 2, 3])
        self.assertIsNone(ssu.resolve_size_list([1, object()]))
        self.assertIsNone(ssu.resolve_size_list(5))

    # ---- value ranges ------------------------------------------------
    def test_statically_fits_int32_int(self):
        self.assertTrue(ssu.statically_fits_int32(0, 100, -100))
        self.assertFalse(ssu.statically_fits_int32(2**31))
        self.assertFalse(ssu.statically_fits_int32())

    def test_statically_fits_int32_unbounded_symbol(self):
        # An unbounded symbol cannot be proven to fit int32
        self.assertFalse(ssu.statically_fits_int32(self.s0))

    # ---- materialization ---------------------------------------------
    def test_materialize_shape_from_anchor(self):
        gm = _symbolic_gm(lambda x: x + 1, torch.randn(6, 4))
        ph = [n for n in gm.graph.nodes if n.op == "placeholder"][0]
        s0 = ph.meta["val"].shape[0]
        anchor_user = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor
        ][0]
        with gm.graph.inserting_before(anchor_user):
            mat = ssu.materialize_shape(gm.graph, [s0, 8], ph)
        self.assertIsNotNone(mat)
        self.assertEqual(mat[1], 8)
        self.assertIsInstance(mat[0], torch.fx.Node)
        self.assertIs(mat[0].target, torch.ops.aten.sym_size.int)

    def test_materialize_shape_unresolvable_returns_none(self):
        gm = _symbolic_gm(lambda x: x + 1, torch.randn(6, 4))
        ph = [n for n in gm.graph.nodes if n.op == "placeholder"][0]
        s0 = ph.meta["val"].shape[0]
        anchor_user = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor
        ][0]
        # s0*s0 is not any anchor dim length -> cannot materialize
        with gm.graph.inserting_before(anchor_user):
            mat = ssu.materialize_shape(gm.graph, [s0 * s0], ph)
        self.assertIsNone(mat)

    # ---- switch -------------------------------------------------------
    def test_switch_off_degrades_to_static(self):
        os.environ["NPU_INDUCTOR_DYNAMIC_FX_PASS"] = "0"
        try:
            # When off, symbolic comparisons are all undecidable, but pure ints still decide
            self.assertFalse(ssu.statically_known_eq(self.s0, self.s0))
            self.assertTrue(ssu.statically_known_eq(3, 3))
            self.assertIsNone(ssu.resolve_size_arg(self.s0))
            self.assertEqual(ssu.resolve_size_arg(4), 4)
        finally:
            os.environ.pop("NPU_INDUCTOR_DYNAMIC_FX_PASS", None)

    def test_materialize_shape_switch_off_symbolic_returns_none(self):
        # Switch off is a hard kill-switch: symbolic dims cannot be materialized,
        # but a fully-static shape still passes through unchanged.
        gm = _symbolic_gm(lambda x: x + 1, torch.randn(6, 4))
        ph = [n for n in gm.graph.nodes if n.op == "placeholder"][0]
        s0 = ph.meta["val"].shape[0]
        anchor_user = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor
        ][0]
        os.environ["NPU_INDUCTOR_DYNAMIC_FX_PASS"] = "0"
        try:
            with gm.graph.inserting_before(anchor_user):
                self.assertIsNone(ssu.materialize_shape(gm.graph, [s0, 8], ph))
                self.assertEqual(
                    ssu.materialize_shape(gm.graph, [2, 8], ph), [2, 8]
                )
        finally:
            os.environ.pop("NPU_INDUCTOR_DYNAMIC_FX_PASS", None)


class TestDynamicShapePasses(unittest.TestCase):
    """Per-pass behavior and boundary safety on dynamic shapes."""

    def setUp(self):
        os.environ.pop("NPU_INDUCTOR_DYNAMIC_FX_PASS", None)

    def test_fold_expand_identity_symbolic(self):
        def fn(x):
            return torch.ops.aten.expand.default(x, [x.size(0), x.size(1)]).relu()

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        before = _count_target(gm, torch.ops.aten.expand.default)
        self.assertGreaterEqual(before, 1)
        agp.fold_expand(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.expand.default), 0)

    def test_view_fold_identity_symbolic(self):
        def fn(x):
            return torch.ops.aten.view.default(x, [x.size(0), x.size(1)]).relu()

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.view_fold_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.view.default), 0)

    def test_fold_reduce_static_one_dim(self):
        # Middle dim is statically 1 (0/1 specialization); symbolic batch dim is kept
        def fn(x):
            return torch.ops.aten.sum.dim_IntList(x, [1], True)

        gm = _symbolic_gm(fn, torch.randn(6, 1, 4))
        before = _count_target(gm, torch.ops.aten.sum.dim_IntList)
        self.assertEqual(before, 1)
        agp.fold_reduce(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.sum.dim_IntList), 0)

    def test_fold_reduce_symbolic_dim_not_folded(self):
        # Reducing a symbolic dim that cannot be proven 1 -> must be kept
        def fn(x):
            return torch.ops.aten.sum.dim_IntList(x, [0], True)

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.fold_reduce(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.sum.dim_IntList), 1)

    def test_fold_slice_full_symbolic(self):
        def fn(x):
            return torch.ops.aten.slice.Tensor(x, 0, 0, x.size(0)).relu()

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.fold_slice(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.slice.Tensor), 0)

    def test_fold_slice_partial_symbolic_not_folded(self):
        # Slicing to s0-1 cannot be proven to cover the full dim -> kept
        def fn(x):
            return torch.ops.aten.slice.Tensor(x, 0, 0, x.size(0) - 1).relu()

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.fold_slice(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.slice.Tensor), 1)

    def test_repeat_to_expand_symbolic(self):
        # x:[s0,1] repeat(1,3) -> pure broadcast, can become expand (mul consumer is broadcast-friendly)
        def fn(x):
            r = torch.ops.aten.repeat.default(x, [1, 3])
            return torch.ops.aten.mul.Tensor(r, r)

        gm = _symbolic_gm(fn, torch.randn(6, 1))
        agp.repeat_to_expand_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.repeat.default), 0)
        self.assertGreaterEqual(_count_target(gm, torch.ops.aten.expand.default), 1)

    def test_repeat_physical_copy_symbolic_not_rewritten(self):
        # x:[s0,4] repeat(1,3): 2nd dim is neither 1 nor kept -> needs a physical copy, keep repeat
        def fn(x):
            r = torch.ops.aten.repeat.default(x, [1, 3])
            return torch.ops.aten.mul.Tensor(r, r)

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.repeat_to_expand_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.repeat.default), 1)

    def test_fold_four_op_add_zeros_symbolic(self):
        def fn(x):
            return x + torch.zeros_like(x)

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        before = _count_target(gm, torch.ops.aten.add.Tensor)
        self.assertGreaterEqual(before, 1)
        agp.fold_four_op_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.add.Tensor), 0)

    def test_cat_to_view_identity_symbolic(self):
        # cat([x[:, 0:2], x[:, 2:s1]], dim=1) covers all of dim1 -> identity view
        def fn(x):
            a = torch.ops.aten.slice.Tensor(x, 1, 0, 2)
            b = torch.ops.aten.slice.Tensor(x, 1, 2, x.size(1))
            return torch.ops.aten.cat.default([a, b], 1)

        gm = _symbolic_gm(fn, torch.randn(6, 5))
        self.assertEqual(_count_target(gm, torch.ops.aten.cat.default), 1)
        agp.cat_to_view_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.cat.default), 0)

    def test_cat_to_view_partial_not_folded(self):
        # Slices do not cover the whole dim (missing tail) -> must not fold
        def fn(x):
            a = torch.ops.aten.slice.Tensor(x, 1, 0, 2)
            b = torch.ops.aten.slice.Tensor(x, 1, 2, 4)
            return torch.ops.aten.cat.default([a, b], 1)

        gm = _symbolic_gm(fn, torch.randn(6, 5))
        agp.cat_to_view_pass(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.cat.default), 1)


class TestSwitchAndStaticRegression(unittest.TestCase):
    """Switch-off degrades to static behavior; existing optimizations still fire on static shapes."""

    def tearDown(self):
        os.environ.pop("NPU_INDUCTOR_DYNAMIC_FX_PASS", None)

    def test_switch_off_symbolic_slice_not_folded(self):
        os.environ["NPU_INDUCTOR_DYNAMIC_FX_PASS"] = "0"

        def fn(x):
            return torch.ops.aten.slice.Tensor(x, 0, 0, x.size(0)).relu()

        gm = _symbolic_gm(fn, torch.randn(6, 4))
        agp.fold_slice(gm.graph)
        # Switch off: symbolic full-slice is undecidable -> kept
        self.assertEqual(_count_target(gm, torch.ops.aten.slice.Tensor), 1)

    def test_switch_off_symbolic_repeat_not_rewritten(self):
        os.environ["NPU_INDUCTOR_DYNAMIC_FX_PASS"] = "0"

        def fn(x):
            r = torch.ops.aten.repeat.default(x, [1, 3])
            return torch.ops.aten.mul.Tensor(r, r)

        gm = _symbolic_gm(fn, torch.randn(6, 1))
        agp.repeat_to_expand_pass(gm.graph)
        # Switch off: symbolic broadcast is not rewritten (materialize refuses sym dims)
        self.assertEqual(_count_target(gm, torch.ops.aten.repeat.default), 1)

    def test_static_expand_still_folded(self):
        def fn(x):
            return torch.ops.aten.expand.default(x, [4, 5]).relu()

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(4, 5))
        agp.fold_expand(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.expand.default), 0)

    def test_static_slice_still_folded(self):
        def fn(x):
            return torch.ops.aten.slice.Tensor(x, 0, 0, 4).relu()

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(4, 5))
        agp.fold_slice(gm.graph)
        self.assertEqual(_count_target(gm, torch.ops.aten.slice.Tensor), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
