import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import sympy
import torch
import torch_npu._inductor  # noqa: F401
from torch._dynamo.source import LocalSource
from torch._inductor.ir import FixedLayout, Layout
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._sympy.functions import FloorDiv, Mod
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter


class TestSymbolicStrideComparison(TestCase):
    def test_linear_and_squared_stride_order(self):
        env = ShapeEnv()
        h = env.create_symbol(256, LocalSource("h"), dynamic_dim=DimDynamic.DYNAMIC)
        env.constrain_symbol_range(h, 33, int_oo)
        side = FloorDiv(h - 1, 32) + 1
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator(env))):
            self.assertFalse(Layout._stride_expr_ge_or_false(side, side**2))
            self.assertTrue(Layout._stride_expr_ge_or_false(side**2, side))

    def test_polynomial_stride_order(self):
        for expanded_left in (False, True):
            with self.subTest(expanded_left=expanded_left):
                env = ShapeEnv()
                h, w, c = [
                    env.create_symbol(value, LocalSource(name), dynamic_dim=DimDynamic.DYNAMIC)
                    for name, value in (("h", 256), ("w", 256), ("c", 16))
                ]
                spatial = (FloorDiv(h - 1, 32) + 1) * (FloorDiv(w - 1, 32) + 1)
                left = sympy.expand(spatial) if expanded_left else spatial
                with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator(env))):
                    self.assertFalse(Layout._stride_expr_ge_or_false(left, c * spatial))
                    self.assertTrue(Layout._stride_expr_ge_or_false(c * spatial, left))

    def test_static_strides_and_zero(self):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            for left, right, expected in ((0, 0, True), (0, 8, False), (8, 0, True),
                                          (8, 8, True), (4, 8, False), (8, 4, True)):
                with self.subTest(left=left, right=right):
                    self.assertEqual(Layout._stride_expr_ge_or_false(left, right), expected)

    def test_unbacked_compound_stride_order(self):
        # Regression from pytorch/pytorch@93879b11e3b5: a structural proof must
        # work without extracting an optimization hint from an unbacked size.
        env = ShapeEnv()
        seq = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(seq, 4, 1024)
        half = FloorDiv(seq, 2)
        reduced = FloorDiv(half**2, half)
        left = 16 * reduced - 30
        right = 16 * (64 * half * reduced - 120 * half - 120 * reduced + 225)
        sizevars = SizeVarAllocator(env)
        layout = FixedLayout(torch.device("npu"), torch.float32, size=[seq, 2], stride=[left, right])
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            with mock.patch.object(sizevars, "optimization_hint", side_effect=AssertionError("unexpected hint")):
                self.assertTrue(layout.is_stride_ordered([0, 1]))

    def test_fallback_guard_has_no_negative_powers(self):
        env = ShapeEnv()
        seq = env.create_unbacked_symint().node.expr
        half = FloorDiv(seq, 2)
        reduced = FloorDiv(half**2, half)
        left = 16 * reduced - 30
        right = 16 * (64 * half * reduced - 120 * half - 120 * reduced + 225)
        sizevars = mock.Mock()
        sizevars.guard_or_false.return_value = False
        with mock.patch(
            "torch_npu._inductor.codegen._sizevars._statically_known_stride_multiple_of", return_value=False
        ) as prove_multiple:
            with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
                self.assertFalse(Layout._stride_expr_ge_or_false(left, right))
        prove_multiple.assert_called_once_with(sizevars, left, right)
        guard = sizevars.guard_or_false.call_args.args[0]
        self.assertIsInstance(guard.lhs, Mod)
        self.assertEqual([power for power in guard.atoms(sympy.Pow) if power.exp.is_negative], [])
        self.assertIn("%", PythonPrinter().doprint(guard))

    def test_restore_preserves_staticmethod_descriptor(self):
        # Baseline restoration changes process-wide registries. Keep that
        # check isolated without re-registering unrelated decompositions.
        subprocess.check_call([sys.executable, "-c", """
import torch_npu._inductor
from torch._inductor.ir import Layout
from torch_npu._inductor.codegen._sizevars import patch_simplify
from torch_npu._inductor.lowering_patch import restore_inductor_baseline
patched = Layout._stride_expr_ge_or_false
patch_simplify()
assert Layout._stride_expr_ge_or_false is patched
restore_inductor_baseline()
assert not getattr(Layout._stride_expr_ge_or_false, '_torch_npu_integer_strides', False)
assert isinstance(vars(Layout)['_stride_expr_ge_or_false'], staticmethod)
patch_simplify()
assert Layout._stride_expr_ge_or_false._torch_npu_integer_strides
"""])


if __name__ == "__main__":
    run_tests()
