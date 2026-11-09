import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import sympy
import torch
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._ordered_set import OrderedSet

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.kernel import flex_attention as npu_flex_attention
from torch_npu._inductor.triton_experimental import launcher_codegen
from torch_npu._inductor.triton_experimental import lowering as experimental_lowering
from torch_npu._inductor.triton_experimental import lowering_override_list
from torch_npu._inductor.triton_experimental.codegen import triton as npu_triton_codegen


class TestCodegenRegressions(TestCase):
    def test_codegen_control_deps_keeps_upstream_lowering(self):
        self.assertIn(control_deps, lowering_override_list.KEEP_UPSTREAM_LOWERING)
        make_fallback = mock.Mock()
        with (
            mock.patch.object(
                experimental_lowering, "lowerings", {control_deps: object()}
            ),
            mock.patch.object(experimental_lowering, "decompositions", {}),
            mock.patch.object(experimental_lowering, "make_fallback", make_fallback),
            mock.patch.object(experimental_lowering, "FALLBACK_LIST", []),
        ):
            experimental_lowering._register_npu_inductor_fallbacks()
        self.assertNotIn(mock.call(control_deps), make_fallback.call_args_list)

    def test_codegen_constant_index_shape_normalization(self):
        upstream_result = IndexingOptions(
            "0", OrderedSet(), "[1, 1]", False, sympy.Integer(0), expand_shape=(1, 1)
        )
        kernel = object.__new__(npu_triton_codegen.NPUTritonKernel)
        kernel._npu_linearize = True
        kernel.range_trees = [SimpleNamespace(numel=4, is_reduction=False)]
        kernel._load_mask = None
        sizevars = SimpleNamespace(statically_known_equals=lambda expr, value: False)
        virtualized = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with (
            mock.patch.object(TritonKernel, "indexing", return_value=upstream_result),
            mock.patch.object(npu_triton_codegen, "triton_codegen_linearize", True),
            mock.patch.object(npu_triton_codegen, "V", virtualized),
            mock.patch.object(kernel, "filter_masks"),
        ):
            result = kernel.indexing(sympy.Integer(0))
        self.assertEqual(result.expand_str, "[1]")
        self.assertEqual(result.expand_shape, (1,))

    def test_codegen_constant_index_shape_normalization_degenerate(self):
        # 0-d extreme (no range trees): keep the honest 0-d shape (#5020).
        upstream_result = IndexingOptions(
            "0", OrderedSet(), "[1, 1]", False, sympy.Integer(0), expand_shape=(1, 1)
        )
        kernel = object.__new__(npu_triton_codegen.NPUTritonKernel)
        kernel._npu_linearize = True
        kernel.range_trees = []
        kernel._load_mask = None
        with (
            mock.patch.object(TritonKernel, "indexing", return_value=upstream_result),
            mock.patch.object(npu_triton_codegen, "triton_codegen_linearize", True),
            mock.patch.object(kernel, "filter_masks"),
        ):
            result = kernel.indexing(sympy.Integer(0))
        self.assertEqual(result.expand_str, "")
        self.assertEqual(result.expand_shape, ())

    def test_codegen_scalar_pointwise_grid_clamp_contract(self):
        cases = (
            ({"npu_num_x_nodes": 0, "grid_type": "Grid1D"}, True),
            ({"npu_num_x_nodes": 1, "grid_type": "Grid1D"}, True),
            ({"npu_num_x_nodes": 2, "grid_type": "Grid1D"}, False),
            ({"npu_num_x_nodes": 0, "grid_type": "Grid2D"}, False),
        )
        for meta, expected in cases:
            with self.subTest(meta=meta):
                actual = launcher_codegen._can_clamp_1d_grid(
                    meta,
                    ["xnumel"],
                    ["out_ptr0", "xnumel", "XBLOCK"],
                )
                self.assertEqual(actual, expected)

    def test_flex_attention_precision_legacy_getter_compatibility(self):
        options = {}
        with mock.patch.object(
            npu_flex_attention, "get_float32_precision", return_value="'tf32'"
        ):
            npu_flex_attention._set_flex_attention_float32_precision(options, torch.float32)
        self.assertEqual(options["FLOAT32_PRECISION"], "'tf32'")

    def test_flex_attention_precision_setter_compatibility(self):
        options = {}
        upstream_setter = mock.Mock()
        with (
            mock.patch.object(npu_flex_attention, "get_float32_precision", None),
            mock.patch.object(
                npu_flex_attention,
                "_upstream_set_float32_precision",
                upstream_setter,
            ),
        ):
            npu_flex_attention._set_flex_attention_float32_precision(options, torch.bfloat16)
        upstream_setter.assert_called_once_with(options, torch.bfloat16)

    def test_flex_attention_precision_respects_explicit_option(self):
        options = {"FLOAT32_PRECISION": "'ieee'"}
        with (
            mock.patch.object(npu_flex_attention, "get_float32_precision", None),
            mock.patch.object(
                npu_flex_attention, "_upstream_set_float32_precision"
            ) as upstream_setter,
        ):
            npu_flex_attention._set_flex_attention_float32_precision(options, torch.float32)
        self.assertEqual(options["FLOAT32_PRECISION"], "'ieee'")
        upstream_setter.assert_not_called()

    def test_codegen_min_max_reduction_propagates_nan(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        x = torch.randn(8, 64, 128, device="npu")
        x[3, 5, :] = float("nan")

        def reduce_values(t):
            return torch.amax(t, dim=1)

        expected = reduce_values(x)
        compiled = torch.compile(
            reduce_values,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, x)
        self.assertIn("npu_triton_heuristics", "\n".join(codes))
        torch.testing.assert_close(actual, expected, equal_nan=True)
        self.assertTrue(bool(actual.isnan().any()))

    def test_codegen_promoted_rtree_scalar_store(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        batch, seq, vocab = 4, 32, 16
        logits = torch.randn(batch, seq, vocab, device="npu")
        labels = torch.randint(0, vocab, (batch, seq), device="npu", dtype=torch.int64)
        labels[:, :4] = -100

        def loss(t, target):
            lsm = torch.log_softmax(t[:, :-1, :].float(), dim=-1)
            return torch.nn.functional.nll_loss(
                lsm.reshape(-1, lsm.shape[-1]),
                target[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        expected = loss(logits, labels)
        compiled = torch.compile(
            loss,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, logits, labels)
        self.assertIn("npu_triton_heuristics", "\n".join(codes))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_pad_transform_division_family_is_sound(self):
        # Reduction-guard elision drops the accumulator ``tl.where`` only when
        # the pad-lane value is provably the reduction identity.  Division by
        # a zero-valued pad yields inf (1/0) which then poisons ``0 * inf``:
        # the lattice must answer "top" (unknown), never "fin"/const-0.
        pad = npu_triton_codegen._pad_transform
        top = npu_triton_codegen._PAD_TOP
        fin = npu_triton_codegen._PAD_FIN
        # truediv by a zero pad: 1/0 -> inf (was wrongly "fin" before the fix)
        self.assertEqual(pad("truediv", [("c", 1.0), ("c", 0.0)]), top)
        # the full __rdiv__-backward chain: 0 * (1/0) -> nan, must stay top
        quotient = pad("truediv", [("c", 1.0), ("c", 0.0)])
        self.assertEqual(pad("mul", [("c", 0.0), quotient]), top)
        # non-constant operands: a finite divisor may still be zero on pads
        self.assertEqual(pad("truediv", [fin, fin]), top)
        self.assertEqual(pad("truediv", [("c", 1.0), fin]), top)
        # reciprocal(0)=inf, rsqrt(0)=inf, rsqrt(neg)=nan, sqrt(neg)=nan,
        # log(0)=-inf: none may be deemed finite
        self.assertEqual(pad("reciprocal", [("c", 0.0)]), top)
        self.assertEqual(pad("rsqrt", [("c", 0.0)]), top)
        self.assertEqual(pad("rsqrt", [("c", -1.0)]), top)
        self.assertEqual(pad("sqrt", [("c", -1.0)]), top)
        self.assertEqual(pad("log", [("c", 0.0)]), top)
        # fmod(x, 0)=nan; pow(0, -1)=inf, pow(neg, frac)=nan; a//0 overflows
        self.assertEqual(pad("fmod", [fin, ("c", 0.0)]), top)
        self.assertEqual(pad("pow", [("c", 0.0), ("c", -1.0)]), top)
        self.assertEqual(pad("remainder", [fin, ("c", 0.0)]), top)
        self.assertEqual(pad("floordiv", [fin, ("c", 0.0)]), top)
        # the rest of the family: finite pads that produce inf/nan
        self.assertEqual(pad("log2", [("c", 0.0)]), top)
        self.assertEqual(pad("log10", [("c", 0.0)]), top)
        self.assertEqual(pad("log1p", [("c", -1.0)]), top)
        self.assertEqual(pad("lgamma", [("c", 0.0)]), top)
        self.assertEqual(pad("exp2", [("c", 1000.0)]), top)
        self.assertEqual(pad("expm1", [("c", 1000.0)]), top)
        self.assertEqual(pad("cosh", [("c", 1000.0)]), top)
        self.assertEqual(pad("sinh", [("c", 1000.0)]), top)
        self.assertEqual(pad("asin", [("c", 2.0)]), top)
        self.assertEqual(pad("acos", [("c", 2.0)]), top)
        self.assertEqual(pad("acosh", [("c", 0.5)]), top)
        self.assertEqual(pad("atanh", [("c", 1.0)]), top)
        self.assertEqual(pad("erfinv", [("c", 2.0)]), top)
        self.assertEqual(pad("mod", [("c", 1.0), ("c", 0.0)]), top)
        # the default rule itself must be top for anything outside the
        # finite-total allowlist: the upstream vocabulary carries these
        # singularity/overflow ops (verified present in ops_handler.py)
        for name in (
            "digamma",
            "polygamma",
            "zeta",
            "ndtri",
            "bessel_y0",
            "bessel_y1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "igamma",
            "igammac",
            "gammainc",
            "gammaincc",
            "div_rn",
            "int_truediv",
            "erfcx",
            "ldexp",
            "i0",
            "i1",
            "log_ndtr",
            "chebyshev",
            "hermite",
            "laguerre",
            "legendre",
            "inline_asm_elementwise",
        ):
            self.assertEqual(pad(name, [fin, ("c", 0.0)]), top)
            self.assertEqual(pad(name, [fin]), top)

    def test_pad_transform_finite_total_allowlist(self):
        # Only ops known total on finite pads (finite operands -> finite
        # result) may answer "fin"; the default for everything else is top.
        pad = npu_triton_codegen._pad_transform
        fin = npu_triton_codegen._PAD_FIN
        top = npu_triton_codegen._PAD_TOP
        for name in (
            "sign",
            "relu",
            "sin",
            "cos",
            "tanh",
            "sigmoid",
            "erf",
            "erfc",
            "atan",
            "asinh",
            "floor",
            "ceil",
            "trunc",
            "round",
            "lt",
            "eq",
            "logical_not",
            "logical_and",
            "bitwise_or",
        ):
            self.assertEqual(pad(name, [fin, fin]), fin, name)
        # a non-finite operand still answers top on the allowlist
        self.assertEqual(pad("sin", [("c", float("inf"))]), top)
        self.assertEqual(pad("relu", [top]), top)

    def test_pad_transform_division_family_exact_constants(self):
        # A constant fold is provable only when it is exactly the value the
        # hardware computes: f32-representable and round-trip verified, so
        # float64-only identities ((1/41)*41 == 1.0, sqrt(11)^2 == 11.0)
        # never feed the guard elision.
        pad = npu_triton_codegen._pad_transform
        top = npu_triton_codegen._PAD_TOP
        fin = npu_triton_codegen._PAD_FIN
        # exactly representable quotients keep their proofs
        self.assertEqual(pad("truediv", [("c", 1.0), ("c", 2.0)]), ("c", 0.5))
        self.assertEqual(pad("truediv", [("c", 10.0), ("c", 2.0)]), ("c", 5.0))
        self.assertEqual(pad("truediv", [("c", 0.0), ("c", 2.0)]), ("c", 0.0))
        self.assertEqual(pad("reciprocal", [("c", 4.0)]), ("c", 0.25))
        self.assertEqual(pad("sqrt", [("c", 4.0)]), ("c", 2.0))
        # float64-exact but fp32-inexact values must NOT fold
        self.assertEqual(pad("truediv", [("c", 1.0), ("c", 41.0)]), top)
        self.assertEqual(pad("reciprocal", [("c", 41.0)]), top)
        self.assertEqual(pad("sqrt", [("c", 11.0)]), top)
        self.assertEqual(pad("log", [("c", 2.0)]), top)
        # an fp32-overflowing product is inf on hardware, not the finite
        # float64 folded here (0 * inf would be nan on the pad lane)
        self.assertEqual(pad("mul", [("c", 0.0), ("c", 3.0e39)]), top)
        self.assertEqual(pad("add", [("c", 3.0e38), ("c", 3.0e38)]), top)
        for name, a, b in (("add", float("inf"), -3e39),
                           ("sub", float("inf"), 3e39),
                           ("mul", float("inf"), 1e-50)):
            self.assertEqual(pad(name, [("c", a), ("c", b)]), top)
        self.assertEqual(pad("add", [("c", 3e39), fin]), top)
        # exp folds only its exact cases
        self.assertEqual(pad("exp", [("c", 0.0)]), ("c", 1.0))
        self.assertEqual(pad("exp", [("c", float("-inf"))]), ("c", 0.0))
        self.assertEqual(pad("exp", [("c", 710.0)]), ("c", float("inf")))
        self.assertEqual(pad("exp", [("c", 1.0)]), top)
        self.assertEqual(pad("exp", [("c", 89.0)]), top)
        # A finite f64 operand can overflow/underflow before fp32 division.
        for a, b in ((3e39, float("inf")), (float("inf"), 3e39),
                     (float("inf"), 1e-50)):
            self.assertEqual(pad("truediv", [("c", a), ("c", b)]), top)
        # IEEE-exact non-finite quotients
        self.assertEqual(pad("truediv", [("c", float("inf")), ("c", 2.0)]), ("c", float("inf")))
        self.assertEqual(pad("truediv", [("c", 1.0), ("c", float("inf"))]), ("c", 0.0))
        self.assertEqual(pad("reciprocal", [("c", float("inf"))]), ("c", 0.0))
        self.assertEqual(pad("sqrt", [("c", float("inf"))]), ("c", float("inf")))
        self.assertEqual(pad("log", [("c", float("inf"))]), ("c", float("inf")))
        # rsqrt folds only its hardware-exact identity results; tl.rsqrt is
        # not correctly rounded, so other positives must stay top
        self.assertEqual(pad("rsqrt", [("c", 1.0)]), ("c", 1.0))
        self.assertEqual(pad("rsqrt", [("c", float("inf"))]), ("c", 0.0))
        self.assertEqual(pad("rsqrt", [("c", 4.0)]), top)
        # max/min with a nan operand: Python max is order-dependent on nan,
        # which does not match tl.maximum, so the result is unknowable
        self.assertEqual(pad("maximum", [("c", float("nan")), ("c", 5.0)]), top)
        self.assertEqual(pad("minimum", [("c", 5.0), ("c", float("nan"))]), top)
        # fin - (c, inf) is exactly -inf
        self.assertEqual(pad("sub", [fin, ("c", float("inf"))]), ("c", float("-inf")))
        # non-finite const-const results are dtype-invariant and stay folded
        self.assertEqual(pad("add", [("c", float("inf")), ("c", 5.0)]), ("c", float("inf")))
        self.assertEqual(pad("mul", [("c", 2.0), ("c", float("inf"))]), ("c", float("inf")))
        self.assertEqual(pad("sub", [("c", float("inf")), ("c", float("inf"))]), ("c", float("nan")))

    def test_codegen_reduction_pad_division_keeps_guard(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        # Exercise masked reciprocal reduction lanes; 257 avoids scalar unrolling.
        # Padding must retain a guard or use a safe divisor fill.
        # Fix the seed to avoid cancellation-sensitive reassociation noise.
        torch.manual_seed(0)
        t = torch.randn(257, device="npu")
        x = torch.randn(257, device="npu")

        def scalar_grad(t, x):
            return (t * x.reciprocal()).sum()

        expected = scalar_grad(t, x)
        compiled = torch.compile(
            scalar_grad, options={"npu_backend": "triton_experimental"}
        )
        actual, codes = run_and_get_code(compiled, t, x)
        self.assertIn("npu_triton_heuristics", "\n".join(codes))
        # sound outcomes only: the accumulator where-guard survives, or the
        # fill solver re-filled the divisor load with -inf (0 * (1/-inf) is
        # the sum identity); an unguarded other=0.0 divisor load is the bug
        joined = "\n".join(codes)
        self.assertTrue(
            "tl.where" in joined or "float('-inf')" in joined or 'float("-inf")' in joined,
            "neither the accumulator guard nor a -inf divisor fill is present",
        )
        self.assertFalse(bool(torch.isnan(actual).any()))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    run_tests()
