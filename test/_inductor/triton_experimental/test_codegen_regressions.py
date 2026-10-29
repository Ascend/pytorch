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


if __name__ == "__main__":
    run_tests()
