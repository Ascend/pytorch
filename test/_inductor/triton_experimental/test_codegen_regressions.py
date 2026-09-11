import os
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
from torch_npu._inductor.triton_experimental import lowering as experimental_lowering
from torch_npu._inductor.triton_experimental import lowering_override_list
from torch_npu._inductor.triton_experimental import npu_triton_heuristics
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
        make_fallback.assert_not_called()

    def test_codegen_constant_index_shape_normalization(self):
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
        self.assertEqual(result.expand_str, "[1]")
        self.assertEqual(result.expand_shape, (1,))

    def test_codegen_scalar_pointwise_grid_clamp_contract(self):
        cases = (
            ({"npu_num_x_nodes": 0, "grid_type": "Grid1D"}, True),
            ({"npu_num_x_nodes": 1, "grid_type": "Grid1D"}, True),
            ({"npu_num_x_nodes": 2, "grid_type": "Grid1D"}, False),
            ({"npu_num_x_nodes": 0, "grid_type": "Grid2D"}, False),
        )
        for meta, expected in cases:
            with self.subTest(meta=meta):
                actual = npu_triton_heuristics._can_clamp_1d_grid(
                    meta,
                    ["xnumel"],
                    ["out_ptr0", "xnumel", "XBLOCK"],
                )
                self.assertEqual(actual, expected)

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
