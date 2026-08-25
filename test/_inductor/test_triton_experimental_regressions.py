# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Owner(s): ["module: inductor"]

from unittest import mock

import sympy
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._ordered_set import OrderedSet

from torch_npu._inductor.triton_experimental import lowering as experimental_lowering
from torch_npu._inductor.triton_experimental import lowering_override_list
from torch_npu._inductor.triton_experimental.codegen import triton as npu_triton_codegen


class TestTritonExperimentalRegressions(TestCase):
    def test_control_deps_is_not_replaced_with_fallback(self):
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

    def test_constant_index_normalizes_emitted_and_cse_shapes(self):
        upstream_result = IndexingOptions(
            "0",
            OrderedSet(),
            "[1, 1]",
            False,
            sympy.Integer(0),
            expand_shape=(1, 1),
        )
        kernel = object.__new__(npu_triton_codegen.NPUTritonKernel)
        kernel.range_trees = []
        kernel._load_mask = None

        with (
            mock.patch.object(
                TritonKernel, "indexing", return_value=upstream_result
            ),
            mock.patch.object(
                npu_triton_codegen, "triton_codegen_linearize", True
            ),
            mock.patch.object(kernel, "filter_masks"),
        ):
            result = kernel.indexing(sympy.Integer(0))

        self.assertEqual(result.expand_str, "[1]")
        self.assertEqual(result.expand_shape, (1,))

    def test_constant_index_keeps_explicit_copy_shape(self):
        upstream_result = IndexingOptions(
            "0",
            OrderedSet(),
            "[1, 1]",
            False,
            sympy.Integer(0),
            expand_shape=(1, 1),
        )
        kernel = object.__new__(npu_triton_codegen.NPUTritonKernel)
        kernel.range_trees = []
        kernel._load_mask = None

        with (
            mock.patch.object(
                TritonKernel, "indexing", return_value=upstream_result
            ),
            mock.patch.object(
                npu_triton_codegen, "triton_codegen_linearize", True
            ),
            mock.patch.object(kernel, "filter_masks"),
        ):
            result = kernel.indexing(sympy.Integer(0), copy_shape=(1, 1))

        self.assertIs(result, upstream_result)
        self.assertEqual(result.expand_str, "[1, 1]")
        self.assertEqual(result.expand_shape, (1, 1))

if __name__ == "__main__":
    run_tests()
