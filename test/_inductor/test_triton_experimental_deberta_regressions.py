# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.triton_experimental import lowering as experimental_lowering
from torch_npu._inductor.triton_experimental import npu_triton_heuristics


class TestTritonExperimentalDebertaRegressions(TestCase):
    def test_bernoulli_fallback_disables_npugraph_capture(self):
        ops = (
            torch.ops.aten.bernoulli.default,
            torch.ops.aten.bernoulli.out,
            torch.ops.aten.bernoulli.p,
            torch.ops.aten.bernoulli_.Tensor,
            torch.ops.aten.bernoulli_.float,
        )
        original_lowerings = {
            op: mock.Mock(return_value=f"lowered:{op}") for op in ops
        }
        lowerings = dict(original_lowerings)

        with (
            mock.patch.object(experimental_lowering, "lowerings", lowerings),
            mock.patch.object(
                experimental_lowering,
                "decompositions",
                {op: object() for op in ops},
            ),
            mock.patch.object(experimental_lowering, "FALLBACK_LIST", []),
        ):
            experimental_lowering._register_npu_inductor_fallbacks()

        for op in ops:
            with self.subTest(op=op):
                graph = mock.Mock(disable_cudagraphs_reason=None)
                with experimental_lowering.V.set_graph_handler(graph):
                    result = lowerings[op]("input", p=0.5)

                self.assertEqual(result, f"lowered:{op}")
                self.assertIn(str(op), graph.disable_cudagraphs_reason)
                self.assertIn(
                    "no graph-safe seed/offset ABI", graph.disable_cudagraphs_reason
                )
                original_lowerings[op].assert_called_once_with("input", p=0.5)

        graph = mock.Mock(disable_cudagraphs_reason="existing reason")
        with experimental_lowering.V.set_graph_handler(graph):
            lowerings[ops[0]]("input")
        self.assertEqual(graph.disable_cudagraphs_reason, "existing reason")

    def test_scalar_pointwise_grid_is_clamped(self):
        can_clamp = npu_triton_heuristics._can_clamp_1d_grid

        cases = (
            (
                "scalar_grid1d",
                {"npu_num_x_nodes": 0, "grid_type": "Grid1D"},
                ["xnumel"],
                ["in_out_ptr0", "xnumel", "XBLOCK"],
                True,
            ),
            (
                "single_node_grid1d",
                {"npu_num_x_nodes": 1, "grid_type": "Grid1D"},
                ["xnumel"],
                ["out_ptr0", "xnumel", "XBLOCK"],
                True,
            ),
            (
                "missing_node_metadata",
                {"grid_type": "Grid1D"},
                ["xnumel"],
                ["out_ptr0", "xnumel", "XBLOCK"],
                False,
            ),
            (
                "multiple_x_nodes",
                {"npu_num_x_nodes": 2, "grid_type": "Grid1D"},
                ["xnumel"],
                ["out_ptr0", "xnumel", "XBLOCK"],
                False,
            ),
            (
                "grid2d",
                {"npu_num_x_nodes": 0, "grid_type": "Grid2D"},
                ["xnumel"],
                ["out_ptr0", "xnumel", "XBLOCK"],
                False,
            ),
            (
                "missing_xnumel",
                {"npu_num_x_nodes": 0, "grid_type": "Grid1D"},
                [],
                ["out_ptr0", "XBLOCK"],
                False,
            ),
            (
                "reduction",
                {"npu_num_x_nodes": 0, "grid_type": "Grid1D"},
                ["xnumel"],
                ["out_ptr0", "xnumel", "XBLOCK", "R0_BLOCK"],
                False,
            ),
        )

        for name, meta, def_args, arg_names, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(can_clamp(meta, def_args, arg_names), expected)


if __name__ == "__main__":
    run_tests()
