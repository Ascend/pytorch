# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

from unittest import mock

import sympy
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental.codegen import triton as npu_triton_codegen


class TestTritonExperimentalNativeBertRegressions(TestCase):
    def test_outer_rsplit_rejects_fused_pointwise_output(self):
        reduction_node = mock.Mock()
        reduction_node.node.data.reduction_type = "sum"
        features = mock.Mock()
        features.get_reduction_hint.return_value = (
            npu_triton_codegen.ReductionHint.OUTER
        )
        features.reduction_nodes.return_value = [reduction_node]

        kernel = mock.Mock(inside_reduction=True, features=features)
        kernel.args.output_buffers = {
            "materialized_pointwise": mock.Mock(inner_name="out_ptr0"),
            "reduction": mock.Mock(inner_name="out_ptr1"),
        }

        self.assertFalse(npu_triton_codegen._npu_rsplit_outer_applicable(kernel))

    def test_outer_rsplit_rejects_nested_reduction_tree(self):
        reduction_node = mock.Mock()
        reduction_node.node.data.reduction_type = "sum"
        features = mock.Mock()
        features.get_reduction_hint.return_value = (
            npu_triton_codegen.ReductionHint.OUTER
        )
        features.reduction_nodes.return_value = [reduction_node]

        first = mock.Mock(name="r0_1")
        first.name = "r0_1"
        second = mock.Mock(name="r0_2")
        second.name = "r0_2"
        reduction_tree = mock.Mock(is_reduction=True)
        reduction_tree.nodes = {"r0_1": first, "r0_2": second}
        reduction_tree.tree_node_mapping = {}

        kernel = mock.Mock(inside_reduction=True, features=features)
        kernel.args.output_buffers = {
            "reduction": mock.Mock(inner_name="out_ptr0")
        }
        kernel.range_trees = [reduction_tree]

        self.assertFalse(npu_triton_codegen._npu_rsplit_outer_applicable(kernel))

    def test_dualview_reduction_fold_reuses_registered_flat_symbol(self):
        def make_node(name, divisor, length, symbol):
            node = mock.Mock()
            node.name = name
            node.divisor = sympy.Integer(divisor)
            node.length = sympy.Integer(length)
            node.symbol.return_value = symbol
            return node

        flat = sympy.Symbol("r0_1", integer=True, nonnegative=True)
        inner = sympy.Symbol("r0_2", integer=True, nonnegative=True)
        outer = sympy.Symbol("r0_3", integer=True, nonnegative=True)
        flat_node = make_node("r0_1", 1, 2048, flat)
        inner_node = make_node("r0_2", 1, 512, inner)
        outer_node = make_node("r0_3", 512, 4, outer)

        tree = mock.Mock(is_reduction=True)
        tree.nodes = {
            sympy.Symbol("flat_expr"): flat_node,
            sympy.Symbol("inner_expr"): inner_node,
            sympy.Symbol("outer_expr"): outer_node,
        }
        kernel = object.__new__(npu_triton_codegen.NPUTritonKernel)
        kernel.inside_reduction = True
        kernel.range_trees = [tree]

        sizevars = mock.Mock()
        sizevars.optimization_hint.side_effect = int
        sizevars.statically_known_equals.side_effect = lambda a, b: a == b
        graph = mock.Mock(sizevars=sizevars)
        index = inner + 512 * outer

        with (
            mock.patch.object(
                npu_triton_codegen.ncfg, "fold_dualview_rnode", True
            ),
            npu_triton_codegen.V.set_graph_handler(graph),
        ):
            folded = kernel._fold_dualview_reduction_index(index)

        self.assertEqual(folded, flat)
        self.assertIn(folded, {flat: flat_node})
        self.assertNotEqual(folded, sympy.Symbol("r0_1"))

    def test_downcast_inout_preserves_input_before_writeback(self):
        def launcher(in_out_ptr0, *, stream=None):
            self.assertIs(in_out_ptr0.dtype, torch.int32)
            self.assertEqual(in_out_ptr0, torch.tensor([1, 2], dtype=torch.int32))
            in_out_ptr0.add_(1)

        launcher._npu_def_args = ["in_out_ptr0"]
        wrapped = npu_triton_heuristics._wrap_launcher_with_downcast(
            launcher,
            downcast_args={"in_out_ptr0": "*i64"},
            mutated_arg_names={"in_out_ptr0"},
        )
        original = torch.tensor([1, 2], dtype=torch.int64)

        wrapped(original, stream=None)

        self.assertEqual(original, torch.tensor([2, 3], dtype=torch.int64))


if __name__ == "__main__":
    run_tests()
