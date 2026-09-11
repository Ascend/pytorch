# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401


class TestRedundantRangeTreeDecomposition(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_partial_alternate_range_tree_span_is_folded(self):
        outer, choices, side = 6, 4, 8

        def select_scatter_update(base, selected):
            update = torch.select_scatter(
                torch.zeros_like(base), selected, 1, choices - 1
            )
            return base + update

        base = torch.randn(outer, choices, side, side, device="npu")
        selected = torch.randn(outer, side, side, device="npu")
        expected = select_scatter_update(base, selected)
        compiled = torch.compile(
            select_scatter_update,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, base, selected)

        self.assertEqual(actual, expected, rtol=1e-4, atol=1e-5)
        source = "\n".join(codes)
        self.assertIn("@triton.jit", source)
        self.assertIn("'npu_num_x_nodes': 3", source)
        self.assertNotIn("torch.ops.aten.select_scatter.default(", source)

    def test_cross_tree_index_expression_is_not_registered_as_alias(self):
        outer, choices, side = 6, 4, 8

        def select_scatter_update(base, selected):
            update = torch.select_scatter(
                torch.zeros_like(base), selected, 1, choices - 1
            )
            return base + update

        base = torch.randn(outer, choices, side, side, device="npu").transpose(
            -1, -2
        )
        selected = torch.randn(outer, side, side, device="npu").transpose(-1, -2)
        expected = select_scatter_update(base, selected)
        compiled = torch.compile(
            select_scatter_update,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, base, selected)

        self.assertEqual(actual, expected, rtol=1e-4, atol=1e-5)
        source = "\n".join(codes)
        self.assertIn("@triton.jit", source)
        self.assertIn("'npu_num_x_nodes': 3", source)
        self.assertNotIn("x3 + 64*y0 =", source)
        self.assertNotIn("torch.ops.aten.select_scatter.default(", source)


if __name__ == "__main__":
    run_tests()
