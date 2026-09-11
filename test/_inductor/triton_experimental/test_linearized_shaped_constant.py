# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401


class TestLinearizedShapedConstant(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_zero_fill_rank_matches_split_linearized_store(self):
        rows, cols, stride = 32, 119, 128

        def zero_in_place(result):
            return result.zero_()

        inp = torch.empty_strided(
            (rows, cols), (stride, 1), dtype=torch.float32, device="npu"
        )
        expected = torch.zeros_like(inp)
        compiled = torch.compile(
            zero_in_place,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, inp)

        self.assertEqual(actual, expected)
        self.assertEqual(actual.stride(), (stride, 1))
        source = "\n".join(codes)
        self.assertIn("tl.full([1, 1], 0.0, tl.float32)", source)
        self.assertNotIn("tl.full([1], 0.0, tl.float32)", source)


if __name__ == "__main__":
    run_tests()
