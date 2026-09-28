# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Regression coverage for the two generic correctness fixes.

Each test anchors the domain of one fix with a compiled-vs-eager numerics
check on a pattern that exercises the fixed path:

- mixed-leaf mul_where / xmask safety: reduction-only workloads previously
  hit a NameError when the emitted kernel referenced an undefined xmask
  (no pointwise leaves). Reduction and reduction+where patterns must
  compile and stay exact.
- rsplit sub-axis numel mapping: cross-core OUTER reductions (rsplit)
  previously mis-sized sub-axes like ``r0_0`` by mapping numel onto the
  wrong parent prefix. Outer-axis reductions must stay exact.

(The third original fix in this family, stride-preserving boundary
downcast, is owned by #46074 — its complete form with the deepcopy
broadcast branch and a dedicated test file lives there.)

The original failures were model-scale (yolov3 graphs); these miniatures
pin the fixed paths' correctness at unit scale.
"""

import unittest

import torch  # noqa: F401
import torch_npu  # noqa: F401


def _compile(fn):
    return torch.compile(fn, options={"npu_backend": "triton_experimental"})


class TestXmaskSafeReductionOnly(unittest.TestCase):
    def test_plain_outer_sum(self):
        torch.npu.manual_seed(0)
        x = torch.randn(64, 96, device="npu")
        fn = lambda x: x.sum(dim=0)  # noqa: E731
        torch.testing.assert_close(_compile(fn)(x), fn(x))

    def test_reduction_with_where_leaf(self):
        # mul/where leaves around a reduction: the mixed-leaf preserve
        # path that previously referenced xmask unconditionally.
        torch.npu.manual_seed(0)
        x = torch.randn(32, 128, device="npu")
        mask = torch.randn(32, 128, device="npu") > 0

        def fn(x, m):
            return torch.where(m, x, torch.zeros_like(x)).sum(dim=-1)
        torch.testing.assert_close(_compile(fn)(x, mask), fn(x, mask))

    def test_reduction_min_max_mix(self):
        torch.npu.manual_seed(0)
        x = torch.randn(16, 256, device="npu")

        def fn(x):
            return x.sum(dim=-1) - x.min(dim=-1).values - x.max(dim=-1).values
        torch.testing.assert_close(_compile(fn)(x), fn(x))


class TestRsplitSubAxisNumel(unittest.TestCase):
    def test_large_outer_reduction(self):
        # Outer-axis reduction big enough to engage the cross-core rsplit
        # (sub-axes r0_0/r0_1 carry the split partials).
        torch.npu.manual_seed(0)
        x = torch.randn(8192, 64, device="npu")
        fn = lambda x: x.sum(dim=0)  # noqa: E731
        torch.testing.assert_close(_compile(fn)(x), fn(x), atol=1e-4, rtol=1e-4)

    def test_outer_reduction_two_splits(self):
        torch.npu.manual_seed(0)
        x = torch.randn(16384, 48, device="npu")

        def fn(x):
            return (x * 2 + 1).sum(dim=0).relu()
        torch.testing.assert_close(_compile(fn)(x), fn(x), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
