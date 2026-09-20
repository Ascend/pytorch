# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Numerics tests for the P1 chained index_select rewrite (guard-only).

The rewrite admits exactly the separable outer-product form
x[:, :, ih[:, None], iw] (first index an (M, 1) column, second 1-D) and
lowers it to chained aclnn index_select externs.  Every other shape
combination (pairing/diagonal both-1-D, single (M, 1) run, rank >= 3)
keeps the upstream aclnnIndex fallback.  Both paths must be numerically
exact vs eager — these tests pin the guard boundary from the correctness
side; which path a case takes is asserted against the compiled artifact
in the campaign verification scripts, not here.
"""

import unittest

import torch  # noqa: F401
import torch_npu  # noqa: F401


def _mk(N, C, H, W, M, K, seed=0):
    torch.npu.manual_seed(seed)
    x = torch.randn(N, C, H, W, device="npu")
    ih = torch.randint(0, H, (M, 1), device="npu", dtype=torch.int64)
    iw = torch.randint(0, W, (K,), device="npu", dtype=torch.int64)
    return x, ih, iw


class TestUnsafeIndexRewrite(unittest.TestCase):
    def _check(self, fn, args):
        compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        torch.testing.assert_close(compiled(*args), fn(*args))

    def test_accepted_separable_outer(self):
        # THE rewritten form: (M,1) column + 1-D row -> outer grid (M, K).
        x, ih, iw = _mk(2, 8, 16, 16, 12, 9)
        self._check(lambda x, ih, iw: x[:, :, ih, iw], (x, ih, iw))

    def test_rejected_pairing_both_1d(self):
        # Both 1-D -> aten diagonal pairing (M,) — not chain-expressible,
        # must keep the fallback and stay exact.
        x, ih, iw = _mk(2, 8, 16, 16, 12, 12)
        self._check(lambda x, a, b: x[:, :, a, b], (x, ih.reshape(-1), iw))

    def test_rejected_single_column(self):
        # Run of one (M,1): aten keeps a trailing size-1 dim that the
        # chained form would drop — must fall back.
        x, ih, _ = _mk(2, 8, 16, 16, 12, 1)
        self._check(lambda x, c: x[:, :, c], (x, ih))

    def test_rejected_rank3(self):
        x, ih, _ = _mk(2, 8, 16, 16, 12, 1)
        idx3 = torch.arange(24, device="npu", dtype=torch.int64).reshape(2, 3, 4) % 16
        self._check(lambda x, i3: x[:, :, i3], (x, idx3))


if __name__ == "__main__":
    unittest.main()
