# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Welford var_mean axis handling: global (dim=None) and 1D (dim=0)."""

import torch
import torch_npu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestWelfordVarMeanAxis(TestCase):
    def _compile(self, fn):
        return torch.compile(
            fn,
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )

    def test_global_var_mean(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        x = torch.randn(32, 64, device="npu")
        expected_var, expected_mean = torch.var_mean(x)
        actual_var, actual_mean = self._compile(lambda t: torch.var_mean(t))(x)
        torch.testing.assert_close(actual_var, expected_var, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-5, atol=1e-6)

    def test_1d_dim0_var_mean(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        x = torch.randn(16384, device="npu")
        expected_var, expected_mean = torch.var_mean(x, dim=0)
        actual_var, actual_mean = self._compile(lambda t: torch.var_mean(t, dim=0))(x)
        torch.testing.assert_close(actual_var, expected_var, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-5, atol=1e-6)

    def test_large_magnitude_mean_does_not_overflow(self):
        """The welford mean must not form an unnormalized partial sum.

        The NPU combine lowers to sum(mean_i * w_i) / sum(w_i). That
        intermediate is the raw total of the reduction axis (rnumel *
        |mean|), so FP32 data whose *sum* exceeds FLT_MAX overflows to +inf
        even though every lane mean and the final mean are representable:
        65536 elements of 1e34 sum to 6.55e38 > 3.40e38. Normalizing each
        lane's weight first (w_i / sum(w)) keeps every partial sum bounded
        by max|mean_i|, so the mean stays 1e34 and finite.
        """
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        x = torch.full((8, 65536), 1e34, device="npu", dtype=torch.float32)
        var, mean = self._compile(lambda t: torch.var_mean(t, dim=1))(x)
        self.assertTrue(
            torch.isfinite(mean).all().item(),
            f"mean overflowed: {mean.flatten()[:4].tolist()}",
        )
        torch.testing.assert_close(
            mean, torch.full_like(mean, 1e34), rtol=1e-5, atol=0.0
        )
        # All samples are identical, so the variance is exactly 0 up to the
        # mean's own rounding; it must not inherit a NaN/inf from the mean.
        self.assertTrue(torch.isfinite(var).all().item())
        self.assertTrue((var >= 0).all().item())


if __name__ == "__main__":
    run_tests()
