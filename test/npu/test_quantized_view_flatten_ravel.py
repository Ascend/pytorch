"""Regressions for aten::view / aten::as_strided on QuantizedPrivateUse1 (Quantized NPU).

Mirrors the quantized-tensor portions of test/test_view_ops.py (test_ravel / test_flatten)
on device=npu only. All comparisons stay on NPU; no QTensor.cpu() or CPU cross-device
int_repr checks (QuantizedPrivateUse1 lacks empty_quantized for migration, and separate
CPU/NPU allocations do not share NPUStorageDesc semantics).
"""
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestQuantizedViewFlattenRavel(TestCase):
    def _qtensor(self, sizes):
        return torch._empty_affine_quantized(
            sizes,
            scale=2,
            zero_point=3,
            dtype=torch.quint8,
            device="npu",
        )

    def _test_ravel(self, tensors, size, nc=False):
        """Same checks as TestOldViewOps._test_ravel for quantized NPU tensors."""
        for src in tensors:
            flat = src.ravel()
            self.assertEqual(flat.shape, torch.Size([size]))
            self.assertEqual(src.view(-1), flat)
            # Quantized NPU may materialize ravel/view when NPUStorageDesc does not match.
            if not (src.is_quantized and src.is_npu):
                self.assertIs(flat._base, src)
            self.assertTrue(flat.is_contiguous())

            if nc:
                nc_src = src.t()
                nc_flat = nc_src.ravel()
                self.assertEqual(nc_flat.shape, torch.Size([size]))
                self.assertEqual(nc_src.contiguous().view(-1), nc_flat)
                self.assertIsNot(nc_flat._base, src)
                self.assertTrue(nc_flat.is_contiguous())

    def _test_flatten(self, tensors):
        """Same checks as TestOldViewOps.test_flatten for quantized NPU tensors."""
        for src in tensors:
            flat = src.flatten(0, -1)
            self.assertEqual(flat.shape, torch.Size([625]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 2)
            self.assertEqual(flat.shape, torch.Size([125, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 1)
            self.assertEqual(flat.shape, torch.Size([25, 5, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(1, 2)
            self.assertEqual(flat.shape, torch.Size([5, 25, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 3)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(-2, -1)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 2)
            self.assertEqual(flat, src)

    def test_quantized_ravel_on_npu(self):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available")

        self._test_ravel([self._qtensor([5, 5, 5, 5])], 625)
        self._test_ravel([
            self._qtensor([0, 2, 3]),
            self._qtensor([3, 0, 2]),
        ], 0)
        self._test_ravel([self._qtensor([5, 5])], 25, nc=True)

    def test_quantized_flatten_on_npu(self):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available")

        self._test_flatten([self._qtensor([5, 5, 5, 5])])


if __name__ == "__main__":
    run_tests()
