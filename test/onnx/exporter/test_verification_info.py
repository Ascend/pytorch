# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.onnx.verification APIs on NPU:
1. Community tests lack constructor, error, boundary and NPU input coverage.
2. Validate torch.onnx.verification.VerificationInfo (extendable).
"""

import dataclasses
import math

import torch
from torch.onnx.verification import VerificationInfo
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestVerificationInfo(TestCase):
    def tensor(self, data, **kwargs):
        value = torch.tensor(data, device=device_type, **kwargs)
        self.assertEqual(value.device.type, "npu")
        return value

    def test_constructor_fields(self):
        histogram = (self.tensor([2.0]), self.tensor([0.0, 1.0]))
        fields = ("output", 0.25, 0.5, histogram, histogram,
                  torch.float16, torch.float32)
        names = [field.name for field in dataclasses.fields(VerificationInfo)]
        for info in (VerificationInfo(*fields),
                     VerificationInfo(**dict(zip(names, fields)))):
            for name, value in zip(names, fields):
                self.assertIs(getattr(info, name), value)
        self.assertEqual(VerificationInfo(*([None] * 7)).name, None)

    def test_constructor_required_arguments(self):
        names = [field.name for field in dataclasses.fields(VerificationInfo)]
        for name in names:
            with self.subTest(missing=name), self.assertRaises(TypeError):
                VerificationInfo(**{key: None for key in names if key != name})
        with self.assertRaises(TypeError):
            VerificationInfo(*([None] * 8))
        with self.assertRaises(TypeError):
            VerificationInfo(**dict.fromkeys(names), unknown=1)

    def test_equal_tensors(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32,
                      torch.int32, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                expected = self.tensor([1, 0, 1], dtype=dtype)
                actual = expected.clone()
                info = VerificationInfo.from_tensors("equal", expected, actual)
                self.assertEqual(info.name, "equal")
                self.assertEqual(info.max_abs_diff, 0.0)
                self.assertEqual(info.max_rel_diff, 0.0)
                self.assertEqual(info.expected_dtype, dtype)
                self.assertEqual(info.actual_dtype, dtype)
                for hist in (info.abs_diff_hist, info.rel_diff_hist):
                    self.assertEqual(hist[0].tolist(), [3.0] + [0.0] * 8)
                    self.assertEqual(hist[0].device.type, "cpu")
                    self.assertEqual(hist[1].numel(), 10)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.device.type, "npu")

    def test_nonzero_error_and_bins(self):
        expected = self.tensor([1.0, 2.0, 4.0], requires_grad=True)
        actual = self.tensor([1.0, 2.5, 3.0], requires_grad=True)
        info = VerificationInfo.from_tensors(
            name="error", expected=expected, actual=actual)
        self.assertEqual(info.max_abs_diff, 1.0)
        self.assertAlmostEqual(info.max_rel_diff, 0.25, places=6)
        self.assertEqual(info.abs_diff_hist[0].tolist(),
                         [1.0, 0, 0, 0, 0, 0, 1, 1, 0])
        self.assertEqual(info.rel_diff_hist[0].tolist(),
                         [1.0, 0, 0, 0, 0, 0, 2, 0, 0])
        self.assertEqual(info.abs_diff_hist[1].tolist(),
                         info.rel_diff_hist[1].tolist())
        self.assertEqual(info.abs_diff_hist[1][0].item(), 0.0)
        self.assertEqual(info.abs_diff_hist[1][-1].item(), 1000000.0)
        self.assertIsNone(expected.grad)
        self.assertIsNone(actual.grad)
        self.assertTrue(expected.requires_grad)

    def test_scalar_and_tensor_input_types(self):
        for scalar in (True, 1, 1.0):
            tensor = self.tensor(scalar)
            for expected, actual in ((tensor, scalar), (scalar, tensor),
                                     (scalar, scalar)):
                with self.subTest(kind=type(scalar).__name__):
                    info = VerificationInfo.from_tensors("", expected, actual)
                    self.assertEqual(info.max_abs_diff, 0)
                    self.assertEqual(info.max_rel_diff, 0)
                    self.assertEqual(info.expected_dtype, tensor.dtype)
                    self.assertEqual(info.actual_dtype, tensor.dtype)
                    self.assertEqual(info.abs_diff_hist[0].sum().item(), 1)

    def test_empty_tensors(self):
        empty = self.tensor([])
        nonempty = self.tensor([1.0])
        for expected, actual in ((empty, empty), (empty, nonempty),
                                 (nonempty, empty)):
            info = VerificationInfo.from_tensors("empty", expected, actual)
            self.assertEqual(info.max_abs_diff, math.inf)
            self.assertEqual(info.max_rel_diff, math.inf)
            self.assertEqual(info.abs_diff_hist[0].sum().item(), 0)
            self.assertEqual(info.rel_diff_hist[0].sum().item(), 0)

    def test_noncontiguous_and_broadcast(self):
        expected = self.tensor([[1.0, 2.0], [3.0, 4.0]]).t()
        self.assertFalse(expected.is_contiguous())
        info = VerificationInfo.from_tensors(None, expected, expected.clone())
        self.assertIsNone(info.name)
        self.assertEqual(info.max_abs_diff, 0)
        info = VerificationInfo.from_tensors("broadcast", self.tensor([1., 2.]),
                                             self.tensor(1.))
        self.assertEqual(info.max_abs_diff, 1)
        self.assertAlmostEqual(info.max_rel_diff, 0.5, places=6)

    def test_zero_normalizer_and_mixed_dtype(self):
        info = VerificationInfo.from_tensors("zero", self.tensor([0.0]),
                                             self.tensor([1.0]))
        self.assertEqual(info.max_abs_diff, 1)
        self.assertAlmostEqual(info.max_rel_diff / 1e7, 1.0, places=6)
        self.assertEqual(info.rel_diff_hist[0].sum().item(), 0)
        info = VerificationInfo.from_tensors("mixed", self.tensor([1]),
                                             self.tensor([1.5]))
        self.assertEqual(info.expected_dtype, torch.int64)
        self.assertEqual(info.actual_dtype, torch.float32)
        self.assertEqual(info.max_abs_diff, 0.5)

    def test_boolean_mismatch(self):
        info = VerificationInfo.from_tensors("bool", self.tensor([True, False]),
                                             self.tensor([False, True]))
        self.assertEqual(info.max_abs_diff, 1)
        self.assertAlmostEqual(info.max_rel_diff / 1e7, 1.0, places=6)
        self.assertEqual(info.expected_dtype, torch.bool)

    def test_nonfinite_values(self):
        for value in (math.nan, math.inf, -math.inf):
            expected = self.tensor([value])
            info = VerificationInfo.from_tensors("nonfinite", expected, expected)
            self.assertTrue(math.isnan(info.max_abs_diff))
            self.assertTrue(math.isnan(info.max_rel_diff))
            self.assertEqual(info.abs_diff_hist[0].sum().item(), 0)

    def test_invalid_inputs(self):
        value = self.tensor([1.0])
        for invalid in (None, "invalid", {"value": 1}):
            for expected, actual in ((invalid, value), (value, invalid)):
                with self.subTest(invalid=invalid), self.assertRaises(
                        (TypeError, ValueError, RuntimeError)):
                    VerificationInfo.from_tensors("invalid", expected, actual)
        with self.assertRaises(RuntimeError):
            VerificationInfo.from_tensors("shape", self.tensor([1., 2.]),
                                          self.tensor([1., 2., 3.]))
        for args in ((), ("missing",), ("missing", value)):
            with self.assertRaises(TypeError):
                VerificationInfo.from_tensors(*args)


if __name__ == "__main__":
    run_tests()
