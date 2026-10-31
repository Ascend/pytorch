# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
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
Add validation cases for tensor creation APIs on NPU:
1. Community tests lack direct coverage of some boundaries and backward results.
2. This file validates torch.asarray (extendable).
"""

from array import array
from itertools import product

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTensorCreationConsistency(TestCase):
    def assert_result(self, result, expected, dtype):
        self.assertEqual(result.device.type, device_type)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.cpu().tolist(), expected)

    def test_asarray_tensor_copy_and_optional_arguments(self):
        x = torch.tensor([1., 2., 3.], device=device_type)
        self.assertIs(torch.asarray(x), x)
        for dtype, device, copy in product(
                (None, x.dtype), (None, str(x.device), x.device, 0),
                (None, False, True)):
            with self.subTest(dtype=dtype, device=device, copy=copy):
                y = torch.asarray(obj=x, dtype=dtype, device=device, copy=copy,
                                  requires_grad=False)
                self.assert_result(y, [1., 2., 3.], x.dtype)
                self.assertEqual(y.data_ptr() == x.data_ptr(), copy is not True)
                self.assertFalse(y.requires_grad)
        for copy in (None, True):
            y = torch.asarray(x, device=device_type, copy=copy)
            self.assert_result(y, [1., 2., 3.], x.dtype)
            self.assertEqual(y.data_ptr() == x.data_ptr(), copy is None)
        alias = torch.asarray(x, copy=False)
        copied = torch.asarray(x, copy=True)
        x[0] = 9
        self.assertEqual(alias[0].item(), 9)
        self.assertEqual(copied[0].item(), 1)

    def test_asarray_dtype_conversion(self):
        dtypes = (torch.bfloat16, torch.float16, torch.float32, torch.float64,
                  torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.bool)
        for src_dtype, dst_dtype in product(dtypes, dtypes):
            with self.subTest(src_dtype=src_dtype, dst_dtype=dst_dtype):
                x = torch.tensor([0, 1], dtype=src_dtype, device=device_type)
                for copy in (None, True):
                    y = torch.asarray(x, dtype=dst_dtype, copy=copy)
                    self.assert_result(y, [0, 1], dst_dtype)
                    self.assertEqual(y.data_ptr() == x.data_ptr(),
                                     src_dtype == dst_dtype and copy is None)
                if src_dtype != dst_dtype:
                    with self.assertRaisesRegex(ValueError, "can't alias"):
                        torch.asarray(x, dtype=dst_dtype, copy=False)

    def test_asarray_scalars_sequences_and_default_device(self):
        cases = ((True, True, torch.bool), (2, 2, torch.int64),
                 (1.5, 1.5, torch.float32),
                 ([1, 2], [1, 2], torch.int64),
                 ((1., 2.), [1., 2.], torch.float32),
                 (range(3), [0, 1, 2], torch.int64),
                 ([[1, 2], [3, 4]], [[1, 2], [3, 4]], torch.int64),
                 ([True, 2, 3.5], [1., 2., 3.5], torch.float32))
        for obj, expected, dtype in cases:
            with self.subTest(obj=obj):
                for kwargs in ({}, {"dtype": None, "device": None, "copy": None},
                               {"device": device_type, "copy": True}):
                    with torch.device(device_type):
                        self.assert_result(torch.asarray(obj, **kwargs), expected, dtype)
                with self.assertRaisesRegex(ValueError, "can't alias"):
                    torch.asarray(obj, device=device_type, copy=False)

    def test_asarray_empty_and_noncontiguous(self):
        for obj in ([], (), [[], []]):
            y = torch.asarray(obj, device=device_type)
            self.assert_result(y, list(obj), torch.get_default_dtype())
        for shape in ((), (0,), (2, 0, 3)):
            x = torch.ones(shape, device=device_type)
            for copy in (None, False, True):
                y = torch.asarray(x, copy=copy)
                self.assertEqual(y.device, x.device)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y, x)
        x = torch.arange(12, dtype=torch.float32, device=device_type).reshape(3, 4).t()
        for copy in (None, False, True):
            y = torch.asarray(x, copy=copy)
            self.assertEqual(y.device, x.device)
            self.assertEqual(y, x)
            self.assertEqual(y.data_ptr() == x.data_ptr(), copy is not True)
            if copy is not True:
                self.assertEqual(y.stride(), x.stride())

    def test_asarray_host_inputs(self):
        # NumPy and buffer objects reside on the host by definition.
        data = np.array([1., 2., 3.], dtype=np.float32)
        for obj in (data, memoryview(data), array("f", [1., 2., 3.])):
            for copy in (None, True):
                y = torch.asarray(obj, dtype=torch.float32, device=device_type, copy=copy)
                self.assert_result(y, [1., 2., 3.], torch.float32)
                y[0] = 7
                self.assertEqual(obj[0], 1.)
            with self.assertRaisesRegex(ValueError, "can't alias"):
                torch.asarray(obj, dtype=torch.float32, device=device_type, copy=False)
        for obj in (np.float32(1.5), np.array(1.5, dtype=np.float32)):
            y = torch.asarray(obj, device=device_type)
            self.assert_result(y, 1.5, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "can't alias NumPy scalars"):
            torch.asarray(np.float32(1.5), device=device_type, copy=False)

    def test_asarray_device_transfer(self):
        x = torch.tensor([1., 2.], device=device_type)
        # CPU tensors are required to verify both directions of device transfer.
        for copy in (None, True):
            host = torch.asarray(x, device="cpu", copy=copy)
            self.assertEqual(host.device.type, "cpu")
            self.assertEqual(host.tolist(), [1., 2.])
            y = torch.asarray(host, device=x.device, copy=copy)
            self.assert_result(y, [1., 2.], torch.float32)
        with self.assertRaisesRegex(ValueError, "can't alias"):
            torch.asarray(x, device="cpu", copy=False)

    def test_asarray_nonfinite_values(self):
        x = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.],
                         device=device_type)
        for copy in (None, False, True):
            y = torch.asarray(x, copy=copy)
            self.assertEqual(y.device, x.device)
            self.assertEqual(y.dtype, x.dtype)
            self.assertEqual(y, x, equal_nan=True)

    def test_asarray_backward_and_detach(self):
        for copy in (None, False, True):
            for requires_grad in (False, True):
                x = torch.tensor([1., 2.], device=device_type, requires_grad=True)
                intermediate = x * 3
                y = torch.asarray(intermediate, copy=copy, requires_grad=requires_grad)
                self.assert_result(y, [3., 6.], torch.float32)
                self.assertEqual(y.requires_grad, requires_grad)
                self.assertEqual(y.grad_fn is not None, requires_grad)
                if requires_grad:
                    y.sum().backward()
                    self.assert_result(x.grad, [3., 3.], torch.float32)
                else:
                    self.assertIsNone(x.grad)
        for copy in (None, True):
            y = torch.asarray([1., 2.], device=device_type, copy=copy, requires_grad=True)
            self.assertTrue(y.is_leaf)
            y.sum().backward()
            self.assert_result(y.grad, [1., 1.], torch.float32)

    def test_asarray_requires_grad_default(self):
        for kwargs in ({}, {"requires_grad": None}):
            x = torch.tensor([1., 2.], device=device_type, requires_grad=True)
            y = torch.asarray(x * 2, **kwargs)
            self.assertTrue(y.requires_grad)
            y.sum().backward()
            self.assert_result(x.grad, [2., 2.], torch.float32)

    def test_asarray_invalid_arguments(self):
        x = torch.tensor([1., 2.], device=device_type)
        for kwargs in ({"dtype": "float32"}, {"device": []}, {"copy": "yes"},
                       {"requires_grad": "yes"}, {"unknown": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                torch.asarray(x, **kwargs)
        for obj in (None, {}, "abc", object()):
            with self.subTest(obj=type(obj)), self.assertRaises((TypeError, ValueError, RuntimeError)):
                torch.asarray(obj, device=device_type)
        with self.assertRaises(ValueError):
            torch.asarray([[1], [2, 3]], device=device_type)
        with self.assertRaises(RuntimeError):
            torch.asarray([1, 2], device=device_type, requires_grad=True)
        with self.assertRaises(TypeError):
            torch.asarray(x, torch.float32)


if __name__ == "__main__":
    run_tests()
