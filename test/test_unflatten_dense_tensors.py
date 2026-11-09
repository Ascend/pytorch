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
Add validation cases for torch._utils APIs on NPU:
1. Extend direct Python API coverage beyond community C++ unflatten tests.
2. This file validates torch._utils._flatten_dense_tensors and
   torch._utils._unflatten_dense_tensors (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFlattenDenseTensors(TestCase):

    def _assert_matches_cpu(self, tensors):
        device_tensors = type(tensors)(tensor.to(device_type) for tensor in tensors)
        expected = torch._utils._flatten_dense_tensors(tensors)
        actual = torch._utils._flatten_dense_tensors(device_tensors)

        self.assertEqual(actual.device.type, device_type)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.dim(), 1)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.cpu(), expected)
        return actual

    def test_list_and_tuple_input(self):
        for container in (list, tuple):
            with self.subTest(container=container.__name__):
                tensors = container([
                    torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    torch.tensor([-1.0, 2.0], dtype=torch.float32),
                ])
                result = self._assert_matches_cpu(tensors)

                self.assertEqual(result.shape, torch.Size([8]))

    def test_single_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        result = self._assert_matches_cpu([tensor])

        self.assertEqual(result.shape, torch.Size([6]))

    def test_non_contiguous_tensor(self):
        cpu_base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        device_base = cpu_base.to(device_type)
        cpu_tensor = cpu_base[:, ::2]
        device_tensor = device_base[:, ::2]

        self.assertFalse(cpu_tensor.is_contiguous())
        self.assertFalse(device_tensor.is_contiguous())
        expected = torch._utils._flatten_dense_tensors([cpu_tensor])
        actual = torch._utils._flatten_dense_tensors([device_tensor])
        self.assertEqual(actual.cpu(), expected)

    def test_empty_tensors(self):
        test_cases = (
            [torch.empty(0), torch.arange(3, dtype=torch.float32)],
            [torch.empty(0, 2), torch.empty(0)],
        )
        for tensors in test_cases:
            with self.subTest(numel=[tensor.numel() for tensor in tensors]):
                result = self._assert_matches_cpu(tensors)
                self.assertEqual(result.numel(), sum(tensor.numel() for tensor in tensors))

    def test_same_dtype(self):
        for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16,
                      torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                tensors = [
                    torch.tensor([0, 1, 2, 3], dtype=dtype),
                    torch.ones(2, dtype=dtype),
                ]
                result = self._assert_matches_cpu(tensors)

                self.assertEqual(result.dtype, dtype)

    def test_mixed_dtype_promotion(self):
        tensors = [
            torch.tensor([1.0, 2.0], dtype=torch.float16),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ]
        result = self._assert_matches_cpu(tensors)

        self.assertEqual(result.dtype, torch.float32)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            torch._utils._flatten_dense_tensors([])

    def test_scalar_and_keyword_input(self):
        tensors = [torch.tensor(2.0).to(device_type), torch.tensor(-3.0).to(device_type)]
        result = torch._utils._flatten_dense_tensors(tensors=tensors)

        self.assertEqual(result, torch.tensor([2.0, -3.0]).to(device_type))
        self.assertEqual(result.device.type, device_type)
        self.assertTrue(result.is_contiguous())

    def test_invalid_input(self):
        for tensors in (None, 1, "invalid", [None], [1]):
            with self.subTest(tensors=tensors):
                with self.assertRaises(TypeError):
                    torch._utils._flatten_dense_tensors(tensors)

    def test_required_argument(self):
        with self.assertRaises(TypeError):
            torch._utils._flatten_dense_tensors()

    def test_generator_raises(self):
        tensors = [torch.ones(2).to(device_type), torch.zeros(3).to(device_type)]

        with self.assertRaises(TypeError):
            torch._utils._flatten_dense_tensors(tensor for tensor in tensors)

    def test_mixed_devices_raise(self):
        cpu_tensor = torch.ones(2)
        device_tensor = torch.zeros(3).to(device_type)

        with self.assertRaises(RuntimeError):
            torch._utils._flatten_dense_tensors([cpu_tensor, device_tensor])


class TestUnflattenDenseTensors(TestCase):
    """Test cases for torch._utils._unflatten_dense_tensors."""

    def setUp(self):
        super().setUp()
        acc = torch.accelerator.current_accelerator()
        self.device = acc.type if acc else "cpu"

    def _to_device(self, t):
        return t.to(self.device)

    def test_round_trip_basic(self):
        # Round-trip: flatten then unflatten should recover original tensors
        t1 = self._to_device(torch.ones(4, 4))
        t2 = self._to_device(torch.zeros(2, 3))
        t3 = self._to_device(torch.randn(5))
        tensors = [t1, t2, t3]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), len(tensors))
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_single_tensor(self):
        t = self._to_device(torch.randn(3, 5, 2))
        flat = torch._utils._flatten_dense_tensors([t])
        result = torch._utils._unflatten_dense_tensors(flat, [t])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, t.shape)
        self.assertEqual(result[0], t)

    def test_multiple_tensors_different_sizes(self):
        sizes = [(1,), (2, 3), (4, 5, 6), (7, 8)]
        tensors = [self._to_device(torch.randn(*s)) for s in sizes]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), len(tensors))
        for i, (r, t) in enumerate(zip(result, tensors)):
            self.assertEqual(r.shape, t.shape, f"tensor {i} shape mismatch")
            self.assertEqual(r, t, f"tensor {i} value mismatch")

    def test_empty_tensor_in_list(self):
        t1 = self._to_device(torch.ones(3, 2))
        t2 = self._to_device(torch.tensor([]))
        t3 = self._to_device(torch.randn(4))
        tensors = [t1, t2, t3]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].numel(), 6)
        self.assertEqual(result[1].numel(), 0)
        self.assertEqual(result[2].numel(), 4)
        self.assertEqual(result[0], t1)
        # Verify empty tensor preserves shape, dtype and device
        self.assertEqual(result[1].shape, t2.shape)
        self.assertEqual(result[1].dtype, t2.dtype)
        self.assertEqual(result[1], t2)
        self.assertEqual(result[2], t3)

    def test_all_empty_tensors(self):
        tensors = [
            self._to_device(torch.tensor([])),
            self._to_device(torch.tensor([])),
        ]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 2)
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_different_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.int32]:
            with self.subTest(dtype=dtype):
                t1 = self._to_device(torch.ones(2, 3, dtype=dtype))
                t2 = self._to_device(torch.zeros(4, dtype=dtype))

                flat = torch._utils._flatten_dense_tensors([t1, t2])
                result = torch._utils._unflatten_dense_tensors(flat, [t1, t2])

                self.assertEqual(result[0].dtype, dtype)
                self.assertEqual(result[1].dtype, dtype)
                self.assertEqual(result[0], t1)
                self.assertEqual(result[1], t2)

    def test_large_num_tensors(self):
        n = 50
        tensors = [self._to_device(torch.randn(i + 1)) for i in range(n)]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), n)
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_tuple_input(self):
        t1 = self._to_device(torch.ones(3, 3))
        t2 = self._to_device(torch.zeros(2))
        tensors = (t1, t2)  # tuple, not list

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 2)
        for r, t in zip(result, tensors):
            self.assertEqual(r, t)


if __name__ == "__main__":
    run_tests()
