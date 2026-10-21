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

"""Validate torch.UntypedStorage.double on NPU.

PyTorch community cases lack direct raw-byte, precision, empty-storage,
and error coverage for this API. Related untyped Storage APIs can be added.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUntypedStorageDouble(TestCase):
    cases = (
        (torch.bool, [False, True]),
        (torch.uint8, [0, 1, 127, 255]),
        (torch.int8, [-128, -1, 0, 127]),
        (torch.int16, [-32768, -1, 0, 32767]),
        (torch.int32, [-16777217, 0, 16777217, 2147483647]),
        (torch.int64, [-(2**53), 0, 2**53, 2**53 + 1]),
        (torch.float16, [-2.5, 0.0, 1.25]),
        (torch.bfloat16, [-2.5, 0.0, 1.25]),
        (torch.float32, [1.0000001192092896, -0.0, float("inf")]),
        (torch.float64, [1 + 2**-52, 1e300, -0.0]),
        (torch.complex64, [1 + 2j, -3 + 0.5j]),
        (torch.complex128, [complex(1 + 2**-52, 2), complex(1e300, -0.5)]),
    )

    def _check_conversion(self, source):
        raw_before = source.cpu().tolist()
        expected_storage = source.cpu().double()
        result = source.double()
        self.assertIsInstance(result, torch.TypedStorage)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.size(), source.nbytes())
        self.assertEqual(result.nbytes(), source.nbytes() * 8)
        expected = torch.empty(0, dtype=torch.float64).set_(expected_storage)
        actual = torch.empty(0, dtype=torch.float64).set_(result.cpu())
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(source.cpu().tolist(), raw_before)
        if source.nbytes():
            self.assertNotEqual(result.data_ptr(), source.data_ptr())

    def test_untyped_storage_double(self):
        for dtype, values in self.cases:
            with self.subTest(dtype=dtype):
                self._check_conversion(
                    torch.tensor(values, dtype=dtype, device=device_type).untyped_storage()
                )

    def test_empty_storage(self):
        for dtype, _ in self.cases:
            with self.subTest(dtype=dtype):
                source = torch.empty(0, dtype=dtype, device=device_type).untyped_storage()
                self._check_conversion(source)

    def test_invalid_arguments(self):
        source = torch.empty(0, dtype=torch.uint8, device=device_type).untyped_storage()
        for value in (None, torch.float64, 1, [], {}, "npu"):
            with self.subTest(positional=value):
                with self.assertRaises(TypeError):
                    source.double(value)
        for keyword in ("dtype", "device", "non_blocking"):
            with self.subTest(keyword=keyword):
                with self.assertRaises(TypeError):
                    source.double(**{keyword: None})

    def test_cpu_path(self):
        for dtype, values in self.cases:
            with self.subTest(dtype=dtype):
                self._check_conversion(torch.tensor(values, dtype=dtype).untyped_storage())


if __name__ == "__main__":
    run_tests()
