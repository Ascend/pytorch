# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU coverage for Tensor.cfloat."""

import torch

from torch_npu.testing.testcase import run_tests, TestCase


class TestTensorCfloat(TestCase):
    def test_type_conversion_via_dtype_name_cfloat(self):
        source_cases = (
            (torch.bool, [True, False, True]),
            (torch.uint8, [0, 1, 255]),
            (torch.int8, [-128, -1, 127]),
            (torch.int16, [-32768, -1, 32767]),
            (torch.int32, [-2, 0, 3]),
            (torch.int64, [-2, 0, 3]),
            (torch.float16, [-1.5, 0.0, 2.25]),
            (torch.bfloat16, [-1.5, 0.0, 2.25]),
            (torch.float32, [-1.5, 0.0, 2.25]),
            (torch.float64, [-1.5, 0.0, 2.25]),
            (torch.complex64, [1 + 2j, 0j, -3 - 4j]),
            (torch.complex128, [1 + 2j, 0j, -3 - 4j]),
        )

        for dtype, values in source_cases:
            with self.subTest(dtype=dtype):
                source_cpu = torch.tensor(values, dtype=dtype)
                source_npu = source_cpu.npu()
                actual = source_npu.cfloat()
                expected = source_cpu.cfloat()

                self.assertEqual(actual.dtype, torch.complex64)
                self.assertEqual(actual.device, source_npu.device)
                torch.testing.assert_close(actual.cpu(), expected)
                torch.testing.assert_close(actual.real.cpu(), expected.real)
                torch.testing.assert_close(actual.imag.cpu(), expected.imag)

        for dtype in (torch.int64, torch.float32, torch.complex64):
            with self.subTest(empty_dtype=dtype):
                source_cpu = torch.empty(0, dtype=dtype)
                source_npu = source_cpu.npu()
                actual = source_npu.cfloat()

                self.assertEqual(actual.dtype, torch.complex64)
                self.assertEqual(actual.device, source_npu.device)
                torch.testing.assert_close(actual.cpu(), source_cpu.cfloat())


if __name__ == "__main__":
    run_tests()
