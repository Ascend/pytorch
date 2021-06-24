# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import random
import torch
import numpy as np
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestRemainder(TestCase):

    def generate_two_tensor(self, min_d, max_d, shape, dtype):
        dividend = np.random.uniform(min_d, max_d, shape).astype(dtype)
        divisor = np.random.uniform(min_d, max_d, shape).astype(dtype)

        npu_dividend = torch.from_numpy(dividend)
        npu_divisor = torch.from_numpy(divisor)

        return npu_dividend, npu_divisor
    
    def generate_single_tensor(self, min_d, max_d, shape, dtype):
        dividend = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_dividend = torch.from_numpy(dividend)
        return npu_dividend
    
    def generate_fp_scalar(self, min_d, max_d):
        scalar = random.uniform(min_d, max_d)
        return scalar
    
    # While operatoring on AICPU, it seems that we do not have to care whether the divisor is scalar or not.
    def cpu_op_exec(self, dividend, divisor):
        output = torch.remainder(dividend, divisor)
        output = output.numpy()
        return output

    def npu_op_exec_both_tensor(self, dividend, divisor):
        output = torch.remainder(dividend, divisor)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_only_dividend_tensor(self, dividend, divisor):
        dividend = dividend.to("npu")
        output = torch.remainder(dividend, divisor)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_remainder_float32_both_tensor(self, device):
        npu_dividend, npu_divisor = self.generate_two_tensor(-100, 100, (5), np.float32)
        cpu_output = self.cpu_op_exec(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_both_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_float32_only_dividend_tensor(self, device):
        npu_dividend = self.generate_single_tensor(-100, 100, (5), np.float32)
        npu_divisor = self.generate_fp_scalar(-10, 10)
        cpu_output = self.cpu_op_exec(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_only_dividend_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_int32_both_tensor(self, device):
        npu_dividend, npu_divisor = self.generate_two_tensor(-100, 100, (5), np.int32)
        cpu_output = self.cpu_op_exec(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_both_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_int32_only_dividend_tensor(self, device):
        npu_dividend = self.generate_single_tensor(-100, 100, (5), np.int32)
        npu_divisor = self.generate_fp_scalar(-10, 10)
        cpu_output = self.cpu_op_exec(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_only_dividend_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output)

    # Because of the limitation of accracy, testcases using fp16 may not pass at the moment.
    def test_remainder_float16_both_tensor(self, device):
        def cpu_op_exec_fp16(dividend, divisor):
            dividend = dividend.to(torch.float32)
            divisor = divisor.to(torch.float32)
            output = torch.remainder(dividend, divisor)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        npu_dividend, npu_divisor = self.generate_two_tensor(-100, 100, (5), np.float16)
        cpu_output = cpu_op_exec_fp16(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_both_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_remainder_float16_only_dividend_tensor(self, device):
        def cpu_op_exec_fp16(dividend, divisor):
            dividend = dividend.to(torch.float32)
            output = torch.remainder(dividend, divisor)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        npu_dividend = self.generate_single_tensor(-100, 100, (5), np.float16)
        npu_divisor = self.generate_fp_scalar(-10, 10)
        cpu_output = cpu_op_exec_fp16(npu_dividend, npu_divisor)
        npu_output = self.npu_op_exec_only_dividend_tensor(npu_dividend, npu_divisor)
        self.assertRtolEqual(cpu_output, npu_output) 

instantiate_device_type_tests(TestRemainder, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()