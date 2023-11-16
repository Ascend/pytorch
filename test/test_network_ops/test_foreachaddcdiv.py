# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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


import sys
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddCMul(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def cpu_op_exec(self, input1, input2, input3, scalar):
        output = torch._foreach_addcdiv(input1, input2, input3, value=scalar)
        return output

    def npu_op_exec(self, input1, input2, input3, scalar):
        output = torch._foreach_addcdiv(input1, input2, input3, value=scalar)
        return output

    def cpu_op_exec_inplace(self, input1, input2, input3, scalar):
        torch._foreach_addcdiv(input1, input2, input3, value=scalar)
        return input1

    def npu_op_exec_inplace(self, input1, input2, input3, scalar):
        torch._foreach_addcdiv(input1, input2, input3, value=scalar)
        return input1

    def test_addcmul_3_3_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1, input2], [input1, input2], [input2, input3], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_10_10_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1, input2], [input1, input2], [input2, input3], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_3_3_float32_inplace(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1, input2], [input1, input2], [input2, input3], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_10_10_float32_inplace(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1, input2], [input1, input2], [input2, input3], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_3_3_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1_cpu, input2_cpu], [input1_cpu, input2_cpu], [input2_cpu, input3_cpu], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.to(torch.float16).numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_10_10_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1_cpu, input2_cpu], [input1_cpu, input2_cpu], [input2_cpu, input3_cpu], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.to(torch.float16).numpy(), npu_tmp1.to("cpu").numpy())

    def test_addcmul_10_23_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 23), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        cpu_output = self.cpu_op_exec([input1, input2], [input1, input2], [input2, input3], 0.5)
        npu_output = self.npu_op_exec([input1_npu, input2_npu], [input1_npu, input2_npu], [input2_npu, input3_npu], 0.5)
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())


if __name__ == "__main__":
    if torch_npu.npu.get_device_name(0)[:10] == 'Ascend910B':
        run_tests()
