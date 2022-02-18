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
import torch
import torch_npu
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import Dtypes, instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class Test_Bitwise_Not(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_bool_data(self, shape):
        input1 = np.random.randint(0, 2, shape).astype(np.bool_)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        output = torch.bitwise_not(input1)
        if output.dtype not in [torch.int32, torch.int8, torch.bool]: 
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.bitwise_not(input1)
        output = output.to("cpu")
        if output.dtype not in [torch.int32, torch.int8, torch.bool]: 
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        torch.bitwise_not(input1, out = input2)
        output = input2.to("cpu")
        if output.dtype not in [torch.int32, torch.int8, torch.bool]: 
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def test_bitwise_not_bool(self, device):
        npu_input1 = self.generate_bool_data((2, 3))
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_not_int16(self, device):
        npu_input1 = self.generate_data(0, 2342, (2, 3), np.int16)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_not_int32(self, device):
        npu_input1 = self.generate_data(0, 34222, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_bitwise_not_int64(self, device):
        npu_input1 = self.generate_data(0, 355553, (2, 3), np.int64)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_not_out(self, device):
        shape_format = [
            [[0, 2342, [2, 3], np.int16], [0, 2342, [10, 20], np.int16]],
            [[0, 34222, [2, 3], np.int32], [0, 34222, [10, 20], np.int32]],
            [[0, 355553, [2, 3], np.int64], [0, 355553, [1, 1], np.int64]],
            ]
        for item in shape_format:
            npu_input1 = self.generate_data(item[0][0], item[0][1], item[0][2], item[0][3])
            npu_input2 = self.generate_data(item[1][0], item[1][1], item[1][2], item[1][3])
            cpu_output = self.cpu_op_exec(npu_input1)
            npu_output1 = self.npu_op_exec_out(npu_input1, npu_input1)
            npu_output2 = self.npu_op_exec_out(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output1)

instantiate_device_type_tests(Test_Bitwise_Not, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
