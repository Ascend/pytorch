# Copyright (c) 2020, Huawei Technologies.
#  
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

import sys
import copy
import numpy as np
import torch
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import graph_mode

class TestScalarCopy(TestCase):
    def op_exec_char(self, input_cpu, input_npu):
        output_cpu = torch.add(input_cpu, 2).char()
        output_npu = torch.add(input_npu, 2).char()
        output_cpu = output_cpu.numpy()
        output_npu_cpu = output_npu.to("cpu")
        output_npu_cpu = output_npu_cpu.numpy()
        return output_cpu, output_npu_cpu

    def op_exec_byte(self, input_cpu, input_npu):
        output_cpu = torch.add(input_cpu, 2).byte()
        output_npu = torch.add(input_npu, 2).byte()
        output_cpu = output_cpu.numpy()
        output_npu_cpu = output_npu.to("cpu")
        output_npu_cpu = output_npu_cpu.numpy()
        return output_cpu, output_npu_cpu

    def op_exec_bool(self, input_cpu, input_npu):
        output_cpu = torch.add(input_cpu, 2).bool()
        output_npu = torch.add(input_npu, 2).bool()
        output_cpu = output_cpu.numpy()
        output_npu_cpu = output_npu.to("cpu")
        output_npu_cpu = output_npu_cpu.numpy()
        return output_cpu, output_npu_cpu

    def op_exec_float(self, input_cpu, input_npu):
        output_cpu = torch.add(input_cpu, 2).float()
        output_npu = torch.add(input_npu, 2).float()
        output_cpu = output_cpu.numpy()
        output_npu_cpu = output_npu.to("cpu")
        output_npu_cpu = output_npu_cpu.numpy()
        return output_cpu, output_npu_cpu

    def op_exec_int(self, input_cpu, input_npu):
        output_cpu = torch.add(input_cpu, 2).int()
        output_npu = torch.add(input_npu, 2).int()
        output_cpu = output_cpu.numpy()
        output_npu_cpu = output_npu.to("cpu")
        output_npu_cpu = output_npu_cpu.numpy()
        return output_cpu, output_npu_cpu
    
    @graph_mode
    def test_scalar_copy(self, device):
        shape_format = [
                [[np.float32, -1, 1], 1, 10],
                [[np.float32, -1, (4, 3)], 1, 10],
                [[np.float32, -1, (20, 13)], 1, 10]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            char_output_cpu, char_output_npu = self.op_exec_char(cpu_input, npu_input)
            self.assertRtolEqual(char_output_cpu, char_output_npu)

            byte_output_cpu, byte_output_npu = self.op_exec_byte(cpu_input, npu_input)
            self.assertRtolEqual(byte_output_cpu, byte_output_npu)

            bool_output_cpu, bool_output_npu = self.op_exec_bool(cpu_input, npu_input)
            self.assertRtolEqual(bool_output_cpu, bool_output_npu)

            float_output_cpu, float_output_npu = self.op_exec_float(cpu_input, npu_input)
            self.assertRtolEqual(float_output_cpu, float_output_npu)

            int_output_cpu, int_output_npu = self.op_exec_int(cpu_input, npu_input)
            self.assertRtolEqual(int_output_cpu, int_output_npu)
    
        
instantiate_device_type_tests(TestScalarCopy, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
