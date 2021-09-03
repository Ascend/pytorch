# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestIm2col(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, ksizes, rates, padding, strides):
        input1 = input1.to("cpu")
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1, ksizes, rates, padding, strides):
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_add_common_shape_format(self, device):
        shape_format = [
            [np.float32, -1, (2, 3, 3, 3)],
            [np.float32, -1, (2, 3, 4, 6)],
            [np.float32, -1, (4, 3, 4, 5)]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 20)
            cpu_output = self.cpu_op_exec(cpu_input1, (2,2), (1,1),(2,2),(1,1))
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, (2,2), (1,1),(2,2),(1,1))
            self.assertRtolEqual(cpu_output, npu_output)
      

    def test_add_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, ksizes, rates, padding, strides):
            input1 = input1.to(torch.float32)
            output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [np.float16, -1, (2, 3, 3, 3)],
            [np.float16, -1, (2, 3, 4, 6)], 
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, (2,2), (1,1),(2,2),(1,1))
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, (2,2), (1,1),(2,2),(1,1))
            self.assertRtolEqual(cpu_output, npu_output)


    def test_im2col__float32_data_range(self, device):
        data_range = [
            [-1.1754943508e-38, -1.1754943508e-38],
            [-3402823500.0, 3402823500.0],
            [-0.000030517578125, 0.000030517578125],
            [3402823500, 3402800000],
            [-9.313225746154785e-10, 9.313225746154785e-10],
            [-3402823500.0, -3402823500.0],
            [-3402823500.0, 3402823500.0],
            [-9.313225746154785e-10, 9.313225746154785e-10],
            [-3402823500.0,-3402823500.0],
            [-0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508],
            [0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508],
            [-0.000000000000000000000000000000000000011754943508, -0.000000000000000000000000000000000000011754943508],
            [-0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508]
        ]
        for item in data_range:
            cpu_input1, npu_input1 = create_common_tensor([np.float32, - 1, (1, 31, 149, 2)], item[0], item[1])
            cpu_output = self.cpu_op_exec(cpu_input1, (2,2), (1,1),(2,2),(1,1))
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, (2,2), (1,1),(2,2),(1,1))
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestIm2col, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()