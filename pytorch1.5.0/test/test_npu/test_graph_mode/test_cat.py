# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch
import numpy as np
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode


class TestCat(TestCase):
    def cpu_op_exec(self, input1, input2, n):
        output = torch.cat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, n):
        output = torch.cat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_exec_out(self, input1, input2, n, input3):
        torch.cat(input1+input2, n, out=input3)
        if not(input3.is_contiguous()):
            input3 = input3.contiguous()
        output = input3.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, n, input3):
        torch.cat(input1+input2, n, out=input3)
        if not(input3.is_contiguous()):
            input3 = input3.contiguous()
        output = input3.to("cpu")
        output = output.numpy()
        return output

    @RunFuncInGraphMode
    def test_cat_shape_float32_format(self, device):
        shape_format = [
            [[np.float32, 0, [32, 16, 1024], 1], [np.float32, 0, [32, 16, 1024], 1], 1],
            [[np.float32, 3, [256, 32, 56, 56],   1], [np.float32, 3, [256, 64, 56, 56],   1], 1],
            [[np.float32, 3, [256, 32, 56, 56],   6], [np.float32, 3, [256, 64, 56, 56],   1], 1],
            [[np.float32, 29, [256, 32, 28, 28],   1], [np.float32, 29, [256, 128, 28, 28],  1], 1],
            [[np.float32, 3, [256, 32, 28, 28],  12], [np.float32, 3, [256, 128, 28, 28],  1], 1],
            [[np.float32, 3, [256, 32, 14, 14],   1], [np.float32, 3, [256, 256, 14, 14],  1], 1],
            [[np.float32, 29, [32, 16, 1024], 1], [np.float32, 29, [32, 16, 1024], 1], 1],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1 = [cpu_input1] * item[0][3]
            npu_input1 = [npu_input1] * item[0][3]
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input2 = [cpu_input2] * item[1][3]
            npu_input2 = [npu_input2] * item[1][3]
            shape_input = item[0][2]
            dim = item[2]
            shape_input[dim] = item[0][2][dim]*item[0][3] + item[1][2][dim]*item[1][3] 
            item3 = item[0]
            item3[2] = shape_input
            cpu_input3, npu_input3 = create_common_tensor(item3, 0, 100)
            
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[2])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[2])
            
            cpu_output3 = self.cpu_op_exec_out(cpu_input1, cpu_input2, item[2], cpu_input3)
            npu_output3 = self.npu_op_exec_out(npu_input1, npu_input2, item[2], npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output3, npu_output3)

    @RunFuncInGraphMode
    def test_cat_shape_float16_format(self, device):
        shape_format = [
            [[np.float16, 3, [256, 32, 56, 56],   1], [np.float16, 3, [256, 64, 56, 56],   1], 1],
            [[np.float16, 3, [256, 32, 56, 56],   6], [np.float16, 3, [256, 64, 56, 56],   1], 1],
            [[np.float16, 3, [256, 32, 28, 28],   1], [np.float16, 3, [256, 128, 28, 28],  1], 1],
            [[np.float16, 2, [256, 32, 28, 28],  12], [np.float16, 2, [256, 128, 28, 28],  1], 1],
            [[np.float16, 3, [256, 32, 14, 14],   1], [np.float16, 3, [256, 256, 14, 14],  1], 1],
            [[np.float16, 3, [256, 32, 14, 14],  24], [np.float16, 3, [256, 256, 14, 14],  1], 1],
            [[np.float16, 3, [256, 32, 7, 7],     1], [np.float16, 3, [256, 512, 7, 7],    1], 1],
            [[np.float16, 29, [256, 32, 7, 7],    16], [np.float16, 29, [256, 512, 7, 7],    1], 1],
            [[np.float16, 0, [1024, 232, 7, 7],   1], [np.float16, 0, [1024, 232, 7, 7],   1], 1],
            [[np.float16, 0, [1024, 116, 14, 14], 1], [np.float16, 0, [1024, 116, 14, 14], 1], 1],
            [[np.float16, 3, [1024, 58, 28, 28],  1], [np.float16, 3, [1024, 58, 28, 28],  1], 1],
            [[np.float16, 3, [16, 1024], 0], [np.float16, 3, [16, 1024], 1], 1],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1 = [cpu_input1] * item[0][3]
            npu_input1 = [npu_input1] * item[0][3]
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input2 = [cpu_input2] * item[1][3]
            npu_input2 = [npu_input2] * item[1][3]
            
            shape_input = item[0][2]
            dim = item[2]
            shape_input[dim] = item[0][2][dim]*item[0][3] + item[1][2][dim]*item[1][3] 
            item3 = item[0]
            item3[2] = shape_input
            cpu_input3, npu_input3 = create_common_tensor(item3, 0, 100)
            
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[2])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[2])
            
            cpu_output3 = self.cpu_op_exec_out(cpu_input1, cpu_input2, item[2], cpu_input3)
            npu_output3 = self.npu_op_exec_out(npu_input1, npu_input2, item[2], npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output3, npu_output3)

    @RunFuncInGraphMode
    def test_cat_null_tensor(self, device):
        x1 = torch.randn(15, 2, 1, 1)
        x2 = torch.randn(0, 2, 1, 1)
        x3 = torch.randn(0, 2, 3, 1)
        y1_cpu = torch.cat([x1, x2], dim=0)
        y2_cpu = torch.cat([x2, x3], dim=2)
        y3_cpu = torch.cat([x2, x2, x2], dim=1)
        x1 = x1.npu()
        x2 = x2.npu()
        x3 = x3.npu()
        y1_npu = torch.cat([x1, x2], dim=0)
        y2_npu = torch.cat([x2, x3], dim=2)
        y3_npu = torch.cat([x2, x2, x2], dim=1)
        self.assertRtolEqual(y1_cpu, y1_npu.cpu())
        self.assertRtolEqual(y2_cpu, y2_npu.cpu())
        self.assertRtolEqual(y3_cpu, y3_npu.cpu())

instantiate_device_type_tests(TestCat, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
