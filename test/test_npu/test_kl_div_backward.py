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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestKlDivBackward(TestCase):
    def generate_data(self, shape, dtype): 
        x = np.array([0.5, 0.1, 0.4]).astype(dtype)
        x =  torch.from_numpy(x)
        y = np.array([0.4, 0.4, 0.2]).astype(dtype)
        y = torch.from_numpy(y)
        return x, y

    def cpu_op_exec(self, input, target, reduction): 
        input.requires_grad_(True)
        target.requires_grad_(True)
        output = torch.kl_div(input.log(), target, reduction=reduction)
        input_cpu = output.detach().numpy()
        if reduction == 0:
            grad_output = torch.ones_like(input)
            output.backward(grad_output)
        else:
            output.backward()
        res = input.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec(self, input, target, reduction):
        input = input.to("npu")
        target = target.to("npu")
        input.requires_grad_(True)
        target.requires_grad_(True)
        output = torch.kl_div(input.log(), target, reduction=reduction)
        input_npu = output.to("cpu")
        input_npu = input_npu.detach().numpy()
        if reduction == 0:
            grad_output = torch.ones_like(input)
            grad_output = grad_output.to("npu")
            output.backward(grad_output)
        else:
            output.backward()
        res = input.grad.to("cpu")
        res = res.numpy()
        return input_npu, res

    def test_kl_div_backward_float32(self, device):
        shape_format = [
            [(3, 1), np.float32, 0],
            [(3, 1), np.float32, 1],
            [(3, 1), np.float32, 2],
        ]

        for item in shape_format:
            npu_x, npu_y = self.generate_data(item[0], item[1])
            cpu_x = copy.deepcopy(npu_x)
            cpu_y = copy.deepcopy(npu_y)
            reduction = item[2]
            cpu_input, cpu_output = self.cpu_op_exec(cpu_x, cpu_y, reduction)
            npu_input, npu_output = self.npu_op_exec(npu_x, npu_y, reduction) 
            self.assertRtolEqual(cpu_input, npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_kl_div_backward_float16(self, device):
        def cpu_op_exec_fp16(input, target, reduction):
            input = input.to(torch.float32)
            target = target.to(torch.float32)
            input.requires_grad_(True)
            target.requires_grad_(True)
            output = torch.kl_div(input.log(), target, reduction=reduction)
            input_cpu = output.detach().numpy()
            if reduction == 0:
                grad_output = torch.ones_like(input)
                output.backward(grad_output)
            else:
                output.backward()
            res = input.grad
            res = res.numpy()
            input_cpu = input_cpu.astype(np.float16)
            res = res.astype(np.float16)
            return input_cpu, res
        
        shape_format = [
            [(3, 1), np.float16, 0],
            [(3, 1), np.float16, 1],
            [(3, 1), np.float16, 2],
        ]

        for item in shape_format:
            npu_x, npu_y = self.generate_data(item[0], item[1])
            cpu_x = copy.deepcopy(npu_x)
            cpu_y = copy.deepcopy(npu_y)
            reduction = item[2]
            cpu_input, cpu_output = cpu_op_exec_fp16(cpu_x, cpu_y, reduction)
            npu_input, npu_output = self.npu_op_exec(npu_x, npu_y, reduction) 
            self.assertRtolEqual(cpu_input, npu_input)
            self.assertRtolEqual(cpu_output, npu_output)        

instantiate_device_type_tests(TestKlDivBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:0")
    run_tests()