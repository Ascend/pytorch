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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMaxBackward(TestCase):
    def cpu_op_exec(self, input):
        input.requires_grad = True
        output = input.max(0, False)
        output[0].backward(torch.ones_like(output[0]))
        output_grad = input.grad
        output_grad = output_grad.detach().numpy()
        output1 = output[0].detach().numpy()
        output2 = output[1].detach().numpy()
        return output_grad, output1, output2

    def npu_op_exec(self, input):
        input.requires_grad = True
        output = input.max(0, False)
        output[0].backward(torch.ones_like(output[0]))
        output_grad = input.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output1 = output[0].detach().cpu().numpy()
        output2 = output[1].detach().cpu().numpy()
        return output_grad, output1, output2

    def test_avg_pool2d_backward_shape_format_fp32(self, device):
        format_list = [-1]
        shape_list = [(2,32,8,600,40),(2,32,16,300,40)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output_grad, cpu_output1, cpu_output2= self.cpu_op_exec(cpu_input)
            npu_output_grad, npu_output1, npu_output2 = self.npu_op_exec(npu_input)

            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    def test_avg_pool2d_backward_shape_format_fp16(self, device):
        format_list = [-1]
        shape_list = [(2,32,8,600),(2,32,16,300,40)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output_grad, cpu_output1, cpu_output2= self.cpu_op_exec(cpu_input)
            npu_output_grad, npu_output1, npu_output2 = self.npu_op_exec(npu_input)
            cpu_output1 = cpu_output1.astype(npu_output1.dtype)
            cpu_output_grad = cpu_output_grad.astype(npu_output_grad.dtype)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

instantiate_device_type_tests(
    TestMaxBackward,
    globals(),
    except_for="cpu")
if __name__ == "__main__":
    run_tests()
