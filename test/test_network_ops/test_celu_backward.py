# Copyright (c) 2023 Huawei Technologies Co., Ltd
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
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCeluBackward(TestCase):

    def cpu_op_exec(self, input_x, inplace):
        flag = False
        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            flag = True
        input_x.requires_grad = True
        celu_input = input_x + 1
        celu = torch.nn.CELU(inplace=inplace)
        output = celu(celu_input)
        output.backward(torch.ones_like(output))
        output_grad = input_x.grad
        if flag:
            output_grad = output_grad.to(torch.float16)
            output = output.to(torch.float16)
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def npu_op_exec(self, input_x, inplace):
        input_x.requires_grad = True
        celu_input = input_x + 1;
        celu = torch.nn.CELU(inplace=inplace)
        output = celu(celu_input)
        output.backward(torch.ones_like(output))
        output_grad = input_x.grad
        output_grad = output_grad.to("cpu")
        output = output.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def test_celu_common_shape_format_fp16(self):
        format_list = [0, 2, 3, 29]
        shape_list = [(234), (23, 13), (2, 3, 4), (64, 3, 34, 23)]
        inplace_list = [True, False]
        shape_format = [
            [[np.float16, i, j], k] for i in format_list for j in shape_list for k in inplace_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    def test_celu_common_shape_format_fp32(self):
        format_list = [0, 2, 3, 29]
        shape_list = [(234), (23, 13), (2, 3, 4), (64, 3, 34, 23)]
        inplace_list = [True, False]
        shape_format = [
            [[np.float32, i, j], k] for i in format_list for j in shape_list for k in inplace_list
        ]        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2.0, 2.0)
            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)


if __name__ == '__main__':
    run_tests()
