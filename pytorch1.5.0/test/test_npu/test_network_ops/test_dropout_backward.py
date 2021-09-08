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

import sys
sys.path.append('..')
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDropOutBackward(TestCase):
    def op_exec(self, npu_flag, input):
        input.requires_grad = True
        out = torch.nn.Dropout(0.5)(input)
        out.backward(torch.ones_like(out))
        out_grad = input.grad
        if npu_flag:
            out_grad = out_grad.to("cpu")
        out_grad = out_grad.detach().numpy()
        if npu_flag:
            out = out.to("cpu")
        out = out.detach().numpy()
        return out_grad, out

    def dropout_list_exec(self, list):
        epsilon = 1e-3
        for item in list:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_grad, cpu_output = self.op_exec(0, cpu_input1)
            npu_output_grad, npu_output = self.op_exec(1, npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            # 该算子随机结果的比较方式
            for a, b in zip(cpu_output.flatten(), npu_output.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

            for a, b in zip(cpu_output_grad.flatten(), npu_output_grad.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

    def test_op_shape_format_fp16(self, device):
        format_list = [-1]
        shape_list = [1, (32, 3, 3)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_op_shape_format_fp32(self, device):
        format_list = [-1]
        shape_list = [1, (32, 3, 3)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

instantiate_device_type_tests(TestDropOutBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()