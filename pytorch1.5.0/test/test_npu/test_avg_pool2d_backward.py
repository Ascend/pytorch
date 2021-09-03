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
import torch.nn as nn
import numpy as np
from torch.testing._internal.common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

def cpu_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad

def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.cpu()

class TestAvgPool2dBackward(TestCase):
    #could not find avg_pool2d_backward in test_torch.py
    def cpu_op_exec(self, input):
        m = nn.AvgPool2d(kernel_size=2, stride=2)

        input.requires_grad = True
        input.register_hook(cpu_input_grad_hook)
    
        output = m(input)
        z = torch.sum(output)
        z.backward()

    def npu_op_exec(self, input):
        m = nn.AvgPool2d(kernel_size=2, stride=2).npu()

        input.requires_grad = True
        input.register_hook(npu_input_grad_hook)

        output = m(input)
        z = torch.sum(output)
        z.backward()

    def test_avg_pool2d_backward_shape_format(self, device):
        shape_format = [
                [np.float32, 0, (64, 10, 16, 14)],
                [np.float32, 3, (256, 2048, 8, 8)],
                [np.float32, 4, (32, 1, 2, 2)],
                [np.float32, 0, (10, 128, 16, 16)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            
            self.cpu_op_exec(cpu_input)
            self.npu_op_exec(npu_input)
            self.assertEqual(cpu_input_grad, npu_input_grad)


instantiate_device_type_tests(TestAvgPool2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()