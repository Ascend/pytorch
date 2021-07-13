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


class TestCudnnConvolutionBackwardInput(TestCase):
    def cpu_op_exec(self, input1, weight, stride, padding, dilation, groups):
        input1.requires_grad = True
        res_forward = torch._convolution(input1,
                                         weight,
                                         bias=None,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         transposed=False,
                                         output_padding=(0, 0),
                                         groups=groups,
                                         benchmark=True,
                                         deterministic=True,
                                         cudnn_enabled=True)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.detach().numpy()
        gradinput = input1.grad
        return res_forward, gradinput

    def npu_op_exec(self, input1, weight, stride, padding, dilation, groups):
        input1.requires_grad = True
        weight = weight.to("npu")
        res_forward = torch._convolution(input1,
                                         weight,
                                         bias=None,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         transposed=False,
                                         output_padding=(0, 0),
                                         groups=groups,
                                         benchmark=True,
                                         deterministic=True,
                                         cudnn_enabled=True)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        gradinput = input1.grad.to("cpu")
        return res_forward, gradinput

    def test_cudnn_convolution_backward_input_shape_format(self, device):
        shape_format = [  # input, weight, stride, padding, dilation, groups
            [[np.float16, 0, (1, 4, 5, 5)], [np.float16, 0, (4, 4, 3, 3)],
             (1, 1), (1, 1), (1, 1), 1],
            [[np.float32, 0, [256, 3, 224, 224]],
             [np.float32, 0, [32, 3, 3, 3]], [2, 2], [0, 0], [1, 1], 1],
            [[np.float16, 3, (256, 8, 1, 1)], [np.float16, 3, (8, 8, 1, 1)],
             (1, 1), (0, 0), (1, 1), 1],
            [[np.float16, 3, [1024, 232, 7, 7]],
             [np.float16, 4, [232, 232, 1, 1]], (1, 1), (0, 0), (1, 1), 1],
            [[np.float32, 0, (1, 4, 5, 5)], [np.float32, 0, (4, 4, 3, 3)],
             (1, 1), (1, 1), (1, 1), 1]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output, cpu_dinput = self.cpu_op_exec(cpu_input1, cpu_input2,
                                                      item[2], item[3],
                                                      item[4], item[5])
            npu_output, npu_dinput = self.npu_op_exec(npu_input1, npu_input2,
                                                      item[2], item[3],
                                                      item[4], item[5])
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_dinput = npu_dinput.to(npu_dinput.dtype)
            self.assertRtolEqual(cpu_output, npu_output, 1e-2)
            self.assertRtolEqual(cpu_dinput, npu_dinput)


instantiate_device_type_tests(TestCudnnConvolutionBackwardInput,
                              globals(),
                              except_for='cpu')
if __name__ == "__main__":
    run_tests()