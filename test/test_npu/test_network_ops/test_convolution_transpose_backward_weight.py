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


class TestCudnnConvolutionTransposeBackwardWeight(TestCase):
    weight_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight, stride, padding, dilation, groups):
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        res_forward = torch._convolution(input1,
                                         weight,
                                         bias=None,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         transposed=True,
                                         output_padding=(0, 0),
                                         groups=groups,
                                         benchmark=True,
                                         deterministic=True,
                                         cudnn_enabled=False)
        print("===cpu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.detach().numpy()
        return res_forward

    def npu_op_exec(self, input1, weight, stride, padding, dilation, groups):
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        weight = weight.to("npu")
        res_forward = torch._convolution(input1,
                                         weight,
                                         bias=None,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         transposed=True,
                                         output_padding=(0, 0),
                                         groups=groups,
                                         benchmark=True,
                                         deterministic=True,
                                         cudnn_enabled=False)
        print("===npu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    def test_cudnn_convolution_transpose_backward_weight_shape_format(
            self, device):
        shape_format = [  # input, weight, stride, padding, dilation, groups
            [[np.float16, 0, (1, 4, 5, 5)], [np.float16, 0, (4, 4, 3, 3)],
             (1, 1), (1, 1), (1, 1), 1],
            [[np.float16, 3, (256, 8, 1, 1)], [np.float16, 3, (8, 8, 1, 1)],
             (1, 1), (0, 0), (1, 1), 1],
            [[np.float16, 3, [1024, 232, 7, 7]],
             [np.float16, 4, [232, 232, 1, 1]], (1, 1), (0, 0), (1, 1), 1],
            # [[np.float32, 0, (1, 4, 5, 5)], [np.float32, 0, (4, 4, 3, 3)],
            #  (1, 1), (1, 1), (1, 1), 1]
        ]

        for item in shape_format:
            self.weight_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[2],
                                          item[3], item[4], item[5])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[2],
                                          item[3], item[4], item[5])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.weight_grad[0] = self.weight_grad[0].to(
                self.weight_grad[1].dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1])


instantiate_device_type_tests(TestCudnnConvolutionTransposeBackwardWeight,
                              globals(),
                              except_for='cpu')
if __name__ == "__main__":
    run_tests()