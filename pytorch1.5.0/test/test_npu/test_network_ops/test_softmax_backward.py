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
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


def input_grad_hook(grad):
    global input_grad
    input_grad = grad


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")


class TestSoftmaxBackward(TestCase):

    def cpu_op_exec(self, input, is_contiguous=True, dim=-1):
        if is_contiguous is False:
            input = input.as_strided([2, 2], [1, 2], 1)
        input.requires_grad = True
        input.register_hook(input_grad_hook)

        output = torch.softmax(input, dim=dim)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input, is_contiguous=True, dim=-1):
        if is_contiguous is False:
            input = input.as_strided([2, 2], [1, 2], 1)
        input.requires_grad = True
        input.register_hook(npu_input_grad_hook)

        output = torch.softmax(input, dim=dim)
        z = output.sum()
        z.backward()
        input = input.cpu()
    
    def cpu_op_exec_nz(self, input, dim=-1):
        input.requires_grad = True
        output = torch.softmax(input, dim=dim)
        output.backward(torch.ones_like(output))
        return input.grad

    def npu_op_exec_nz(self, input, dim=-1):
        input.requires_grad = True
        output = torch.softmax(input, dim=dim)
        output.backward(torch.ones_like(output))
        return input.grad.cpu()

    def test_softmax_backward_shape_format(self, device):
        shape_format = [
            [np.float32, 0, 5],
            [np.float32, 3, (64, 10)],
            [np.float32, 3, (256, 2048, 7, 7)],
            [np.float32, 3, (32, 1, 3, 3)],
            [np.float32, 0, (10, 128)]
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)
            input2, npu_input2 = create_common_tensor(item, 10, 100)

            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)
            self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())

            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())

    def test_softmax_backward_shape_format_fp16(self, device):
        shape_format = [
            [np.float16, 0, 5],
            [np.float16, 3, (64, 10)],
            [np.float16, 3, (256, 2048, 7, 7)],
            [np.float16, 3, (32, 1, 3, 3)],
            [np.float16, 0, (10, 128)]
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)
            input2, npu_input2 = create_common_tensor(item, 10, 100)

            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)

            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)

            self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy())

            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy())
    
    def test_softmax_backward_shape_format_dim(self, device):
        shape_format = [
            [[np.float32, -1, (7, 3, 3, 13, 3, 19, 2, 3)], 5],
            [[np.float16, -1, (7, 3, 3, 13, 3, 19, 2, 3)], 5]  
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            self.npu_op_exec(npu_input1, dim=item[1])
            if item[0][0] is np.float16:
                self.cpu_op_exec(input1.float(), dim=item[1])
                self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy())
            else:
                self.cpu_op_exec(input1, dim=item[1])
                self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())
    
    def test_softmax_backward_shape_format_nz_fp32(self, device):
        shape_format = [
            [np.float32, 29, 5],
            [np.float32, 29, (64, 10)],
            [np.float32, 29, (32, 3, 3)],
            [np.float32, 29, (256, 2048, 7, 7)],
        ]

        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)

            cpu_input1_grad = self.cpu_op_exec_nz(input1)
            npu_input1_grad = self.npu_op_exec_nz(npu_input1)

            self.assertRtolEqual(cpu_input1_grad.numpy(), npu_input1_grad.numpy())

    def test_softmax_backward_shape_format_nz_fp16(self, device):
        shape_format = [
            [np.float16, 29, 5],
            [np.float16, 29, (64, 10)],
            [np.float16, 29, (32, 3, 3)],
            [np.float16, 29, (256, 2048, 7, 7)],
        ]

        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 10, 100)

            if input1.dtype == torch.float16:
                cpu_input1_grad = self.cpu_op_exec_nz(input1.float()).half()
            else:
                cpu_input1_grad = self.cpu_op_exec_nz(input1)
            npu_input1_grad = self.npu_op_exec_nz(npu_input1)

            self.assertRtolEqual(cpu_input1_grad.numpy(), npu_input1_grad.numpy())

    def cpu_exec_case_in_hrnet_ocr(self, x, y, z):
        x.requires_grad = True
        mm1_out = x @ y
        softmax_out = F.softmax(mm1_out, dim=-1)
        mm2_out = softmax_out @ z
        l = mm2_out.sum()
        l.backward()
        return x.grad
    
    def npu_exec_case_in_hrnet_ocr(self, x, y, z):
        x.requires_grad = True
        mm1_out = x @ y
        mm1_out.npu_format_cast_(2)
        softmax_out = F.softmax(mm1_out, dim=-1)
        mm2_out = softmax_out @ z
        l = mm2_out.sum()
        l.backward()
        return x.grad.cpu()

    def test_softmax_backward_case_in_hrnet_ocr(self, device):
        N  = 32768
        cpu_x, npu_x = create_common_tensor([np.float16, -1, (1, 19, 16)], -2, 2)
        cpu_y, npu_y = create_common_tensor([np.float16, -1, (1, 16, N)], -2, 2)
        cpu_z, npu_z = create_common_tensor([np.float16, -1, (1, N, 16)], -2, 2)
        
        cpu_x_grad = self.cpu_exec_case_in_hrnet_ocr(cpu_x.float(), cpu_y.float(), cpu_z.float()).half()
        npu_x_grad = self.npu_exec_case_in_hrnet_ocr(npu_x, npu_y, npu_z)
        self.assertRtolEqual(cpu_x_grad.numpy(), npu_x_grad.numpy(), prec16=0.009)


instantiate_device_type_tests(TestSoftmaxBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
