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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestNpuLinearBackward(TestCase):
    def cpu_op_exec(self, x, weight, bias):
        x.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True
        output = torch.nn.functional.linear(x, weight, bias)
        loss = output.sum()
        loss.backward()
        return output.detach().numpy(), x.grad.numpy(), weight.grad.numpy(), bias.grad.numpy()
    
    def npu_op_exec(self, x, weight, bias):
        x.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True
        output = torch.npu_linear(x, weight, bias)
        loss = output.sum()
        loss.backward()
        return output.cpu().detach().numpy(), x.grad.cpu().numpy(), weight.grad.cpu().numpy(), bias.grad.cpu().numpy()

    def test_npu_linear_backward_shape_format_fp32(self, device):
        shape_format = [
                [[np.float32, -1, (6144, 1024)], [np.float32, -1, (256, 1024)], [np.float32, -1, (256)]],
                [[np.float32, -1, (123, 456)], [np.float32, -1, (789, 456)], [np.float32, -1, (789)]],
        ]

        for item in shape_format:
            cpu_x, npu_x = create_common_tensor(item[0], -2, 2)
            cpu_w, npu_w = create_common_tensor(item[1], -2, 2)
            cpu_b, npu_b = create_common_tensor(item[2], -2, 2)
            cpu_output, cpu_x_grad, cpu_w_grad, cpu_b_grad = self.cpu_op_exec(cpu_x, cpu_w, cpu_b)
            npu_output, npu_x_grad, npu_w_grad, npu_b_grad = self.npu_op_exec(npu_x, npu_w, npu_b)
            self.assertRtolEqual(cpu_output, npu_output, 0.0002)
            self.assertRtolEqual(cpu_x_grad, npu_x_grad)
            self.assertRtolEqual(cpu_w_grad, npu_w_grad)
            self.assertRtolEqual(cpu_b_grad, npu_b_grad)
    
    def test_npu_linear_shape_format_fp16(self, device):
        shape_format = [
                [[np.float16, -1, (6144, 1024)], [np.float16, -1, (256, 1024)], [np.float16, -1, (256)]],
                [[np.float16, -1, (123, 456)], [np.float16, -1, (789, 456)], [np.float16, -1, (789)]],
        ]

        for item in shape_format:
            cpu_x, npu_x = create_common_tensor(item[0], -2, 2)
            cpu_w, npu_w = create_common_tensor(item[1], -2, 2)
            cpu_b, npu_b = create_common_tensor(item[2], -2, 2)
            cpu_output, cpu_x_grad, cpu_w_grad, cpu_b_grad = self.cpu_op_exec(
                cpu_x.float(), cpu_w.float(), cpu_b.float())
            npu_output, npu_x_grad, npu_w_grad, npu_b_grad = self.npu_op_exec(npu_x, npu_w, npu_b)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)
            self.assertRtolEqual(cpu_x_grad.astype(np.float16), npu_x_grad)
            self.assertRtolEqual(cpu_w_grad.astype(np.float16), npu_w_grad)
            self.assertRtolEqual(cpu_b_grad.astype(np.float16), npu_b_grad)

instantiate_device_type_tests(TestNpuLinearBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

