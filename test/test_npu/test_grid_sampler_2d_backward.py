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

class TestGridSampler2dBackward(TestCase):
    def cpu_op_exec(self, input, sample):
        input.requires_grad = True
        out = torch.grid_sampler(input, sample, 0, 0, True)
        grad_output = torch.ones(out.size(), dtype=torch.float)
        out.backward(gradient=grad_output)
        output = input.grad.numpy()
        return output

    def npu_op_exec(self, input, sample): 
        input.requires_grad = True
        out = torch.grid_sampler(input, sample, 0, 0, True)
        grad_output = torch.ones(out.size(), dtype=torch.float).npu()
        out.backward(gradient=grad_output)
        output = input.grad.to("cpu").numpy()
        return output

    def test_grid_sampler_2d_backward_fp32(self, device):
        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float32, -1, j] for j in shape_list
        ]
        sample_format = [np.float32, -1, [100, 1, 1, 2]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_sample)
            # npu_output = self.npu_op_exec(npu_input, npu_sample)
            # self.assertRtolEqual(cpu_output, npu_output)
    
    def test_grid_sampler_2d_backward_fp16(self, device):
        def cpu_op_fp16_exec(input, sample):
            input = input.to(torch.float32)
            sample = sample.to(torch.float32)
            input.requires_grad = True
            out = torch.grid_sampler(input, sample, 0, 0, True)
            grad_output = torch.ones(out.size(), dtype=torch.float)
            out.backward(gradient=grad_output)
            output = input.grad.numpy()
            output = output.astype(np.float16)
            return output

        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float16, -1, j] for j in shape_list
        ]
        sample_format = [np.float16, -1, [100, 1, 1, 2]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = cpu_op_fp16_exec(cpu_input, cpu_sample)
            # npu_output = self.npu_op_exec(npu_input, npu_sample)
            # self.assertRtolEqual(cpu_output, npu_output.astype(np.float16))

instantiate_device_type_tests(TestGridSampler2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:4")
    run_tests()