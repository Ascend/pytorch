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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGridSampler(TestCase):
    def cpu_op_exec(self, input1, sample):
        input1.requires_grad = True
        sample.requires_grad = True
        output = torch.grid_sampler(input1, sample, 0, 0, True)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad.numpy()
        sample_grad = sample.grad.numpy()
        return input_grad, sample_grad

    def npu_op_exec(self, input1, sample):
        input1.requires_grad = True
        sample.requires_grad = True
        output = torch.grid_sampler(input1, sample, 0, 0, True)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad.to("cpu").numpy()
        sample_grad = sample.grad.to("cpu").numpy()
        return input_grad, sample_grad

    def result_grid_sampler(self, shape_format, sample_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_sample = cpu_sample.to(torch.float32)
            cpu_grad1, cpu_grad2 = self.cpu_op_exec(cpu_input, cpu_sample)
            npu_grad1, npu_grad2 = self.npu_op_exec(npu_input, npu_sample)
            cpu_grad1 = cpu_grad1.astype(npu_grad1.dtype)
            cpu_grad2 = npu_grad2.astype(npu_grad1.dtype)
            self.assertRtolEqual(cpu_grad1, npu_grad1)
            self.assertRtolEqual(cpu_grad2, npu_grad2)

    def test_grid_sampler_fp32(self):
        format_list = [0]
        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float32, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float32, 0, [100, 1, 1, 2]]
        self.result_grid_sampler(shape_format, sample_format)

    def test_grid_sampler_fp16(self):
        format_list = [0]
        shape_list = [[1, 1, 3, 3], [1, 2, 3, 4]]
        shape_format = [
            [np.float16, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float16, 0, [1, 2, 2, 2]]
        self.result_grid_sampler(shape_format, sample_format)


if __name__ == "__main__":
    run_tests()
