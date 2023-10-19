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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGridSampler2D(TestCase):
    def cpu_op_exec(self, input1, grid):
        output = torch.grid_sampler_2d(input1, grid, 0, 0, True)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, grid):
        output = torch.grid_sampler_2d(input1, grid, 0, 0, True)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_fp16_exec(self, input1, grid):
        input1 = input1.to(torch.float32)
        grid = grid.to(torch.float32)
        output = torch.grid_sampler_2d(input1, grid, 0, 0, True)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def test_grid_sampler_2d_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (1, 2, 4, 20)], [np.float32, 0, (1, 10, 8, 2)]],
            [[np.float32, 0, (1, 4, 64, 10)], [np.float32, 0, (1, 2, 32, 2)]],
            [[np.float32, 0, (2, 2048, 7, 7)], [np.float32, 0, (2, 2048, 14, 2)]],
            [[np.float32, 4, (32, 1, 3, 3)], [np.float32, 4, (32, 20, 30, 2)]],
            [[np.float32, 29, (1, 2, 10, 128)], [np.float32, 4, (1, 10, 5, 2)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_grid, npu_grid = create_common_tensor(item[1], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_grid)
            npu_output = self.npu_op_exec(npu_input, npu_grid)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_grid_sampler_2d_fp16_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (1, 2, 4, 20)], [np.float16, 0, (1, 10, 8, 2)]],
            [[np.float16, 0, (1, 4, 64, 10)], [np.float16, 0, (1, 2, 32, 2)]],
            [[np.float16, 0, (2, 2048, 7, 7)], [np.float16, 0, (2, 2048, 14, 2)]],
            [[np.float16, 4, (32, 1, 3, 3)], [np.float16, 4, (32, 20, 30, 2)]],
            [[np.float16, 29, (1, 2, 10, 128)], [np.float16, 4, (1, 10, 5, 2)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_grid, npu_grid = create_common_tensor(item[1], -3, 3)
            cpu_output = self.cpu_op_fp16_exec(cpu_input, cpu_grid)
            npu_output = self.npu_op_exec(npu_input, npu_grid)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
