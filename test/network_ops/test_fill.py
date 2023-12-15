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


class TestFill_(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.fill_(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.fill_(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_fills_shape_format_fp16(self, device='npu'):
        format_list = [0, 3]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [0.8, 1.25, torch.tensor(0.8), torch.tensor(1.25)]
        shape_format = [
            [[np.float16, i, j], v] for i in format_list for j in shape_list for v in value_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_shape_format_fp32(self, device='npu'):
        format_list = [0, 3]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [0.8, 1.25, torch.tensor(0.8), torch.tensor(1.25)]
        shape_format = [
            [[np.float32, i, j], v] for i in format_list for j in shape_list for v in value_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_zero_dim_shape_format(self, device='npu'):
        dtype_list = [torch.float16, torch.float32]
        init_list = [3, 73, 1000]
        value_list = [0.8, 1.25, torch.tensor(0.8), torch.tensor(1.25)]
        shape_format = [
            [i, j, v] for i in dtype_list for j in init_list for v in value_list
        ]
        for item in shape_format:
            cpu_input1 = torch.tensor(item[1], dtype=item[0])
            npu_input1 = cpu_input1.npu()
            cpu_output = self.cpu_op_exec(cpu_input1, item[2])
            npu_output = self.npu_op_exec(npu_input1, item[2])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
