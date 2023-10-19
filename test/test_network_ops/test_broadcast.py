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


class TestBroadCastTensors(TestCase):
    def cpu_op_exec(self, input1, input2):
        output1, output2 = torch.broadcast_tensors(input1, input2)
        return output1.numpy(), output2.numpy()

    def npu_op_exec(self, input1, input2, npu_format=None):
        input1 = input1.npu()
        input2 = input2.npu()
        if npu_format is not None:
            input1 = torch.npu_format_cast(input1, npu_format)
            input2 = torch.npu_format_cast(input2, npu_format)
        output1, output2 = torch.broadcast_tensors(input1, input2)
        return output1.cpu().numpy(), output2.cpu().numpy()

    def test_broadcast_tensors_common_shape_format(self, device='npu'):
        shape_format = [
            [[1, 3], (2, 1), torch.float32],
            [[1, 9], (5, 1), torch.float32],
            [[3, 1], (1, 3), torch.float32],
        ]
        for item in shape_format:
            cpu_input1 = torch.randn(item[0], dtype=item[2])
            cpu_input2 = torch.randn(item[1], dtype=item[2])
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output1, npu_output2 = self.npu_op_exec(cpu_input1, cpu_input2)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_broadcast_tensors_discontiguous_shape(self, device='npu'):
        shape_format = [
            [[1, 6], (4, 1), torch.float32],
            [[1, 18], (10, 1), torch.float32],
        ]
        for item in shape_format:
            cpu_input1 = torch.randn(item[0], dtype=item[2])[:, ::2]
            cpu_input2 = torch.randn(item[1], dtype=item[2])[::2, :]
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output1, npu_output2 = self.npu_op_exec(cpu_input1, cpu_input2)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_broadcast_tensors_format(self, device='npu'):
        shape_format = [
            [[2, 3, 4], (3, 1), torch.float32, 0],
            [[2, 3, 4], (3, 1), torch.float32, 2],
            [[2, 3, 4], (3, 1), torch.float32, 3],
            [[2, 3, 4], (3, 1), torch.float32, 4],
            [[2, 3, 4], (3, 1), torch.float32, 29],
            [[2, 3, 4], (3, 1), torch.float32, 30],
        ]
        for item in shape_format:
            cpu_input1 = torch.randn(item[0], dtype=item[2])[:, ::2]
            cpu_input2 = torch.randn(item[1], dtype=item[2])[::2, :]
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_format = item[3]
            npu_output1, npu_output2 = self.npu_op_exec(cpu_input1, cpu_input2, npu_format)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
