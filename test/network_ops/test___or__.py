# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
# See the License for the specific language governing permissions or
# limitations under the License.

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class Test__Or__(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.__or__(input2)
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        return output.numpy()

    def npu_op_exec(self, input1, input2):
        output = input1.__or__(input2)
        output = output.to("cpu")
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        return output.numpy()

    def test___Or___shape_format(self, device="npu"):
        shape_format = [
            [[np.int32, 0, [256, 1000]], [1]],
            [[np.int32, 0, [256, 1000]], [np.int32, 0, [256, 1000]]],
            [[np.int16, 0, [256, 1000]], [2]],
            [[np.int16, 0, [256, 1000]], [np.int16, 0, [256, 1000]]],
            [[np.int8, 0, [256, 1000]], [3]],
            [[np.int8, 0, [256, 1000]], [np.int8, 0, [256, 1000]]],
        ]

        for item in shape_format:
            if len(item[1]) > 1:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
                cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
                npu_output = self.npu_op_exec(npu_input1, npu_input2)
                self.assertRtolEqual(cpu_output, npu_output)
            else:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                cpu_output = self.cpu_op_exec(cpu_input1, item[1][0])
                npu_output = self.npu_op_exec(npu_input1, item[1][0])
                self.assertRtolEqual(cpu_output, npu_output)

        cpu_input1 = torch.tensor([True, False, True, False, True], dtype=torch.bool)
        npu_input1 = torch.tensor([True, False, True, False, True], dtype=torch.bool, device="npu")
        cpu_input2 = torch.tensor([True, False, True, False, False], dtype=torch.bool)
        npu_input2 = torch.tensor([True, False, True, False, False], dtype=torch.bool, device="npu")
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
