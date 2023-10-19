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
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestL1lossbackward(TestCase):
    def cpu_op_exec(self, input1, input2, input3, reduction):
        criterion = nn.L1Loss(reduction=reduction)
        input2.requires_grad = True
        loss = criterion(input2, input3)
        if reduction == "none":
            loss.backward(input1)
        else:
            loss.backward()
        output = input2.grad.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, reduction):
        input2.requires_grad = True
        criterion = nn.L1Loss(reduction=reduction)
        criterion = criterion.to("npu")
        loss = criterion(input2, input3)
        if reduction == "none":
            loss.backward(input1)
        else:
            loss.backward()
        output = input2.grad.to("cpu").numpy()
        return output

    def test_l1lossbackward_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4)], [np.float32, -1, (4)],
             [np.float32, -1, (4)], "none"],
            [[np.float32, -1, ()], [np.float32, -1, ()],
             [np.float32, -1, ()], "none"],
            [[np.float32, -1, (4, 3)], [np.float32, -1, (4, 3)],
             [np.float32, -1, (4, 3)], "sum"],
            [[np.float32, -1, (4, 1, 5)], [np.float32, -1, (4, 1, 5)],
             [np.float32, -1, (4, 1, 5)], "mean"],
            [[np.float32, -1, (4, 3)], [np.float32, -1, (4, 3)],
             [np.float32, -1, (4, 3)], "none"],
            [[np.float32, -1, (4, 1, 5)], [np.float32, -1, (4, 1, 5)],
             [np.float32, -1, (4, 1, 5)], "none"],
            [[np.float32, -1, (110, 55)], [np.float32, -1, (110, 55)],
             [np.float32, -1, (110, 55)], "none"],
            [[np.float32, -1, (11, 13, 12, 32)], [np.float32, -1, (11, 13, 12, 32)],
             [np.float32, -1, (11, 13, 12, 32)], "none"],
            [[np.float32, -1, (110, 55)], [np.float32, -1, (110, 55)],
             [np.float32, -1, (110, 55)], "sum"],
            [[np.float32, -1, (11, 13, 12, 32)], [np.float32, -1, (11, 13, 12, 32)],
             [np.float32, -1, (11, 13, 12, 32)], "sum"],
            [[np.float32, -1, (110, 55)], [np.float32, -1, (110, 55)],
             [np.float32, -1, (110, 55)], "sum"],
            [[np.float32, 0, (11, 13, 12, 32)], [np.float32, 0, (11, 13, 12, 32)],
             [np.float32, 0, (11, 13, 12, 32)], "mean"],
            [[np.float32, 3, (110, 55)], [np.float32, 4, (110, 55)],
             [np.float32, 4, (110, 55)], "mean"],
            [[np.float32, 29, (11, 13, 12, 32)], [np.float32, 3, (11, 13, 12, 32)],
             [np.float32, 4, (11, 13, 12, 32)], "mean"],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -1, 1)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, item[3])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_l1lossbackward_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1, input2, input3, reduction):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            input3 = input3.to(torch.float32)
            input2.requires_grad = True
            criterion = nn.L1Loss(reduction=reduction)
            loss = criterion(input2, input3)
            if reduction == "none":
                loss.backward(input1)
            else:
                loss.backward()
            output = input2.grad.numpy().astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (4, 3)], [np.float16, -1, (4, 3)],
             [np.float16, -1, (4, 3)], "none"],
            [[np.float16, -1, (4, 1, 5)], [np.float16, -1, (4, 1, 5)],
             [np.float16, -1, (4, 1, 5)], "none"],
            [[np.float16, -1, (110, 55)], [np.float16, -1, (110, 55)],
             [np.float16, -1, (110, 55)], "none"],
            [[np.float16, -1, (11, 13, 12, 32)], [np.float16, -1, (11, 13, 12, 32)],
             [np.float16, -1, (11, 13, 12, 32)], "none"],
            [[np.float16, -1, (4, 3)], [np.float16, -1, (4, 3)],
             [np.float16, -1, (4, 3)], "sum"],
            [[np.float16, -1, (4, 1, 5)], [np.float16, -1, (4, 1, 5)],
             [np.float16, -1, (4, 1, 5)], "mean"],
            [[np.float16, -1, (11, 13, 12, 32)], [np.float16, -1, (11, 13, 12, 32)],
             [np.float16, -1, (11, 13, 12, 32)], "none"],
            [[np.float16, -1, (110, 55)], [np.float16, -1, (110, 55)],
             [np.float16, -1, (110, 55)], "sum"],
            [[np.float16, -1, (11, 13, 12, 32)], [np.float16, -1, (11, 13, 12, 32)],
             [np.float16, -1, (11, 13, 12, 32)], "sum"],
            [[np.float16, -1, (110, 55)], [np.float16, -1, (110, 55)],
             [np.float16, -1, (110, 55)], "sum"],
            [[np.float16, 0, (11, 13, 12, 32)], [np.float16, 0, (11, 13, 12, 32)],
             [np.float16, 0, (11, 13, 12, 32)], "mean"],
            [[np.float16, 3, (110, 55)], [np.float16, 4, (110, 55)],
             [np.float16, 4, (110, 55)], "mean"],
            [[np.float16, 29, (11, 13, 12, 32)], [np.float16, 3, (11, 13, 12, 32)],
             [np.float16, 4, (11, 13, 12, 32)], "mean"],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10000)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 10000)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 10000)

            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, cpu_input3, item[3])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
