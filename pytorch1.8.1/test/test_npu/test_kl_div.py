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


class TestKlDiv(TestCase):

    def cpu_op_exec(self, input1, input2, reduction):
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, reduction):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_kl_div_common_shape_format(self, device):
        shape_format = [
            [[np.float32, 0, (4, 1, 2, 3)], [np.float32, 0, (4, 1, 2, 3)], 0],
            [[np.float32, 0, (4, 1, 5)], [np.float32, 0, (4, 1, 5)], 1],
            [[np.float32, 0, (14, 21, 52, 10, 22)], [
                np.float32, 0, (14, 21, 52, 10, 22)], 2],
            # 130device unsupports float64 
            # [[np.float64, 0, (24, 9, 15)], [np.float64, 0, (24, 9, 15)], 2],
            # [[np.float64, -1, (24, 11)], [np.float64, -1, (24, 11)], 1],
            # [[np.float64, 0, (14, 21, 52, 10, 22)], [np.float64, 0, (14, 21, 52, 10, 22)], 0]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, reduction)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_kl_div_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, input2, reduction):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.kl_div(input1, input2, reduction=reduction)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 0, (14, 21, 22, 33)], [
                np.float16, 0, (14, 21, 22, 33)], 0],
            [[np.float16, 0, (4, 10, 5)], [np.float16, 0, (4, 10, 5)], 1],
            [[np.float16, 0, (4, 1, 50)], [np.float16, 0, (4, 1, 50)], 2],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            reduction = item[2]
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, reduction)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, reduction)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestKlDiv, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
