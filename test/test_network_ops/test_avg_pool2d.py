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
import torch_npu
import torch.nn as nn
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class TestAvgPool2d(TestCase):
    def cpu_op_exec(self, input1, ceil_mode):
        m = nn.AvgPool2d(3, stride=(6, 5), padding=0, ceil_mode=ceil_mode)
        output = m(input1)
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, input1, ceil_mode):
        m = nn.AvgPool2d(3, stride=(6, 5), padding=0, ceil_mode=ceil_mode).npu()
        output = m(input1)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_avg_pool2d_backward_shape_format_fp16(self, device):
        shape_format = [
            [[np.float16, 0, (1, 3, 147, 147)], True],
            [[np.float16, 0, (1, 3, 147, 147)], True]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input.float(), item[1]).astype(np.float16)
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output, prec16=0.002)

    def test_avg_pool2d_backward_shape_format_fp32(self, device):
        shape_format = [
            [[np.float32, 0, (1, 3, 147, 147)], True],
            [[np.float32, 0, (1, 3, 147, 147)], True]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output, 0.0009)

instantiate_device_type_tests(TestAvgPool2d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
