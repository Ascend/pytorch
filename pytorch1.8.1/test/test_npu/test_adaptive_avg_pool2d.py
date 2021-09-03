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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestAdaptiveAvgPool2d(TestCase):
    def cpu_op_exec(self, input, output_size):
        m = nn.AdaptiveAvgPool2d(output_size)
        output= m(input)
        return output.numpy()

    def npu_op_exec(self, input, output_size):
        m = nn.AdaptiveAvgPool2d(output_size).npu()
        output = m(input)
        return output.cpu().numpy()

    def test_adaptiveAvgPool2d_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = [(32, 16, 16),
                      (16, 1024, 256),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        # TODO(Ascend): tbe operator has problem in precision and (x, 1) case and so on.
        output_list = [(4, 4), (3, 5), (1), (1, None), (None, 2)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_adaptiveAvgPool2d_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [(32, 16, 16),
                      (16, 1024, 256),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        output_list = [(4, 4), (3, 5), (1), (1, None), (None, 2)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestAdaptiveAvgPool2d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
