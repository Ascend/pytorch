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

import torch.nn as nn
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAdaptiveMaxPool2d(TestCase):
    def cpu_op_exec(self, input1, output_size):
        m = nn.AdaptiveMaxPool2d(output_size)
        output = m(input1)
        return output.numpy()

    def npu_op_exec(self, input1, output_size):
        m = nn.AdaptiveMaxPool2d(output_size).npu()
        output = m(input1)
        return output.cpu().numpy()

    def test_adaptive_max_pool2d_shape_format_fp32_6(self, device="npu"):
        format_list = [-1]
        shape_list = [(1, 5, 9, 9)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        output_list = [(3, 3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output, 0.0004)


if __name__ == "__main__":
    run_tests()
