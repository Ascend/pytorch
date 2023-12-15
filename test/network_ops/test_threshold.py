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


class TestThreshold(TestCase):

    def cpu_op_exec(self, input1, threshold, value):
        output = torch.nn.functional.threshold(input1, threshold, value)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, threshold, value):
        output = torch.nn.functional.threshold(input1, threshold, value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_threshold_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (1, 5)], [1.0], [20.0]],
            [[np.int32, 0, (1, 5)], [2], [20]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 3)
            cpu_threshold = npu_threshold = item[1][0]
            cpu_value = npu_value = item[2][0]
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_threshold, cpu_value)
            npu_output = self.npu_op_exec(npu_input1, npu_threshold, npu_value)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_threshold_inplace_common_shape_format(self, device="npu"):
        def cpu_op_inplace_exec(input1, threshold, value):
            torch.nn.functional.threshold_(input1, threshold, value)
            output = input1.numpy()
            return output

        def npu_op_inplace_exec(input1, threshold, value):
            torch.nn.functional.threshold_(input1, threshold, value)
            output = input1.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
            [[np.float32, 0, (1, 5)], [1.0], [20.0]],
            [[np.int32, 0, (1, 5)], [2], [20]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 3)
            cpu_threshold = npu_threshold = item[1][0]
            cpu_value = npu_value = item[2][0]
            cpu_output = cpu_op_inplace_exec(cpu_input1, cpu_threshold, cpu_value)
            npu_output = npu_op_inplace_exec(npu_input1, npu_threshold, npu_value)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
