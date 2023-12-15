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


class TestLogSoftmax(TestCase):
    def cpu_op_exec(self, input1, dim):
        output = torch.nn.functional.log_softmax(input1, dim)
        output = output.numpy()
        return output

    def npu_op_exec_new(self, input1, dim):
        output = torch.nn.functional.log_softmax(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def logsoftmax_result(self, shape_format):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            print(item, " dim=", dim)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, 0)
            npu_output = self.npu_op_exec_new(npu_input1, 0)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_logsoftmax_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [32, 48, 64]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [32, 48, 1024]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [32, 24, 18, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [32, 24, 18, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)


if __name__ == "__main__":
    run_tests()
