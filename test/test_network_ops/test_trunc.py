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


class TestTrunc(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def cpu_op_exec(self, input1):
        output = torch.trunc(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.trunc(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_trunc_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float32, -1, (2, 3)]],
            [[np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (10,)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_add_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.trunc(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (2, 3)]],
            [[np.float16, -1, (4, 3, 1)]],
            [[np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (10,)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
