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


class TestPdist(TestCase):
    def cpu_op_exec_default(self, input1):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()
        output = torch.nn.functional.pdist(input1)
        if stype == torch.float16:
            output = output.half()
        output = output.numpy()
        return output

    def npu_op_exec_default(self, input1):
        output = torch.nn.functional.pdist(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, p):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()
        output = torch.nn.functional.pdist(input1, p)
        if stype == torch.float16:
            output = output.half()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, p):
        output = torch.nn.functional.pdist(input1, p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_pdist_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 2, (30, 301)], -2, 2, 0.0],
            [[np.float16, 2, (10, 1600)], -2, 2, 1.0],
            [[np.float16, 2, (9, 10250)], -2, 2, 2.0],
            [[np.float16, 2, (111, 10025)], -2, 2, 10.0],
            [[np.float32, 2, (10, 256)], -2, 2, 0.0],
            [[np.float32, 2, (20, 234)], -2, 2, 1.0],
            [[np.float32, 2, (8, 1025)], -2, 2, 2.0],
            [[np.float32, 2, (100, 7025)], -2, 2, 10.0],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = self.cpu_op_exec(cpu_input1, item[3])
            npu_output = self.npu_op_exec(npu_input1, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 2, (5, 360)], -2, 2],
            [[np.float32, 2, (10, 3600)], -2, 2],
            [[np.float32, 2, (50, 0)], -2, 2],
            [[np.float32, 2, (1, 110)], -2, 2],
            [[np.float32, 2, (0, 0)], -2, 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = self.cpu_op_exec_default(cpu_input1)
            npu_output = self.npu_op_exec_default(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
