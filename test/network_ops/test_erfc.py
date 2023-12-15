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


class TestErfc(TestCase):
    def get_shapeFormat(self):
        shape_format = [
            [np.float32, 0, (4, 3, 10, 9)],
            [np.float32, -1, (2, 4, 3)],
            [np.float32, 3, (20, 13)],
            [np.float32, 4, (20, 13)],
            [np.float32, 2, (100, 50)],
            [np.float32, 30, (20, 13, 10, 15, 20)]
        ]
        return shape_format

    def cpu_op_exec(self, input1):
        output = torch.erfc(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.erfc(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        torch.erfc_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        torch.erfc_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, cpu_out):
        torch.erfc(input1, out=cpu_out)
        output = cpu_out.numpy()
        return output

    def npu_op_exec_out(self, input1, npu_out):
        torch.erfc(input1, out=npu_out)
        output = npu_out.to("cpu")
        output = output.numpy()
        return output

    def test_erfc_float32_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erfc_float16_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erfc_float321_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erfc_float161_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erfc_out_float32_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_out, npu_out = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erfc_out_float16_common_shape_format(self, device="npu"):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_out, npu_out = create_common_tensor(item, 1, 100)
            cpu_out = cpu_out.to(torch.float32)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1, npu_out)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
