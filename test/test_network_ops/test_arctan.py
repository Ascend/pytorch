# Copyright (c) 2022 Huawei Technologies Co., Ltd
# Copyright (c) 2022, Facebook CORPORATION.
# All rights reserved.
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


class TestArctan(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.arctan(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.arctan(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.arctan(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.arctan_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        output = torch.arctan_(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_arctan_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, 1]],
            [[np.float32, 0, (64, 10)]],
            [[np.float32, 3, (256, 2048, 7, 7)]],
            [[np.float32, 4, (32, 1, 3, 3)]],
            [[np.float32, 29, (10, 128)]]
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_arctan_out_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, 1]],
            [[np.float32, 0, (64, 10)]],
            [[np.float32, 3, (256, 2048, 7, 7)]],
            [[np.float32, 4, (32, 1, 3, 3)]],
            [[np.float32, 29, (10, 128)]]
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -1, 1)
            cpu_output1 = self.cpu_op_exec(cpu_input1)
            npu_output1 = self.npu_op_exec_out(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_arctan_inp_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, 1]],
            [[np.float32, 0, (64, 10)]],
            [[np.float32, 3, (256, 2048, 7, 7)]],
            [[np.float32, 4, (32, 1, 3, 3)]],
            [[np.float32, 29, (10, 128)]]
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_output1 = self.cpu_inp_op_exec(cpu_input1)
            npu_output1 = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
