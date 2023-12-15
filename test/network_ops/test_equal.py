# Copyright (c) 2021 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTensorEqual(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.equal(input1, input2)
        output = torch.tensor(output)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.equal(input1, input2)
        output = torch.tensor(output).to("cpu")
        output = output.numpy()
        return output

    def test_tensor_equal_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.float32, 29, (4, 3, 1)], [np.float32, 29, (4, 1, 5)]],
            [[np.float32, 2, (4, 3, 2)], [np.float32, 2, (4, 3, 2)]],
            [[np.float32, 3, (7, 3, 2)], [np.float32, 3, (7, 3, 2)]],
            [[np.float32, 4, (8, 4)], [np.float32, 4, (8, 4)]],
            [[np.int32, 0, (2, 3)], [np.int32, 0, (2, 3)]],
            [[np.int32, 2, (4, 3, 1)], [np.int32, 2, (4, 1, 5)]],
            [[np.int32, 2, (4, 3, 2)], [np.int32, 2, (4, 3, 2)]],
            [[np.int32, -1, (7, 3, 2)], [np.int32, -1, (7, 3, 2)]],
            [[np.int8, 0, (7, 3)], [np.int8, 0, (7, 3)]],
            [[np.int8, 2, (4, 3, 2)], [np.int8, 2, (4, 3, 2)]],
            [[np.uint8, 0, (3, 2)], [np.uint8, 0, (3, 2)]],
            [[np.uint8, 2, (4, 3, 2)], [np.uint8, 2, (4, 3, 2)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

        cpu_input1, npu_input1 = create_common_tensor(shape_format[11][0], 1, 100)
        cpu_input2 = cpu_input1
        npu_input2 = npu_input1
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_equal_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1, input2):
            output = torch.equal(input1, input2)
            output = torch.tensor(output)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        def npu_op_exec_fp16(input1, input2):
            output = torch.equal(input1, input2)
            output = torch.tensor(output).to("cpu")
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 0, (4, 3)], [np.float16, 0, (4, 3)]],
            [[np.float16, 29, (4, 3, 1)], [np.float16, 29, (4, 1, 5)]],
            [[np.float16, 2, (4, 3, 2)], [np.float16, 2, (4, 3, 2)]],
            [[np.float16, 3, (7, 3, 2)], [np.float16, 3, (7, 3, 2)]],
            [[np.float16, 4, (8, 4)], [np.float16, 4, (8, 4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = npu_op_exec_fp16(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

        cpu_input1, npu_input1 = create_common_tensor(shape_format[2][0], 2, 100)
        cpu_input2 = cpu_input1
        npu_input2 = npu_input1
        cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
        npu_output = npu_op_exec_fp16(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
