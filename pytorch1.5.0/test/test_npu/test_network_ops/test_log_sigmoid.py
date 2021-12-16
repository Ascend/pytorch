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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestLogsigmoid(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.nn.functional.logsigmoid(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.nn.functional.logsigmoid(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, out):
        output = torch.nn.functional.logsigmoid(input1, out=out)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, out):
        output = torch.nn.functional.logsigmoid(input1, out=out)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_log_sigmoid_shape_format(self, device):
        shape_format = [
            [[np.float32, 0, (6, 4)]],
            [[np.float32, 3, (2, 4, 5)]],
            [[np.float32, 4, (1, 2, 3, 3)]],
            [[np.float32, 29, (11, 22, 33, 43)]],
            [[np.float32, 2, (2, 11, 51, 8, 3)]],
            [[np.float32, 2, (2, 11, 51, 8, 3, 8)]],
            [[np.float32, 2, (2, 11, 51, 8, 20, 12, 6)]],
            [[np.float32, 2, (2, 11, 51, 8, 3, 2, 4, 7)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -50, 50)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_out, npu_out = create_common_tensor(item[0], -50, 50)
            cpu_output = self.cpu_op_exec_out(cpu_input, cpu_out)
            cpu_output = self.npu_op_exec_out(npu_input, npu_out)
            self.assertRtolEqual(cpu_output, cpu_output)

    def test_log_sigmoid_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.nn.functional.logsigmoid(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        def cpu_op_exec_fp16_out(input1, out):
            input1 = input1.to(torch.float32)
            out = out.to(torch.float32)
            output = torch.nn.functional.logsigmoid(input1, out=out)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 0, (6, 4)]],
            [[np.float16, 3, (2, 4, 5)]],
            [[np.float16, 4, (1, 2, 3, 3)]],
            [[np.float16, 29, (10, 22, 33, 33)]],
            [[np.float16, 2, (2, 11, 51, 8, 3)]],
            [[np.float16, 2, (2, 11, 51, 8, 3, 8)]],
            [[np.float16, 2, (2, 11, 51, 8, 20, 12, 6)]],
            [[np.float16, 2, (2, 11, 51, 8, 3, 2, 4, 7)]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -50, 50)
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_out, npu_out = create_common_tensor(item[0], -50, 50)
            cpu_out = cpu_op_exec_fp16_out(cpu_input1, cpu_out)
            npu_out = self.npu_op_exec_out(npu_input1, npu_out)
            self.assertRtolEqual(cpu_out, npu_out)


instantiate_device_type_tests(TestLogsigmoid, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
