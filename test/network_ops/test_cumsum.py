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


class TestCumsum(TestCase):
    def cpu_op_exec(self, input1, dim):
        output = torch.cumsum(input1, dim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, dtype):
        if dtype is None:
            output = torch.cumsum(input1, dim)
        else:
            output = torch.cumsum(input1, dim, dtype=dtype)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_fp16_exec(self, input1, dim):
        input1 = input1.to(torch.float32)
        output = torch.cumsum(input1, dim)
        output = output.numpy().astype(np.float16)
        return output

    def npu_op_out_exec(self, input1, output, dim):
        torch.cumsum(input1, dim, out=output)
        output = output.to("cpu").numpy()
        return output

    def test_cumsum_common_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 2, 3, 4)], torch.float32],
            [[np.float32, 0, (2, 3, 4)], None],
            [[np.float32, 0, (3, 4)], None],
            [[np.float16, 0, (1, 2, 3, 4)], torch.float16],
            [[np.float16, 0, (2, 3, 4)], None],
            [[np.float16, 0, (3, 4)], None],
            [[np.int32, 0, (1, 2, 3, 4)], torch.int32],
            [[np.int32, 0, (2, 3, 4)], None],
            [[np.int32, 0, (3, 4)], None],
        ]
        dim = 0
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 4)

            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_exec(npu_input1, dim, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cumsum_out_common_shape_format(self):
        shape_format = [
            [[[np.float32, 0, (1, 2, 3, 4)], [np.float32, 0, (1, 2, 3, 4)]],
             [[np.float32, 0, (2, 3, 4)], [np.float32, 0, (2, 3, 4)]],
             [[np.float32, 0, (3, 4)], [np.float32, 0, (3, 4)]],
             [[np.float32, 0, (3, 4)], [np.float32, 0, (3, 5)]],
             [[np.float32, 0, (3, 4)], [np.float32, 0, (3, 4, 1)]],
             [[np.float32, 0, (3, 4)], [np.float32, 0, (4, 3)]]],
            [[[np.float16, 0, (1, 2, 3, 4)], [np.float16, 0, (1, 2, 3, 4)]],
             [[np.float16, 0, (2, 3, 4)], [np.float16, 0, (2, 3, 4)]],
             [[np.float16, 0, (3, 4)], [np.float16, 0, (3, 4)]],
             [[np.float16, 0, (3, 4)], [np.float16, 0, (3, 5)]],
             [[np.float16, 0, (3, 4)], [np.float16, 0, (4, 3)]],
             [[np.float16, 0, (3, 4)], [np.float16, 0, (3, 4, 1)]]],
        ]
        dim = 0
        for item in shape_format[0]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 4)
            _, npu_input2 = create_common_tensor(item[1], 1, 4)
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, dim)
            self.assertEqual(cpu_output.shape, npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output)

        for item in shape_format[1]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 4)
            _, npu_input2 = create_common_tensor(item[1], 1, 4)
            cpu_output = self.cpu_op_fp16_exec(cpu_input1, dim)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, dim)
            self.assertEqual(cpu_output.shape, npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
