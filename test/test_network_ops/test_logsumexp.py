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


class TestLogsumexp(TestCase):

    def generate_data(self, min1, max1, shape, dtype):
        x = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_x = torch.from_numpy(x)
        return npu_x

    def cpu_op_exec(self, input1, dim, keepdim):
        output = torch.logsumexp(input1, dim, keepdim=keepdim)
        return output

    def npu_op_exec(self, input1, dim, keepdim):
        output = torch.logsumexp(input1, dim, keepdim=keepdim)
        output = output.to("cpu")
        return output

    def cpu_op_out_exec(self, input1, dim, out, keepdim):
        torch.logsumexp(input1, dim, keepdim=keepdim, out=out)
        return out

    def npu_op_out_exec(self, input1, dim, out, keepdim):
        torch.logsumexp(input1, dim, keepdim=keepdim, out=out)
        output = out.to("cpu")
        return output

    def test_logsumexp_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (3, 4, 2)], [np.float32, 0, (3, 4, 1)], 2, True],
            [[np.float32, 0, (3, 4, 2)], [np.float32, 0, (3, 4)], 2, False],
            [[np.float32, 0, (3, 4, 2)], [np.float32, 0, (3,)], [1, 2], False],
            [[np.float32, 0, (2, 3, 4, 2)], [np.float32, 0, (2, 3, 1, 2)], 2, True],
            [[np.float32, 0, (2, 3, 4, 2)], [np.float32, 0, (2, 3, 2)], 2, False],
            [[np.float32, 0, (2, 3, 4, 2)], [np.float32, 0, (2, 3)], [2, 3], False],
            [[np.float16, 0, (3, 4, 2)], [np.float16, 0, (3, 4, 1)], 2, True],
            [[np.float16, 0, (3, 4, 2)], [np.float16, 0, (3, 4)], 2, False],
            [[np.float16, 0, (3, 4, 2)], [np.float16, 0, (3,)], [1, 2], False],
            [[np.float16, 0, (2, 3, 4, 2)], [np.float16, 0, (2, 3, 1, 2)], 2, True],
            [[np.float16, 0, (2, 3, 4, 2)], [np.float16, 0, (2, 3, 2)], 2, False],
            [[np.float16, 0, (2, 3, 4, 2)], [np.float16, 0, (2, 3)], [2, 3], False]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_out, npu_out = create_common_tensor(item[1], 1, 10)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            if cpu_out.dtype == torch.float16:
                cpu_out = cpu_out.to(torch.float32)
            cpu_out_result = self.cpu_op_out_exec(cpu_input, item[2], cpu_out, item[3])
            npu_out_result = self.npu_op_out_exec(npu_input, item[2], npu_out, item[3])
            cpu_out_result = cpu_out_result.to(npu_out_result.dtype)
            self.assertRtolEqual(cpu_out_result.numpy(), npu_out_result.numpy())

            cpu_result = self.cpu_op_exec(cpu_input, item[2], item[3])
            npu_result = self.npu_op_exec(npu_input, item[2], item[3])
            cpu_result = cpu_result.to(npu_result.dtype)
            self.assertRtolEqual(cpu_result.numpy(), npu_result.numpy())

    def test_logsumexp_dimname1(self, device="npu"):
        cpu_input = self.generate_data(-10, 10, (2, 14, 69, 96, 1824), np.float32)
        cpu_input.names = ['A', 'B', 'C', 'D', 'E']
        dim = ['C']
        keepdim = True
        cpu_out = self.cpu_op_exec(cpu_input, dim, keepdim)
        npu_out = self.npu_op_exec(cpu_input.npu(), dim, keepdim)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_logsumexp_dimname2(self, device="npu"):
        cpu_input = self.generate_data(-10, 10, (14, 69, 96, 1824), np.float32)
        cpu_input.names = ['A', 'B', 'C', 'D']
        dim = ['B', 'C']
        keepdim = False
        cpu_out = self.cpu_op_exec(cpu_input, dim, keepdim)
        npu_out = self.npu_op_exec(cpu_input.npu(), dim, keepdim)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_logsumexp_dimname3(self, device="npu"):
        cpu_input = self.generate_data(-10, 10, (14, 69, 96, 1824), np.float32)
        cpu_input.names = ['A', 'B', 'C', 'D']
        dim = ['B', 'C', 'D']
        keepdim = False
        cpu_out = self.cpu_op_exec(cpu_input, dim, keepdim)
        npu_out = self.npu_op_exec(cpu_input.npu(), dim, keepdim)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())


if __name__ == "__main__":
    run_tests()
