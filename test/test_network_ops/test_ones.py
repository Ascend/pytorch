# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import torch
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestOnes(TestCase):
    def cpu_op_exec(self, shape, dtype):
        output = torch.ones(size=shape, dtype=dtype)
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, shape, dtype):
        output = torch.ones(size=shape, device='npu', dtype=dtype)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def cpu_op_name_exec(self, shape, name, dtype):
        output = torch.ones(size=shape, names=name, dtype=dtype)
        output = output.detach().numpy()
        return output

    def npu_op_name_exec(self, shape, name, dtype):
        output = torch.ones(size=shape, names=name, device='npu', dtype=dtype)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def cpu_op_out_exec(self, shape, output, dtype):
        torch.ones(size=shape, dtype=dtype, out=output)
        return output

    def npu_op_out_exec(self, shape, output, dtype):
        torch.ones(size=shape, dtype=dtype, device='npu', out=output)
        output = output.to("cpu")
        return output

    def test_ones_format(self, device="npu"):
        shape_format = [
            [(2, 3, 4, 1, 5), torch.float32],
            [(1, 100, 7), torch.int32],
            [(10, 1, 7), torch.int8],
            [(1, 2, 7), torch.uint8],
            [(33, 44, 55), torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1])
            npu_output = self.npu_op_exec(item[0], item[1])

            self.assertRtolEqual(cpu_output, npu_output)

    def test_ones_out_format(self, device="npu"):
        shape_format = [
            [(2, 3, 4, 1, 5), torch.float32],
            [(1, 100, 7), torch.int32],
            [(10, 1, 7), torch.int8],
            [(1, 2, 7), torch.uint8],
            [(33, 44, 55), torch.float16],
        ]
        for item in shape_format:
            cpu_out = torch.randn(item[0], dtype=torch.float32)
            cpu_out = cpu_out.to(item[1])
            npu_out = cpu_out.to('npu')
            cpu_output = self.cpu_op_out_exec(item[0], cpu_out, item[1])
            npu_output = self.npu_op_out_exec(item[0], npu_out, item[1])

            self.assertRtolEqual(cpu_output, npu_output)

    def test_ones_name_format(self, device="npu"):
        shape_format = [
            [(2, 3, 4, 1, 5), ('A', 'B', 'C', 'D', 'E'), torch.float32],
            [(1, 100, 7), ('C', 'H', 'W'), torch.int32],
            [(10, 1, 7), ('C', 'H', 'W'), torch.int8],
            [(1, 2, 7), ('C', 'H', 'W'), torch.uint8],
            [(33, 44, 55), ('C', 'H', 'W'), torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_name_exec(item[0], item[1], item[2])
            npu_output = self.npu_op_name_exec(item[0], item[1], item[2])

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
