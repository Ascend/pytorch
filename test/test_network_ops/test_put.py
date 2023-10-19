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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestPut(TestCase):

    def cpu_op_exec(self, input_x, index, source, accumulate):
        input_cpu = copy.deepcopy(input_x)
        if input_cpu.dtype == torch.float16:
            input_cpu = input_cpu.to(torch.float32)
            source = source.to(torch.float32)
        index = index.to("cpu").long()
        source = source.to("cpu")
        output = input_cpu.put_(index, source, accumulate).to(input_x.dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_x, index, source, accumulate):
        input_x = input_x.to("npu")
        index = index.to("npu")
        source = source.to("npu")
        output = input_x.put_(index, source, accumulate)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def get_result(self, shape_format):
        for item in shape_format:
            maxVal = 1
            for i in item[0][2]:
                maxVal = maxVal * i
            accumulate = True
            if np.random.randint(0, 999) % 2 == 1:
                accumulate = False

            input_x_cpu, input_x_npu = create_common_tensor(item[0], 1, 100)
            index_cpu, index_npu = create_common_tensor(item[1], 0, maxVal)
            source_cpu, source_npu = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_exec(input_x_cpu, index_cpu, source_cpu, accumulate)
            npu_output = self.npu_op_exec(input_x_npu, index_npu, source_npu, accumulate)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_put_aicpu_common_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4, 3)], [np.int64, -1, (4, 1)], [np.float32, -1, (4)]],
            [[np.float32, -1, (4, 3, 5)], [np.int64, -1, (4, 2)], [np.float32, -1, (4, 2)]],
            [[np.float32, -1, (5, 6, 4, 3)], [np.int64, -1, (8, 2)], [np.float32, -1, (8, 2)]],
            [[np.float32, -1, (8, 9, 5, 4, 3, 10)], [np.int64, -1, (9, 1)], [np.float32, -1, (9)]],
            [[np.float32, -1, (5, 5, 5, 6, 4, 3)], [np.int64, -1, (2, 4, 5)], [np.float32, -1, (2, 4, 5)]],
            [[np.float32, -1, (6, 9, 10, 2, 5, 7, 3)], [np.int64, -1, (3, 4, 5)], [np.float32, -1, (3, 4, 5)]]
        ]
        self.get_result(shape_format)

    def test_put_aicore_common_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4, 3)], [np.int32, -1, (4, 1)], [np.float32, -1, (4)]],
            [[np.float32, -1, (4, 3, 5)], [np.int32, -1, (4, 2)], [np.float32, -1, (4, 2)]],
            [[np.float32, -1, (5, 6, 4, 3)], [np.int32, -1, (8, 2)], [np.float32, -1, (8, 2)]],
            [[np.float32, -1, (8, 9, 5, 4, 3, 10)], [np.int32, -1, (9, 1)], [np.float32, -1, (9)]],
            [[np.float32, -1, (5, 5, 5, 6, 4, 3)], [np.int32, -1, (2, 4, 5)], [np.float32, -1, (2, 4, 5)]],
            [[np.float32, -1, (6, 9, 10, 2, 5, 7, 3)], [np.int32, -1, (3, 4, 5)], [np.float32, -1, (3, 4, 5)]]
        ]
        self.get_result(shape_format)

    def test_put_aicpu_common_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, -1, (4, 3)], [np.int64, -1, (4, 1)], [np.float16, -1, (4)]],
            [[np.float16, -1, (4, 3, 5)], [np.int64, -1, (4, 2)], [np.float16, -1, (4, 2)]],
            [[np.float16, -1, (5, 6, 4, 3)], [np.int64, -1, (8, 2)], [np.float16, -1, (8, 2)]],
            [[np.int32, -1, (2, 4, 3, 8)], [np.int64, -1, (10)], [np.int32, -1, (5, 2)]],
            [[np.int32, -1, (9, 3, 4, 3, 9)], [np.int64, -1, (10, 1)], [np.int32, -1, (10)]],
            [[np.float16, -1, (8, 9, 5, 4, 3, 10)], [np.int64, -1, (9, 1)], [np.float16, -1, (9)]],
            [[np.float16, -1, (5, 5, 5, 6, 4, 3)], [np.int64, -1, (2, 4, 5)], [np.float16, -1, (2, 4, 5)]],
            [[np.float16, -1, (6, 9, 10, 2, 5, 7, 3)], [np.int64, -1, (3, 4, 5)], [np.float16, -1, (3, 4, 5)]]
        ]
        self.get_result(shape_format)

    def test_put_aicore_common_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, -1, (4, 3)], [np.int32, -1, (4, 1)], [np.float16, -1, (4)]],
            [[np.float16, -1, (4, 3, 5)], [np.int32, -1, (4, 2)], [np.float16, -1, (4, 2)]],
            [[np.float16, -1, (5, 6, 4, 3)], [np.int32, -1, (8, 2)], [np.float16, -1, (8, 2)]],
            [[np.float16, -1, (8, 9, 5, 4, 3, 10)], [np.int32, -1, (9, 1)], [np.float16, -1, (9)]],
            [[np.float16, -1, (5, 5, 5, 6, 4, 3)], [np.int32, -1, (2, 4, 5)], [np.float16, -1, (2, 4, 5)]],
            [[np.float16, -1, (6, 9, 10, 2, 5, 7, 3)], [np.int32, -1, (3, 4, 5)], [np.float16, -1, (3, 4, 5)]]
        ]
        self.get_result(shape_format)

    def test_put_empty_shape(self, device="npu"):
        shape_format = [
            [np.float32, -1, (4, 3)],
            [np.float32, -1, (4, 3, 5)],
            [np.float16, -1, (5, 6, 4, 3)],
        ]
        for item in shape_format:
            maxVal = 1
            for i in item[2]:
                maxVal = maxVal * i
            accumulate = True
            if np.random.randint(0, 999) % 2 == 1:
                accumulate = False

            input_x_cpu, input_x_npu = create_common_tensor(item, 1, 100)
            index_cpu = torch.tensor([], dtype=torch.int64)
            index_npu = index_cpu.npu()
            source_cpu = torch.tensor([])
            source_npu = source_cpu.npu()
            cpu_output = self.cpu_op_exec(input_x_cpu, index_cpu, source_cpu, accumulate)
            npu_output = self.npu_op_exec(input_x_npu, index_npu, source_npu, accumulate)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
