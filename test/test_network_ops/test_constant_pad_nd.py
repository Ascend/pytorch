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


class TestConstantPadNd(TestCase):
    def op_exec_cpu(self, input1, pad_shape, value=0):
        output = torch.constant_pad_nd(input1, pad_shape, value=value)
        output = output.numpy()
        return output

    def op_exec_npu(self, input1, pad_shape, value=0):
        input1 = input1.to("npu")
        output = torch.constant_pad_nd(input1, pad_shape, value=value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def constant_pad_nd_shape_format(self, shape_format):
        for item in shape_format:
            value = item[-1] if len(item) > 2 else 0
            input_cpu, input_npu = create_common_tensor(item[0], 1, 1)
            pad_shape = item[1]
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            cpu_output = self.op_exec_cpu(input_cpu, pad_shape, value=value)
            npu_output = self.op_exec_npu(input_npu, pad_shape, value=value)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_constant_pad_nd_shape_with_value(self):
        # Note: only fp16 suppport fill any float value currently!
        dtype_list = [np.float16]
        format_list = [0]
        pad_list = [(1, 2, 2, 2), (1, 2)]
        shape_list = [(16, 128), (1, 2, 16, 128)]
        value_list = [0.5, 1.47]
        shape_format = [
            [[i, j, k], l, m] for i in dtype_list
            for j in format_list
            for k in shape_list
            for l in pad_list
            for m in value_list
        ]
        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_with_3_value(self):
        # Note: only fp16 suppport fill any float value currently!
        dtype_list = [np.float16]
        format_list = [3]
        pad_list = [(1, 2, 2, 2), (1, 2)]
        shape_list = [(1, 2, 16, 128)]
        value_list = [0.5, 1.47]
        shape_format = [
            [[i, j, k], l, m] for i in dtype_list
            for j in format_list
            for k in shape_list
            for l in pad_list
            for m in value_list
        ]
        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_1d(self):
        dtype_list = [np.float16, np.float32]
        format_list = [0]
        pad_list = [(1, 2)]
        shape_format = [
            [[i, j, [18]], k] for i in dtype_list for j in format_list for k in pad_list
        ]

        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_nd(self):
        dtype_list = [np.float16, np.float32]
        format_list = [0]
        pad_list = [(1, 2, 2, 2), (1, 2)]
        shape_list = [(16, 128), (2, 16, 128), (1, 2, 16, 128)]
        shape_format = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in pad_list
        ]

        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_1_nd(self):
        dtype_list = [np.float16, np.float32]
        format_list = [3]
        pad_list = [(1, 2, 2, 2), (1, 2)]
        shape_list = [(1, 2, 16, 128)]
        shape_format = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in pad_list
        ]

        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_nd_int32(self):
        dtype_list = [np.int32]
        format_list = [0]
        pad_list = [(1, 2, 2, 2), (1, 2)]
        shape_list = [(16, 128), (2, 16, 128), (1, 2, 16, 128)]
        shape_format = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in pad_list
        ]

        self.constant_pad_nd_shape_format(shape_format)


if __name__ == "__main__":
    run_tests()
