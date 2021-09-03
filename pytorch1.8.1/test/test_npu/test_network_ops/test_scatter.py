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
import numpy as np
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestScatter(TestCase):
    def cpu_op_exec(self, shape, dim, index, src):
        input = torch.zeros(shape)
        cpu_output = input.scatter(dim, index, src)

        return cpu_output.numpy()

    def npu_op_exec(self, shape, dim, index, src, isTensor = True):
        input = torch.zeros(shape).npu()
        index = index.npu()
        if (isTensor) :
            src = src.npu()
        npu_output = input.scatter(dim, index, src)
        npu_output = npu_output.cpu()

        return npu_output.numpy()

    def cpu_op_exec_inplace(self, shape, dim, index, src):
        input = torch.zeros(shape)
        input.scatter_(dim, index, src)

        return input.numpy()

    def npu_op_exec_inplace(self, shape, dim, index, src, isTensor = True):
        input = torch.zeros(shape).npu()
        index = index.npu()
        if (isTensor) :
            src = src.npu()
        input.scatter_(dim, index, src)
        input = input.cpu()

        return input.numpy()

    def test_scatter_shape_format(self, device):
        shape_format = [
                [0, [3,5], [np.float32, 0, [2,5]]],
                [0, [3,5], [np.float32, 3, [2,5]]],
                [1, [3,5], [np.int32, 0, [2,5]]],
                [-1, [3,5], [np.int32, 0, [2,5]]],
                [1, [3,5], [np.float16, 0, [2,5]]],
                [-1, [3,5], [np.float16, 0, [2,5]]],
        ]

        index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[2], 1, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(item[1], item[0], index, cpu_input)
            npu_output = self.npu_op_exec(item[1], item[0], index, npu_input)

            if npu_output.dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec(item[1], item[0], index, 1.23)
            npu_output = self.npu_op_exec(item[1], item[0], index, 1.23, False)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec_inplace(item[1], item[0], index, cpu_input)
            npu_output = self.npu_op_exec_inplace(item[1], item[0], index, npu_input)

            if npu_output.dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec_inplace(item[1], item[0], index, 1.23)
            npu_output = self.npu_op_exec_inplace(item[1], item[0], index, 1.23, False)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestScatter, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
