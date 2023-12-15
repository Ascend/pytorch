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
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaxPool2dWithIndices(TestCase):

    def cpu_op_exec(self, input1, kernel_size, stride, padding, dilation, ceil_mode):
        output, _ = F.max_pool2d_with_indices(input1, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation,
                                              ceil_mode=ceil_mode, return_indices=True)
        return output

    def npu_op_exec(self, input1, kernel_size, stride, padding, dilation, ceil_mode):
        output, _ = F.max_pool2d_with_indices(input1, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation,
                                              ceil_mode=ceil_mode, return_indices=True)
        return output.cpu()

    def test_max_pool2d_with_indices(self):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 4]
        shape_list = [[256, 64, 112, 112], [1024, 24, 112, 112],
                      [1024, 24, 56, 112], [1024, 24, 112, 56],
                      [1234, 48, 56], [26, 16, 43]]
        shape_format = [
            [[i, j, k], [3, 3], [2, 2], 1, 1, False] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])
            if item[0][0] == np.float16:
                cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
