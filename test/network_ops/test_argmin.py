# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
# coding: utf-8

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestArgmin(TestCase):
    @Dtypes(torch.float, torch.half)
    def test_argmin(self, device, dtype):
        inputValues = [-1000, -1, 0, 0.5, 1, 2, 1000]
        expectedOutput = [0.0000, 0.2689, 0.5, 0.6225, 0.7311, 0.8808, 1.000]
        precision_4dps = 0.0002
        a = torch.tensor(inputValues, dtype=dtype, device=device)
        self.assertRtolEqual(torch.tensor(inputValues, dtype=dtype, device=device).sigmoid().cpu(),
                             torch.tensor(expectedOutput, dtype=dtype, device=device).cpu(),
                             precision_4dps)

    def cpu_op_exec(self, input1, dims, keepdim=False):
        output = torch.argmin(input1, dim=dims, keepdim=keepdim)
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dims, keepdim=False):
        output = torch.argmin(input1, dim=dims, keepdim=keepdim)
        output = output.to("cpu")
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def test_argmin_shape_format(self):
        shape_format = [
            [[np.float32, 0, (6, 4)], 0, False],
            [[np.float32, 2, (6, 4)], 1, True],
            [[np.float32, 0, (2, 4, 5)], 2, True],
            [[np.float32, 0, (1, 2, 3, 3)], 2, False],
            [[np.float32, 0, (1, 2, 3, 3)], 3, True],
            [[np.float16, 0, (6, 4)], 0, False],
            [[np.float16, 2, (6, 4)], 1, True],
            [[np.float16, 0, (2, 4, 5)], 2, True],
            [[np.float16, 3, (1, 2, 3, 3)], 2, False],
            [[np.float16, 0, (1, 2, 3, 3)], 3, True],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], keepdim=item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], keepdim=item[2])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
