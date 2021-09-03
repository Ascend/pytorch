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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestGridSampler(TestCase):
    def cpu_op_exec(self, input1, input2, interpolation_mode = 0, padding_mode = 0, align_corners = True):
        output = torch.grid_sampler(input1, input2, interpolation_mode, padding_mode, align_corners)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, interpolation_mode = 0, padding_mode = 0, align_corners = True):
        output = torch.grid_sampler(input1, input2, interpolation_mode, padding_mode, align_corners)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_fp16_exec(self, input1, input2, interpolation_mode = 0, padding_mode = 0, align_corners = True):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.grid_sampler(input1, input2, interpolation_mode, padding_mode, align_corners)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def test_grid_sampler(self, device):
        shape_format = [
            [[[np.float32, -1, (100, 1, 28, 28)],[np.float32, -1, (100, 1, 1, 2)]],
            [[np.float32, -1, (100, 64, 32, 28)],[np.float32, -1, (100, 1, 1, 2)]],
            [[np.float32, -1, (2000, 1, 28, 28)],[np.float32, -1, (2000, 1, 1, 2)]]],
            [[[np.float16, -1, (1, 1, 3, 3)],[np.float16, -1, (1, 2, 2, 2)]],
            [[np.float16, -1, (1, 2, 3, 4)],[np.float16, -1, (1, 2, 2, 2)]]]
        ]
        for item in shape_format[0]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_output1 = self.cpu_op_exec(cpu_input1,cpu_input2)
            npu_output1 = self.npu_op_exec(npu_input1,npu_input2)
            cpu_output2 = self.cpu_op_exec(cpu_input1,cpu_input2, 0, 1, True)
            npu_output2 = self.npu_op_exec(npu_input1,npu_input2, 0, 1, True)
            cpu_output3 = self.cpu_op_exec(cpu_input1,cpu_input2, 0, 0, False)
            npu_output3 = self.npu_op_exec(npu_input1,npu_input2, 0, 0, False)
            cpu_output4 = self.cpu_op_exec(cpu_input1,cpu_input2, 1, 0, False)
            npu_output4 = self.npu_op_exec(npu_input1,npu_input2, 1, 0, False)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)
            self.assertRtolEqual(cpu_output4, npu_output4)

        for item in shape_format[1]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_output = self.cpu_op_fp16_exec(cpu_input1,cpu_input2)
            npu_output = self.npu_op_exec(npu_input1,npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestGridSampler, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()
