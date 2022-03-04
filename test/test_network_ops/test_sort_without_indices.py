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
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSortWithoutIndices(TestCase):
    def cpu_default_op_exec(self, input1):
        output, _ = torch.sort(input1)
        output = output.to(torch.float16).numpy()
        return output
    
    def npu_default_op_exec(self, input1):
        output = torch_npu.npu_sort_v2(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_exec(self, input1, descending):
        output, _ = torch.sort(input1, descending=descending)
        output = output.to(torch.float16).numpy()
        return output
    
    def npu_op_exec(self, input1, descending):
        output = torch_npu.npu_sort_v2(input1, descending=descending)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def create_shape_format(self):
        dtype_list = [np.float16]
        format_list = [0]
        shape_list = [(1, 5000), (1, 50000), (1, 289600), (1, 409600)]
        descend_list = [True, False]

        shape_format = [[[i, j, k], h] for i in dtype_list
                        for j in format_list for k in shape_list for h in descend_list]
        return shape_format

    def test_sort_v2_shape_format(self):
        for item in self.create_shape_format():
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) == 1:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float))
                npu_output = self.npu_default_op_exec(npu_input1)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float), item[1])
                npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_sort_v2_shape_format_big_range(self, device="npu"):
        for item in self.create_shape_format():
            cpu_input1, npu_input1 = create_common_tensor(item[0], -60000, 60000)
            if len(item) == 1:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float))
                npu_output = self.npu_default_op_exec(npu_input1)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float), item[1])
                npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()