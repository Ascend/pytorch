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

class TestMaskedScatter(TestCase):
    def cpu_op_exec(self, input, maskbool, source):
        cpu_output = torch.masked_scatter(input, maskbool, source)
        return cpu_output.numpy()

    def npu_op_exec(self, input, maskbool, source):
        input = input.to("npu")
        maskbool = maskbool.to("npu")
        source = source.to("npu")
        npu_output = torch.masked_scatter(input, maskbool, source)
        npu_output = npu_output.to("cpu")
        return npu_output.numpy()

    def cpu_inp_op_exec(self, input, maskbool, source):
        cpu_output = input.masked_scatter_(maskbool, source)
        return cpu_output.numpy()

    def npu_inp_op_exec(self, input, maskbool, source):
        maskbool = maskbool.to("npu")
        npu_output = input.masked_scatter_(maskbool, source)
        npu_output = npu_output.to("cpu")
        return npu_output.numpy()

    def test_masked_scatter_float(self, device):
        dtype_list = [np.float32]
        format_list = [0, 3]
        shape_list = [[4, 5],[3, 4, 5], [2, 3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = torch.randn(4, 1)
        maskbool = mask.ge(0.5)
        
        for item in shape_format:
          print(item)
          cpu_input, npu_input = create_common_tensor(item, 0, 100)
          cpu_source, npu_source = create_common_tensor(item, 0, 100)
          cpu_output2 = self.cpu_inp_op_exec(cpu_input, maskbool, cpu_source)
          npu_output2 = self.npu_inp_op_exec(npu_input, maskbool, npu_source)
          self.assertRtolEqual(cpu_output2, npu_output2)
          
    def test_masked_scatter_int(self, device):
        dtype_list = [np.int32, np.int64]
        format_list = [0]
        shape_list = [[4, 5],[3, 4, 5], [2, 3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = torch.randn(4, 1)
        maskbool = mask.ge(0.5)
        
        for item in shape_format:
          print(item)
          cpu_input, npu_input = create_common_tensor(item, 0, 100)
          cpu_source, npu_source = create_common_tensor(item, 0, 100)
          cpu_output2 = self.cpu_inp_op_exec(cpu_input, maskbool, cpu_source)
          npu_output2 = self.npu_inp_op_exec(npu_input, maskbool, npu_source)
          self.assertRtolEqual(cpu_output2, npu_output2)

instantiate_device_type_tests(TestMaskedScatter, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()