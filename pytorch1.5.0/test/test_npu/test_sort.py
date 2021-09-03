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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestSort(TestCase):  
    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        output, indice = torch.sort(input1)
        return output, indice

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output, indice = torch.sort(input1)
        output = output.to("cpu")
        indice = indice.to("cpu")
        return output,indice

    def test_sort_float(self, device):
        npu_input1= self.generate_data(0, 100, [4, 5], np.float32)
        cpu_output,cpu_indice = self.cpu_op_exec(npu_input1)
        npu_output,npu_indice = self.npu_op_exec(npu_input1)
        cpu_indice = cpu_indice.type_as(torch.tensor([],dtype=torch.int32))
        self.assertRtolEqual(cpu_indice, npu_indice)

instantiate_device_type_tests(TestSort, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
    