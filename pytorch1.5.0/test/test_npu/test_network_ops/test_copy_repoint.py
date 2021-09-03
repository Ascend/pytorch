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

import time

import torch
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class CopyRepoint(TestCase):
    def test_copy_repoint(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [2, 3, 29]
        shape_list = [
                      [4, 16, 16, 512],
                      [4, 16, 15, 512]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            output_npu = input_npu.unsqueeze(0) * 1
            output_cpu = input_cpu.unsqueeze(0)
            self.assertRtolEqual(output_npu.to("cpu").numpy(), output_cpu.numpy())
                
instantiate_device_type_tests(CopyRepoint, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()