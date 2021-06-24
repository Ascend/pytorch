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


class TestSelectToContiguous(TestCase):
    def test_SelectToContiguous(self, device):
        dtype_list = [ np.float16 ,np.float32 ]
        format_list = [0,3,29]
        shape_list = [[6, 9, 4]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        start = time.time()

        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            for dim in range(1,len(item[2])):
                npu_out = a1_npu.unfold(dim,2,2).contiguous()
                cpu_out = a1_cpu.unfold(dim,2,2).contiguous()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())                
        
        end = time.time()   
        print("Unfold to contiguous uses: %.2f s (CPU+NPU, Typical time required: 10-13s)" %(end-start)) 
        print("TBE Ops used: TransposeD; SliceD(If dimmension is not divisible)")   
                
instantiate_device_type_tests(TestSelectToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()