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

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class ReshapeCopy(TestCase):
    def test_NZ_view_copy(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [2, 3, 16, 16], #No padding for NZ format 
                      [2, 3, 15, 16], #Padding for NZ format
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            output_npu = input_npu.view(1, 6, input_npu.size(2), input_npu.size(3)).clone()
            output_cpu = input_cpu.view(1, 6, input_cpu.size(2), input_cpu.size(3)).clone()

            self.assertRtolEqual(output_npu.to("cpu").numpy(), output_cpu.numpy())       

    def test_NZ_unsqueeze_copy(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [2, 3, 16, 16], #No padding for NZ format 
                      [2, 3, 15, 16], #Padding for NZ format
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for i in range(4):
            for item in shape_format: 
                input_cpu, input_npu = create_common_tensor(item, 0, 100)
                output_npu = input_npu.unsqueeze(i).clone()
                output_cpu = input_cpu.unsqueeze(i).clone()

                self.assertRtolEqual(output_npu.to("cpu").numpy(), output_cpu.numpy()) 
    
    def test_NZ_flatten_copy(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [2, 3, 16, 16], #No padding for NZ format 
                      [2, 3, 16, 15], #Padding for NZ format
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            output_npu = torch.flatten(input_npu, 0, 1).clone()
            output_cpu = torch.flatten(input_cpu, 0, 1).clone()

            self.assertRtolEqual(output_npu.to("cpu").numpy(), output_cpu.numpy())   
    
    def test_slice_copy(self, device):
        # this case: slice at the first dim, tensor with offset remains contiguous, which can be directly copied
        dtype_list = [np.float16 ,np.float32]
        format_list = [2, 3, 29]
        shape_list = [
                      [20, 16, 16], 
                      [20, 16, 15], 
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            output_npu = input_npu[:10,:,:].clone()
            output_cpu = input_cpu[:10,:,:].clone()

            self.assertRtolEqual(output_npu.to("cpu").numpy(), output_cpu.numpy())  
                
instantiate_device_type_tests(ReshapeCopy, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()