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

import os
import torch
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor, create_common_tensor_for_broadcast

os.environ["PTCOPY_ENABLE"] = "1"

# Optimized view Ops contains Transpose, permute, narrow, indexing, select, unfold 
class SingleViewCopyToContiguous(TestCase):
    def test_view_copy(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      #No padding for NZ format
                      [2, 3, 16, 16],
                      #Padding for NZ format  
                      [2, 3, 15, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # The key axis remains unchanged for NZ format
            npu_out1 = npu_input.view(1, 6, npu_input.size(2), npu_input.size(3)).clone()
            cpu_out1 = cpu_input.view(1, 6, cpu_input.size(2), cpu_input.size(3)).clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 

            # The key axis changes for NZ format
            npu_out2 = npu_input.view(1, 6, npu_input.size(2)*npu_input.size(3), 1).clone()
            cpu_out2 = cpu_input.view(1, 6, cpu_input.size(2)*cpu_input.size(3), 1).clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())        

    def test_unsqueeze_copy(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [2, 3, 16, 16], 
                      [2, 3, 15, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for i in range(4):
            for item in shape_format: 
                cpu_input, npu_input = create_common_tensor(item, 0, 100)
                npu_out = npu_input.unsqueeze(i).clone()
                cpu_out = cpu_input.unsqueeze(i).clone()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy()) 
    
    def test_flatten_copy(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [2, 3, 16, 16],
                      [2, 3, 16, 15],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            npu_out = torch.flatten(npu_input, 0, 1).clone()
            cpu_out = torch.flatten(cpu_input, 0, 1).clone()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())   
    
    def test_narrow_at_first_axis_copy(self, device):
        # this case: slice at the first dim, tensor with offset remains contiguous
        dtype_list = [np.float16, np.float32]
        format_list = [2, 3, 29]
        shape_list = [
                      [20, 16, 16], 
                      [20, 16, 15], 
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # contiguous and no offset
            npu_out1 = npu_input[:10,:,:].clone()
            cpu_out1 = cpu_input[:10,:,:].clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())  
            
            # contiguous but has offset
            npu_out2 = npu_input[1:10,:,:].clone()
            cpu_out2 = cpu_input[1:10,:,:].clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())       
                
instantiate_device_type_tests(SingleViewCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()