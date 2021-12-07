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
    def test_narrow_copy_contiguous(self, device):
        # AssertionError: required dtype in [np.bool, np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]
        # However, considering the dtypes that Transdata supports, only np.float16, np.float32 are tested.
        dtype_list = [np.float16, np.float32]
        format_list_4D = [0, 3, 29, 4]
        shape_list_4D = [[2, 32, 16, 20]]
        format_list_5D = [30, 32, 33]
        shape_list_5D = [[2, 32, 16, 20, 15]]
        shape_format_4D = [
            [i, j, k] for i in dtype_list for j in format_list_4D for k in shape_list_4D
        ]
        shape_format_5D = [
            [i, j, k] for i in dtype_list for j in format_list_5D for k in shape_list_5D
        ]
        shape_format = shape_format_4D + shape_format_5D

        for item in shape_format:    
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # for narrow with step=1, if narrow at the first axis, it will generate a contiguous tensor
            npu_out1 = npu_input[:,:16,:,:].contiguous()
            npu_out2 = npu_input[:,:,1:16,:].contiguous()
            npu_out3 = npu_input[:,:,:,2:16].contiguous()

            cpu_out1 = cpu_input[:,:16,:,:].contiguous()
            cpu_out2 = cpu_input[:,:,1:16,:].contiguous()
            cpu_out3 = cpu_input[:,:,:,2:16].contiguous()
            if npu_input.dim() == 5:
                cpu_out4 = cpu_input[:,:,:,:,3:10].contiguous()
                npu_out4 = npu_input[:,:,:,:,3:10].contiguous()
                self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())

    def test_indexing_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32, np.int8, np.int32, np.uint8, np.bool]
        format_list = [-1]
        shape_list = [[10,32,16,9], [10,32,16,9,10]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:    
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # for indexing with step>1 -- StridedSlice
            npu_out1 = npu_input[::2].contiguous()
            npu_out2 = npu_input[:,1:17:4].contiguous()
            npu_out3 = npu_input[:,:,2:16:5].contiguous()
            npu_out4 = npu_input[:,:,:,3:9:2].contiguous()
            npu_out5 = npu_input[::2,1:17:4,2:16:5,3:9:2].contiguous()

            cpu_out1 = cpu_input[::2].contiguous()
            cpu_out2 = cpu_input[:,1:17:4].contiguous()
            cpu_out3 = cpu_input[:,:,2:16:5].contiguous()
            cpu_out4 = cpu_input[:,:,:,3:9:2].contiguous()
            #indexing at each axis
            cpu_out5 = cpu_input[::2,1:17:4,2:16:5,3:9:2].contiguous()
            
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy()) 
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy()) 
            self.assertRtolEqual(npu_out5.to("cpu").numpy(), cpu_out5.numpy()) 
            if cpu_input.dim() == 5:
                cpu_out6 = cpu_input[:,:,:,:,1:7:3].contiguous()
                npu_out6 = npu_input[:,:,:,:,1:7:3].contiguous()
                self.assertRtolEqual(npu_out6.to("cpu").numpy(), cpu_out6.numpy()) 
    
    def test_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [[2,32,16,9], [2,32,16,9,10]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for dim in range(1,len(item[2])):
                npu_out = npu_input.select(dim,1).contiguous()
                cpu_out = cpu_input.select(dim,1).contiguous()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())             
                
instantiate_device_type_tests(SingleViewCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()