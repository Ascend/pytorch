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

class CombinedContiguous(TestCase):
    def test_view_permute_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: view+permute
            output_npu1 = input_npu.view(input_npu.size(0) * input_npu.size(1), input_npu.size(2), input_npu.size(3)).transpose(0, 1).clone()
            output_cpu1 = input_cpu.view(input_cpu.size(0) * input_cpu.size(1), input_cpu.size(2), input_cpu.size(3)).transpose(0, 1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

            # case 2: permute+view
            output_npu2 = input_npu.permute(1,0,2,3).view(input_npu.size(1), input_npu.size(0), input_npu.size(2)*input_npu.size(3)).clone()
            output_cpu2 = input_cpu.permute(1,0,2,3).view(input_cpu.size(1), input_cpu.size(0), input_cpu.size(2)*input_cpu.size(3)).clone()
            self.assertRtolEqual(output_npu2.to("cpu").numpy(), output_cpu2.numpy()) 
    
    def test_unsqueeze_unfold_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [4,2,4],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
    
        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+unfold
            output_npu1 = input_npu.unsqueeze(1).unfold(0,2,2).clone()
            output_cpu1 = input_cpu.unsqueeze(1).unfold(0,2,2).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

            # case 2: unfold+unsqueeze
            output_npu2 = input_npu.unfold(2,2,2).unsqueeze(1).clone()
            output_cpu2 = input_cpu.unfold(2,2,2).unsqueeze(1).clone()
            self.assertRtolEqual(output_npu2.to("cpu").numpy(), output_cpu2.numpy())
    
    def test_view_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                     ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: view+select
            output_npu1 = input_npu.view(input_npu.size(0), input_npu.size(1) * input_npu.size(2), input_npu.size(3)).select(2,1).clone()
            output_cpu1 = input_cpu.view(input_npu.size(0), input_npu.size(1) * input_npu.size(2), input_npu.size(3)).select(2,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+view
            output_npu1 = input_npu.select(2,1).view(input_npu.size(1),input_npu.size(0),-1).clone()
            output_cpu1 = input_cpu.select(2,1).view(input_npu.size(1),input_npu.size(0),-1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

    def test_squeeze_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 1, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: squeeze+select
            output_npu1 = input_npu.squeeze().select(2,1).clone()
            output_cpu1 = input_cpu.squeeze().select(2,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+squeeze
            output_npu1 = input_npu.select(2,1).squeeze().clone()
            output_cpu1 = input_cpu.select(2,1).squeeze().clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

    def test_flatten_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: flatten+select
            output_npu1 = input_npu.flatten(2).select(1,1).clone()
            output_cpu1 = input_cpu.flatten(2).select(1,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+flatten
            output_npu1 = input_npu.select(2,1).flatten(1).clone()
            output_cpu1 = input_cpu.select(2,1).flatten(1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

    def test_slice_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: slice+select  slice_dim should be the last one
            output_npu1 = input_npu[:,2:4].select(3,1).clone()
            output_cpu1 = input_cpu[:,2:4].select(3,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+slice  slice_dim should be the last one
            output_npu1 = input_npu.select(3,1)[:,1:2].clone()
            output_cpu1 = input_cpu.select(3,1)[:,1:2].clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

    def test_indexing_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: indexing+select  indexing_dim+1 should be less than select_dim
            output_npu1 = input_npu[:4:2].select(3,1).clone()
            output_cpu1 = input_cpu[:4:2].select(3,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+indexing   indexing_dim+1 should be less than select_dim
            output_npu1 = input_npu.select(3,1)[:4:2].clone()
            output_cpu1 = input_cpu.select(3,1)[:4:2].clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
    
    def test_unsqueeze_select_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+select  
            output_npu1 = input_npu.unsqueeze(0).select(2,1).clone()
            output_cpu1 = input_cpu.unsqueeze(0).select(2,1).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: select+unsqueeze   
            output_npu1 = input_npu.select(2,1).unsqueeze(0).clone()
            output_cpu1 = input_cpu.select(2,1).unsqueeze(0).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
    
    def test_view_slice_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: view + slice 
            output_npu1 = input_npu.view(20,1200,16)[:,20:150,:].clone()
            output_cpu1 = input_cpu.view(20,1200,16)[:,20:150,:].clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: slice + view 
            output_npu1 = input_npu[:,10:19,:,:].view(20,360,16).clone()
            output_cpu1 = input_cpu[:,10:19,:,:].view(20,360,16).clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            
    def test_squeeze_indexing_contiguous(self, device):
        dtype_list = [np.float16 ,np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [3, 1, 10, 5],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            input_cpu, input_npu = create_common_tensor(item, 0, 100)
            # case 1: squeeze + indexing can not be optimize, check if come to optimize_slice branch 
            output_npu1 = input_npu.squeeze(1)[:,1:6:3,:].clone()
            output_cpu1 = input_cpu.squeeze(1)[:,1:6:3,:].clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())
            # case 2: squeeze + indexing can be optimize because covered by slice  
            output_npu1 = input_npu.squeeze(1)[:,1:6:2,:].clone()
            output_cpu1 = input_cpu.squeeze(1)[:,1:6:2,:].clone()
            self.assertRtolEqual(output_npu1.to("cpu").numpy(), output_cpu1.numpy())

         
instantiate_device_type_tests(CombinedContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()