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

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor, check_operators_in_prof

os.environ["PTCOPY_ENABLE"] = "1"

# Optimized view Ops contains Transpose, permute, narrow, strideslice, select, unfold 
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
            # Directly using d2dcopy without transdata(View_d2dCopyAsync)
            # case1. base format
            match_case1 = (item[1] == 0)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.view(1, 6, npu_input.size(2), npu_input.size(3)).clone()
            # case2. The key axis remains unchanged for NZ format
            match_case2 = (item[1] == 29)
            if match_case1 or match_case2:
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof), True, "d2dCopyAsync is not called!")
            cpu_out1 = cpu_input.view(1, 6, cpu_input.size(2), cpu_input.size(3)).clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 

            # The key axis changes for NZ format
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.view(1, 6, npu_input.size(2)*npu_input.size(3), 1).clone()
            if match_case1:
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof), True, "d2dCopyAsync is not called!")
            cpu_out2 = cpu_input.view(1, 6, cpu_input.size(2)*cpu_input.size(3), 1).clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())        

    def test_unsqueeze_copy(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [2, 3, 29]
        shape_list = [
                      [3, 16, 16], 
                      [3, 15, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for i in range(3):
            for item in shape_format: 
                cpu_input, npu_input = create_common_tensor(item, 0, 100)
                # Directly using d2dcopy without transdata(View_d2dCopyAsync)
                # case1. base format 
                match_case1 = (item[1] == 2)
                # case2. The key axis remains unchanged for NZ format
                match_case2 = (item[1] == 29 and i < 2)
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    npu_out = npu_input.unsqueeze(i).clone()
                if match_case1 or match_case2:
                    self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
                else:
                    self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof), True, "d2dCopyAsync is not called!")
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
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out = torch.flatten(npu_input, 0, 1).clone()
            if item[1] == 3:
                # Using d2dcopy with transdata(d2dCopyAsync)
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof), True, "d2dCopyAsync is not called!")
            else:
                # Directly using d2dcopy without transdata(View_d2dCopyAsync)
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            
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
            # Directly using d2dcopy without transdata(View_d2dCopyAsync)
            # The key axis remains unchanged for NZ format in all cases.
            # case1. base format 
            match_case1 = (item[1] == 2)
            # case2. NZ format but no padding
            match_case2 = (item[1] == 29 and item[2] == [20, 16, 16])

            # contiguous and no offset
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input[:10,:,:].clone()
            # case3. NZ format with padding but no offset
            match_case3 = (item[1] == 29 and True)
            if match_case1 or match_case2 or match_case3:
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['narrow_npuSlice'], prof), True, "narrow_npuSlice is not called!")
            cpu_out1 = cpu_input[:10,:,:].clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())  
            
            # contiguous but has offset
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[1:10,:,:].clone()
            match_case3 = False
            if match_case1 or match_case2 or match_case3:
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['narrow_npuSlice'], prof), True, "narrow_npuSlice is not called!")
            cpu_out2 = cpu_input[1:10,:,:].clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_select_at_first_axis_to_single_element_tensor_copy(self, device):
        dtype_list = [torch.float32]
        format_list = [2, 3, 29]
        shape_format = [
            [i, j] for i in dtype_list for j in format_list
        ]
        
        for item in shape_format: 
            cpu_input = torch.tensor([1.0]).to(item[0])
            npu_input = cpu_input.npu().npu_format_cast(item[1])

            match_case = (item[1] == 2)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input[0].clone()
            if match_case:
                self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync'], prof), True, "View_d2dCopyAsync is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['narrow_npuSlice'], prof), True, "narrow_npuSlice is not called!")
            cpu_out1 = cpu_input[0].clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[0] + 1
            if match_case:
                self.assertEqual(check_operators_in_prof(['memory_repoint'], prof), True, "memory_repoint is not called!")
            else:
                # refresh storage desc after transdata
                self.assertEqual(check_operators_in_prof(['Identity'], prof), True, "Identity is not called!")
            cpu_out2 = cpu_input[0] + 1
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

                
instantiate_device_type_tests(SingleViewCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()