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

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization

# Note: NPU only support trans-contiguous with base format, so format_list uses -1
class CombinedUnsqueezeXCopyToContiguous(TestCase):
    def test_unsqueeze_permute_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [2, 3, 4, 5],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+permute ==> can be optimized as single permute(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.unsqueeze(1).transpose(2,3).contiguous()
            self.assertEqual(check_operators_in_prof(['npuTranspose'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(1).transpose(2,3).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+unsqueeze ==> can be optimized as single permute(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.permute(1,0,2,3).unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['npuTranspose'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out2 = cpu_input.permute(1,0,2,3).unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_narrow_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      #3D-->4D-->3D
                      [20, 30, 40], 
                      #4D-->5D-->4D test memory allocation
                      [2, 300, 400, 500], 
                      #5D-->6D-->5D
                      [20, 30, 40, 50, 60]  
                     ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+narrow
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.unsqueeze(0)[:,:,1:10].contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(0)[:,:,1:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow+unsqueeze
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,1:10].unsqueeze(2).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input[:,1:10].unsqueeze(2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [2, 3, 4],
                      [2, 300, 400, 500],
                      [2, 3, 4, 5, 6]
                     ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+select
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.unsqueeze(0).select(2,1).contiguous()
            cpu_out1 = cpu_input.unsqueeze(0).select(2,1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+unsqueeze
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.select(1,1).unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'narrow_npuSlice'], prof), True, "Error operators called!")
            cpu_out2 = cpu_input.select(1,1).unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
     
    def test_unsqueeze_unfold_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [4, 2, 4],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
    
        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+unfold:size==step ==> can be optimized as reshape+permute
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.unsqueeze(1).unfold(0,2,2).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'npuTranspose'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(1).unfold(0,2,2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: unfold+unsqueeze: size!=step ==> cannot be optimized(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.unfold(2,2,3).unsqueeze(1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuAsStrided'], prof, ['npuCombined']), True, "Error operators called!") 
            cpu_out2 = cpu_input.unfold(2,2,3).unsqueeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_unsqueeze_strideslice_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 200, 40],
                      [20, 200, 40, 10]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + strideslice ==> cannot be optimized(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.unsqueeze(1)[:,:,20:150:3].contiguous()
            self.assertEqual(check_operators_in_prof(['npuAsStrided'], prof, ['npuCombined']), True, "Error operators called!") 
            cpu_out1 = cpu_input.unsqueeze(1)[:,:,20:150:3].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice + squeeze ==> cannot be optimized(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,:,10:19:3].unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['npuAsStrided'], prof, ['npuCombined']), True, "Error operators called!") 
            cpu_out2 = cpu_input[:,:,10:19:3].unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

         
instantiate_device_type_tests(CombinedUnsqueezeXCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()