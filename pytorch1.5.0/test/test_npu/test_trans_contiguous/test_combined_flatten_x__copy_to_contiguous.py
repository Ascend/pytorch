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
class CombinedFlattenXCopyToContiguous(TestCase):
    def test_flatten_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: flatten+select
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.flatten(2).select(1,1).contiguous()
            self.assertEqual(check_operators_in_prof(['npuMatch', 'select_npuStridedSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.flatten(2).select(1,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+flatten == can be optimized as single select(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input.select(2,1).flatten(1).contiguous()
            self.assertEqual(check_operators_in_prof(['select_npuStridedSlice'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out2 = cpu_input.select(2,1).flatten(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_flatten_strideslice_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: flatten+strideslice ==> can be optimized as slice(contiguous with offset) + select
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out1 = npu_input.flatten()[2:100:10].contiguous()
            self.assertEqual(check_operators_in_prof(['View_d2dCopyAsync', 'select_npuStridedSlice'], prof), True, "Error operators called!")
            cpu_out1 = cpu_input.flatten()[2:100:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice+flatten==> can be optimized as single strideslice(npuCombined should not be called)
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                npu_out2 = npu_input[:,2:20:3].flatten().contiguous()
            self.assertEqual(check_operators_in_prof(['npuStridedSlice'], prof, ['npuCombined']), True, "Error operators called!")
            cpu_out2 = cpu_input[:,2:20:3].flatten().contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

         
instantiate_device_type_tests(CombinedFlattenXCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()