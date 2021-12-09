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

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization
os.environ["PTCOPY_ENABLE"] = "1"

# Note: NPU only support trans-contiguous with base format, so format_list uses -1
class CombinedViewsCopyToContiguous(TestCase):
    def test_view_permute_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [200, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+permute
            npu_out1 = npu_input.view(npu_input.size(0) * npu_input.size(1), npu_input.size(2), npu_input.size(3)).transpose(0, 1).contiguous()
            cpu_out1 = cpu_input.view(cpu_input.size(0) * cpu_input.size(1), cpu_input.size(2), cpu_input.size(3)).transpose(0, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+view
            npu_out2 = npu_input.permute(1, 0, 2, 3).view(npu_input.size(1), npu_input.size(0), npu_input.size(2)*npu_input.size(3)).contiguous()
            cpu_out2 = cpu_input.permute(1, 0, 2, 3).view(cpu_input.size(1), cpu_input.size(0), cpu_input.size(2)*cpu_input.size(3)).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 
    
    def test_view_select_copy_contiguous(self, device):
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
            # case 1: view+select
            npu_out1 = npu_input.view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)).select(2, 1).contiguous()
            cpu_out1 = cpu_input.view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)).select(2, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+view
            npu_out2 = npu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            cpu_out2 = cpu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_view_narrow_copy_contiguous(self, device):
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
            # case 1: view + narrow 
            npu_out1 = npu_input.view(20, 1200, 16)[:,20:150,:].contiguous()
            cpu_out1 = cpu_input.view(20, 1200, 16)[:,20:150,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow + view 
            npu_out2 = npu_input[:,10:19,:,:].view(20, 360, 16).contiguous()
            cpu_out2 = cpu_input[:,10:19,:,:].view(20, 360, 16).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_view_indexing_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 10],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view + indexing 
            npu_out1 = npu_input.view(20, 1200, 10)[:,20:150:3,:].contiguous()
            cpu_out1 = cpu_input.view(20, 1200, 10)[:,20:150:3,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: indexing + view 
            npu_out2 = npu_input[10:19:3,:,:].view(3, 2400, 5).contiguous()
            cpu_out2 = cpu_input[10:19:3,:,:].view(3, 2400, 5).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_squeeze_permute_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [2, 1, 3, 4],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze+permute
            npu_out1 = npu_input.squeeze(1).transpose(0,1).contiguous()
            cpu_out1 = cpu_input.squeeze(1).transpose(0,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+squeeze
            npu_out2 = npu_input.permute(1,0,2,3).squeeze(0).contiguous()
            cpu_out2 = cpu_input.permute(1,0,2,3).squeeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 
    
    def test_squeeze_narrow_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 30, 40, 16],
                      [20, 1, 30, 40]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + narrow 
            npu_out1 = npu_input.squeeze(1)[:,1:10,:].contiguous()
            cpu_out1 = cpu_input.squeeze(1)[:,1:10,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow + squeeze
            npu_out2 = npu_input[:,:,:,10:19].squeeze(1).contiguous()
            cpu_out2 = cpu_input[:,:,:,10:19].squeeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_squeeze_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze+select
            npu_out1 = npu_input.squeeze().select(2,1).contiguous()
            cpu_out1 = cpu_input.squeeze().select(2,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+squeeze
            npu_out2 = npu_input.select(2,1).squeeze().contiguous()
            cpu_out2 = cpu_input.select(2,1).squeeze().contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_squeeze_indexing_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 1, 200, 40, 10],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + indexing 
            npu_out1 = npu_input.squeeze(1)[:,20:150:3].contiguous()
            cpu_out1 = cpu_input.squeeze(1)[:,20:150:3].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: indexing + squeeze
            npu_out2 = npu_input[:,:,10:19:3].squeeze(1).contiguous()
            cpu_out2 = cpu_input[:,:,10:19:3].squeeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy()) 

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
            # case 1: unsqueeze+permute
            npu_out1 = npu_input.unsqueeze(1).transpose(0,1).contiguous()
            cpu_out1 = cpu_input.unsqueeze(1).transpose(0,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+unsqueeze
            npu_out2 = npu_input.permute(1,0,2,3).unsqueeze(0).contiguous()
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
            npu_out1 = npu_input.unsqueeze(0)[:,1:10].contiguous()
            cpu_out1 = cpu_input.unsqueeze(0)[:,1:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow+unsqueeze
            npu_out2 = npu_input[:,1:10].unsqueeze(2).contiguous()
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
            npu_out1 = npu_input.unsqueeze(0).select(2,1).contiguous()
            cpu_out1 = cpu_input.unsqueeze(0).select(2,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+unsqueeze
            npu_out2 = npu_input.select(1,1).unsqueeze(0).contiguous()
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
            # case 1: unsqueeze+unfold
            npu_out1 = npu_input.unsqueeze(1).unfold(0,2,2).contiguous()
            cpu_out1 = cpu_input.unsqueeze(1).unfold(0,2,2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: unfold+unsqueeze
            npu_out2 = npu_input.unfold(2,2,2).unsqueeze(1).contiguous()
            cpu_out2 = cpu_input.unfold(2,2,2).unsqueeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_unsqueeze_indexing_copy_contiguous(self, device):
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
            # case 1: squeeze + indexing 
            npu_out1 = npu_input.unsqueeze(1)[:,:,20:150:3].contiguous()
            cpu_out1 = cpu_input.unsqueeze(1)[:,:,20:150:3].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: indexing + squeeze
            npu_out2 = npu_input[:,:,10:19:3].unsqueeze(0).contiguous()
            cpu_out2 = cpu_input[:,:,10:19:3].unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

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
            npu_out1 = npu_input.flatten(2).select(1,1).contiguous()
            cpu_out1 = cpu_input.flatten(2).select(1,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+flatten
            npu_out2 = npu_input.select(2,1).flatten(1).contiguous()
            cpu_out2 = cpu_input.select(2,1).flatten(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
    
    def test_flatten_indexing_copy_contiguous(self, device):
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
            # case 1: flatten+indexing
            npu_out1 = npu_input.flatten()[2:100:10].contiguous()
            cpu_out1 = cpu_input.flatten()[2:100:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: indexing+flatten
            npu_out2 = npu_input[:,2:20:3].flatten().contiguous()
            cpu_out2 = cpu_input[:,2:20:3].flatten().contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_permute_narrow_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 50],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+narrow
            npu_out1 = npu_input.permute(1,3,2,0)[:10].contiguous()
            cpu_out1 = cpu_input.permute(1,3,2,0)[:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: narrow+permute
            npu_out2 = npu_input[:,1:10].permute(1,0,3,2).contiguous()
            cpu_out2 = cpu_input[:,1:10].permute(1,0,3,2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_permute_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 50],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+select
            npu_out1 = npu_input.permute(1,3,2,0).select(1,2).contiguous()
            cpu_out1 = cpu_input.permute(1,3,2,0).select(1,2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: select+permute
            npu_out2 = npu_input.select(1,0).permute(1,0,2).contiguous()
            cpu_out2 = cpu_input.select(1,0).permute(1,0,2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_permute_indexing_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
                      [20, 30, 40, 50],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+indexing-no offset
            npu_out1 = npu_input.permute(1,3,2,0)[::2].contiguous()
            cpu_out1 = cpu_input.permute(1,3,2,0)[::2].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: indexing+permute-with offset
            npu_out2 = npu_input[:,1:10:3].permute(1,3,0,2).contiguous()
            cpu_out2 = cpu_input[:,1:10:3].permute(1,3,0,2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_narrow_select_copy_contiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [
                      [20, 30, 40, 16],
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: narrow+select  
            # select the last dim
            npu_out1 = npu_input[:,2:4].select(3,1).contiguous()
            cpu_out1 = cpu_input[:,2:4].select(3,1).contiguous()
            # select the -2 dim
            npu_out2 = npu_input[:,2:4].select(2,2).contiguous()
            cpu_out2 = cpu_input[:,2:4].select(2,2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            # case 2: select+narrow
            # select the last dim
            npu_out3 = npu_input.select(3,1)[:,1:2].contiguous()
            cpu_out3 = cpu_input.select(3,1)[:,1:2].contiguous()
            # select the -2 dim
            npu_out4 = npu_input.select(0,2)[:,1:2].contiguous()
            cpu_out4 = cpu_input.select(0,2)[:,1:2].contiguous()
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

    def test_narrow_indexing_copy_contiguous(self, device):
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
            # case 1: narrow+indexing 
            # slice at adjacent axes
            npu_out1 = npu_input[2:4,::2].contiguous()
            cpu_out1 = cpu_input[2:4,::2].contiguous()
            # slice at non-adjacent axes
            npu_out2 = npu_input[:,2:4,:,1:10:2].contiguous()
            cpu_out2 = cpu_input[:,2:4,:,1:10:2].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            
            # case 2: indexing+narrow
            # slice at adjacent axes
            npu_out3 = npu_input[1:10:2,1:10].contiguous()
            cpu_out3 = cpu_input[1:10:2,1:10].contiguous()
            # slice at non-adjacent axes
            npu_out4 = npu_input[::2,:,1:10].contiguous()
            cpu_out4 = cpu_input[::2,:,1:10].contiguous()
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

    def test_indexing_select_contiguous(self, device):
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
            # case 1: indexing+select
            npu_out1 = npu_input[:10:2].select(3,1).contiguous()
            cpu_out1 = cpu_input[:10:2].select(3,1).contiguous()
            npu_out2 = npu_input[1:10:2].select(2,1).contiguous()
            cpu_out2 = cpu_input[1:10:2].select(2,1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            # case 2: select+indexing
            npu_out3 = npu_input.select(3,1)[:,1:10:2].contiguous()
            cpu_out3 = cpu_input.select(3,1)[:,1:10:2].contiguous()
            npu_out4 = npu_input.select(2,1)[:,::3].contiguous()
            cpu_out4 = cpu_input.select(2,1)[:,::3].contiguous()
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())
    
    def test_broadcast_permute_contiguous(self, device):
        dtype_list = [np.float16, np.float32, np.int32, np.int8, np.uint8]
        format_list = [-1]
        shape_list = [
                    [[2, 1, 3],    [1, 2, 4, 3]],
                    [[2, 1, 3],    [5, 2, 4, 3]],
                    ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
            ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor_for_broadcast(item, 0, 100)
            npu_out1 = npu_input.expand(item[2][1]).transpose(1,3).contiguous()
            cpu_out1 = cpu_input.expand(item[2][1]).transpose(1,3).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
         
instantiate_device_type_tests(CombinedViewsCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()