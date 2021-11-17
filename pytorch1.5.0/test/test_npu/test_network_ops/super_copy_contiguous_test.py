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

def create_common_tensor_new(item, minValue, maxValue):
    dtype = item[0]
    format = item[1]
    shape = item[2]
    input1 = np.random.uniform(minValue, maxValue, shape[0]).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    npu_input = torch.from_numpy(input1).to("npu")
    if format != -1:
        npu_input = npu_input.npu_format_cast(format)
    return cpu_input, npu_input

# Optimized view Ops contains Transpose, permute, narrow, indexing, select, unfold 
class SuperContiguous(TestCase):
    def test_BroadcastToContiguous(self, device):
        dtype_list = [np.float16 ,np.float32, np.int32, np.int8, np.uint8]
        format_list = [0]
        shape_list = [
                    [[1],          [5]],
                    [[1, 2],       [3, 2]],
                    [[1, 2, 1],    [1, 2, 3]],
                    [[1, 2, 1, 3], [4, 2, 5, 3]],
                    [[2, 3, 4],    [1, 2, 3, 4]],
                    [[2, 3],       [1, 1, 2, 3]],
                    [[1, 3],       [1, 1, 4, 3]],
                    [[1, 3],       [2, 1, 4, 3]],
                    [[1, 3],       [1, 2, 4, 3]],
                    [[3, 1],       [2, 1, 3, 1]],
                    [[3, 1],       [1, 2, 3, 1]],
                    ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        broadcast_time = 0
        broadcast_time_exper = 30

        for item in shape_format: 
            a1_cpu, a1_npu = create_common_tensor_new(item, 0, 100)
            broadcast_start = time.time()
            npu_out1 = a1_npu.expand(item[2][1]).contiguous()
            broadcast_end = time.time()
            broadcast_time += (broadcast_end - broadcast_start)

            cpu_out1 = a1_cpu.expand(item[2][1]).contiguous()

            npu_out1 = npu_out1.to("cpu");
            self.assertEqual(npu_out1.size(), cpu_out1.size())
            self.assertRtolEqual(npu_out1.numpy(), cpu_out1.numpy())

        print("------------------------Broadcast---------------------------") 
        print("Broadcast to contiguous uses: %.2f s " %(broadcast_time)) 
        print("Typical time required: 25-30s, Ops: broadcastToD")
        self.assertTrue(broadcast_time < broadcast_time_exper)

    def test_BroadcastAndTransposeToContiguous(self, device):
        dtype_list = [np.float16 ,np.float32, np.int32, np.int8, np.uint8]
        format_list = [0]
        shape_list = [
                    [[2, 1, 3],    [1, 2, 4, 3]],
                    [[2, 1, 3],    [5, 2, 4, 3]],
                    ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
            ]

        for item in shape_format:
            a1_cpu, a1_npu = create_common_tensor_new(item, 0, 100)
            npu_out1 = a1_npu.expand(item[2][1]).transpose(1,3).contiguous()
            cpu_out1 = a1_cpu.expand(item[2][1]).transpose(1,3).contiguous()

            npu_out1 = npu_out1.to("cpu");
            self.assertEqual(npu_out1.size(), cpu_out1.size())
            self.assertRtolEqual(npu_out1.numpy(), cpu_out1.numpy())
        print("------------------------Broadcast&Transpose---------------------------")

    def test_PermuteToContiguous(self, device):
        dtype_list = [np.bool, np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]
        format_list = [0]
        shape_list = [[2, 6, 9, 4]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        permute_time = 0
        permute_time_exper = 5

        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            permute_start = time.time()
            npu_out1 = a1_npu.permute(1,0,2,3).contiguous()
            npu_out2 = a1_npu.permute(2,3,0,1).contiguous()
            permute_end = time.time()
            permute_time += (permute_end - permute_start)

            cpu_out1 = a1_cpu.permute(1,0,2,3).contiguous()
            cpu_out2 = a1_cpu.permute(2,3,0,1).contiguous()

            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())                  
  
        print("------------------------Permute---------------------------") 
        print("Permute to contiguous uses: %.2f s " %(permute_time)) 
        print("Typical time required: 2-5s, Ops: TransposeD")
        self.assertTrue(permute_time < permute_time_exper)
    
    def test_NarrowToContiguous(self, device):
        # AssertionError: required dtype in [np.bool, np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]
        # However, considering the dtypes that Transdata supports, only np.float16, np.float32 are tested.
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29, 4]
        shape_list = [[2, 32, 16, 9]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        narrow_time = 0
        narrow_time_exper = 3
        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            # for narrow with step=1 -- SliceD
            narrow_start = time.time()
            npu_out1 = a1_npu[:,:16,:,:].contiguous()
            npu_out2 = a1_npu[:,:,:16,:].contiguous()
            narrow_end = time.time()
            narrow_time += (narrow_end - narrow_start)

            cpu_out1 = a1_cpu[:,:16,:,:].contiguous()
            cpu_out2 = a1_cpu[:,:,:16,:].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

        print("------------------------Narrow---------------------------") 
        print("Narrow to contiguous uses: %.2f s"%(narrow_time))
        print("Typical time required: 1-3s, Ops: SliceD")
        self.assertTrue(narrow_time < narrow_time_exper)


    def test_IndexingToContiguous(self, device):
        dtype_list = [np.float16, np.float32, np.int8, np.int32, np.uint8, np.bool]
        format_list = [0]
        shape_list = [[10,32,16,9]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        indexing_time = 0
        indexing_time_exper = 4
        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            # for indexing with step>1 -- StridedSliceD
            indexing_start = time.time()
            npu_out1 = a1_npu[::2,::4,1:16:5,:].contiguous()
            indexing_end = time.time()
            indexing_time += (indexing_end - indexing_start)

            cpu_out1 = a1_cpu[::2,::4,1:16:5,:].contiguous()

            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy()) 
        print("------------------------Indexing---------------------------") 
        print("Indexing to contiguous uses: %.2f s"%(indexing_time))
        print("Typical time required: 1-4s, Ops: StridedSliceD")
        self.assertTrue(indexing_time < indexing_time_exper)
    
    def test_SelectToContiguous(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29, 4]
        shape_list = [[2,32,16,9]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        select_time = 0
        select_time_exper = 22
        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            for dim in range(1,len(item[2])):
                select_start = time.time()
                npu_out = a1_npu.select(dim,1).contiguous()
                select_end = time.time()
                select_time += (select_end - select_start)
                cpu_out = a1_cpu.select(dim,1).contiguous()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())  
        print("------------------------Select---------------------------")
        print("Select to contiguous uses: %.2f s "%(select_time))
        print("Typical time required: 18-22s, Ops: SliceD")
        self.assertTrue(select_time < select_time_exper)   
    
    def test_UnfoldToContiguous(self, device):
        dtype_list = [np.float16, np.float32, np.int8, np.int32, np.uint8, np.bool]
        format_list = [0]
        shape_list = [[6, 9, 4]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        unfold_time = 0
        unfold_time_exper = 5

        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            for dim in range(1, len(item[2]) - 1):
                unfold_start = time.time()
                npu_out = a1_npu.unfold(dim,3,3).contiguous()
                unfold_end = time.time()
                unfold_time += (unfold_end - unfold_start)

                cpu_out = a1_cpu.unfold(dim,3,3).contiguous()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())                
        print("------------------------Unfold---------------------------")
        print("Unfold to contiguous uses: %.2f s " %(unfold_time)) 
        print("Typical time required: 2-5s, Ops: TransposeD [optional:SliceD]")  
        self.assertTrue(unfold_time < unfold_time_exper)
                
instantiate_device_type_tests(SuperContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()