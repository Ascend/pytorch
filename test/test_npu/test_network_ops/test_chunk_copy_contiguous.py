# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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
import time

class TestChunkToContiguous(TestCase):
    def cpu_op_exec(self, input, chunk, dim):
        outputs = torch.chunk(input, chunk, dim)
        output_list = []
        for i in range(len(outputs)):
            output_list.append(outputs[i].contiguous())
        output_tensor = torch.cat(output_list)
        return output_tensor.numpy()

    def npu_op_exec(self, input, chunk, dim):
        outputs = torch.chunk(input, chunk, dim)
        output_list = []
        for i in range(len(outputs)):
            output_list.append(outputs[i].contiguous().to("cpu"))
        output_tensor = torch.cat(output_list)
        return output_tensor.numpy()

    def test_chunk_to_contiguous(self, device):
        data_type_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [[2, 6, 6]]
        shape_format = [
            [i,j,k] for i in data_type_list for j in format_list for k in shape_list
        ]
        cpu_time = 0.
        npu_time = 0.
        for item in shape_format:
            for dim in range(0, len(item[-1])):
                cpu_input, npu_input = create_common_tensor(item, 1, 100)
                cpu_start = time.time()
                cpu_output = self.cpu_op_exec(cpu_input, 2, dim)
                cpu_end = time.time()
                npu_start = time.time()
                npu_output = self.npu_op_exec(npu_input, 2, dim)
                npu_end = time.time()
                cpu_time += cpu_end - cpu_start
                npu_time += npu_end - npu_start
                self.assertRtolEqual(cpu_output, npu_output)
    
        self.assertTrue(npu_time < 30, f"execute time:{npu_time:.2f}s should be less than 30s")
        print(f"chunk to contiguous use: {cpu_time:.5f} s (CPU)")
        print(f"chunk to contiguous use: {npu_time:.5f} s (NPU)")
        print("TBE Ops used: NpuSlice")

instantiate_device_type_tests(TestChunkToContiguous, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
