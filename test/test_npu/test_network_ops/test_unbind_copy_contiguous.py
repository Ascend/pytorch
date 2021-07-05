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
import time

class TestUnbindToContiguous(TestCase):

    def cpu_op_exec(self, input1, dim):
        output_tuple= torch.unbind(input1, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1.append(output_tuple[i].contiguous())
        output = torch.cat(listtuple1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim):
        output_tuple = torch.unbind(input1, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1.append(output_tuple[i].contiguous())
        output = torch.cat(listtuple1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_unbind_common_shape_format(self, device):
        data_type_list = [np.float16, np.float32, np.uint8,np.int8,np.int64]
        format_list = [0]
        shape_list = [[1, 4, 2, 3], [3, 2 ,3]]
        shape_format = [
            [i,j,k] for i in data_type_list for j in format_list for k in shape_list
        ]
        cpu_time = 0.
        npu_time = 0.
        for item in shape_format:
            dim = np.random.randint(0, len(item[-1]))
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_start = time.time()
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            cpu_end = time.time()
            npu_start = time.time()
            npu_output = self.npu_op_exec(npu_input1, dim)
            npu_end = time.time()
            cpu_time += cpu_end - cpu_start
            npu_time += npu_end - npu_start
            self.assertRtolEqual(cpu_output, npu_output)
        print(f"unbind to contiguous use: {cpu_time:.5f} s (CPU)")
        print(f"unbind to contiguous use: {npu_time:.5f} s (NPU)")
        print(f"TBE Ops used: Slice")
    
instantiate_device_type_tests(TestUnbindToContiguous, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
