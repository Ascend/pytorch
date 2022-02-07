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
import os
import torch
import numpy as np
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests

os.environ["DYNAMIC_COMPILE_ENABLE"] = "1"
os.environ["TASK_QUEUE_ENABLE"] = "1"


class TestDynamicMulWithQueEnable(TestCase):
    def create_random_shape_tensor(self, item, minValue, maxValue):
        ftype = item[0]
        dtype = item[1]
        dim = item[2]
        shape_num = item[3]
        npu_tensor_list = []
        cpu_tensor_list = []
        for i in range(shape_num):
            shape = np.random.randint(1, 100, dim)
            op_input = np.random.uniform(minValue, maxValue, shape).astype(dtype)
            cpu_input = torch.from_numpy(op_input)
            npu_input = torch.from_numpy(op_input).npu()
            if ftype not in (-1, 0):
                npu_input = npu_input.npu_format_cast(ftype)
            cpu_tensor_list.append(cpu_input)
            npu_tensor_list.append(npu_input)
        return cpu_tensor_list, npu_tensor_list

    def cpu_exec(self, tensor_list):
        outputs = []
        for i in tensor_list:
            if i.dtype == torch.float16:
                i = i.float()
            out = torch.mul(i, 1)
            outputs.append(out)
        return outputs

    def npu_dynamic_exec(self, tensor_list, npu_outpus):
        torch.npu.synchronize()
        old = time.time()
        for i in tensor_list:
            out = torch.mul(i, 1)
            npu_outpus.append(out)
        torch.npu.synchronize()
        now = time.time()
        return now - old, npu_outpus

    def test_mul_all(self, device):
        format_list = [0, 3]
        dtype_list = [np.float32, np.float16]
        dim_list = [1, 2]
        items = [
            [i, j, k, 200] for i in format_list for j in dtype_list for k in dim_list
        ]
        torch.npu.increase_step()
        torch.npu.increase_step()
        for item in items:
            cpu_tensor_list, npu_tensor_list = self.create_random_shape_tensor(item, -100, 100)
            cpu_outputs = self.cpu_exec(cpu_tensor_list)
            npu_outputs = []
            dynamic_time, npu_outputs = self.npu_dynamic_exec(npu_tensor_list, npu_outputs)
            print("format: {}, dtype: {}, dim: {}, number: {}".format(item[0], item[1], item[2], item[3]))
            print("dynamic time: {:.6f}ms:".format(dynamic_time * 1000))
            
            for i in range(len(npu_tensor_list)):
                cpu_output = cpu_outputs[i]
                npu_output = npu_outputs[i]
                
                self.assertRtolEqual(cpu_output.to(npu_output.dtype).numpy(), npu_output.cpu().numpy())


instantiate_device_type_tests(TestDynamicMulWithQueEnable, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
