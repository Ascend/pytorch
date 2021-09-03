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
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import time

class TestTopk(TestCase):
    def cpu_op_exec(self, input1, k, dim, largest, sorted1):
        output, index = torch.topk(input1, k, dim, largest, sorted1)
        output = output.numpy()
        index = index.numpy()
        return output, index
    
    def npu_op_exec(self, input1, k, dim, largest, sorted1):
        output, index = torch.topk(input1, k, dim, largest, sorted1)
        output = output.to("cpu")
        index = index.to("cpu")
        output = output.numpy()
        index = index.numpy()
        return output, index
    
    def test_topk_shape_format(self, device):
        np.random.seed(0)
        shape_format = [
            # [k, dim, [input_type, input_format, input_shape, min, max], largest, sorted]
            # dim
            [3, 0, [np.float32, 0, [8, 10], 0, 100], True, True],
            [3, 1, [np.float32, 0, [8, 10], 0, 100], True, True],
            [5, 2, [np.float32, 0, [8, 9, 10], 0, 1000], True, True],
            [5, 3, [np.float32, 0, [8, 9, 10, 11], 0, 1000], True, True],
            # dtype
            [3, 0, [np.int32, 0, [8, 10], 0, 100], True, True],
            [5, 2, [np.int32, 0, [8, 9, 10], 0, 1000], True, True],
            # random
            [5, 0, [np.float32, 0, [31, 47], 0, 100], True, True],
            [5, 1, [np.float32, 0, [42, 53, 7], 0, 100], True, True],
            # largest
            [3, 0, [np.float32, 0, [8, 10], 0, 100], False, True],
        ]

        cnt = 0
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[2], item[2][3], item[2][4])
            cpu_output, cpu_index = self.cpu_op_exec(cpu_input, item[0], item[1], item[3], item[4])
            npu_output, npu_index = self.npu_op_exec(npu_input, item[0], item[1], item[3], item[4])
            # 目前只支持fp16,fp32降低阈值判断
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)
            #self.assertRtolEqual(cpu_index, npu_index)
            cnt += 1
    
    def test_topk_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, k, dim, largest, sorted1):
            input1 = input1.to(torch.float32)
            output, index = torch.topk(input1, k, dim, largest, sorted1)
            output = output.numpy().astype(np.float16)
            index = index.numpy().astype(np.int32)
            return output, index

        np.random.seed(0)
        shape_format = [
            # [k, dim, [input_type, input_format, input_shape, min, max], largest, sorted]
            # dim
            [3, 0, [np.float16, 0, [8, 10], 0, 100], True, True],
            [3, 1, [np.float16, 0, [8, 10], 0, 100], True, True],
            [5, 2, [np.float16, 0, [8, 9, 10], 0, 1000], True, True],
            [5, 3, [np.float16, 0, [8, 9, 10, 11], 0, 1000], True, True],
        ]

        cnt = 0
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[2], item[2][3], item[2][4])
            cpu_output, cpu_index = cpu_op_exec_fp16(cpu_input, item[0], item[1], item[3], item[4])
            npu_output, npu_index = self.npu_op_exec(npu_input, item[0], item[1], item[3], item[4])
            cpu_index = cpu_index.astype(npu_index.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            #self.assertRtolEqual(cpu_index, npu_index)
            cnt += 1

instantiate_device_type_tests(TestTopk, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()