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


class TestIndex(TestCase):
    def cpu_op_exec(self, input1, dim, index):
        output = torch.index_select(input1, dim, index)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, index):
        index = index.to("npu")
        output = torch.index_select(input1, dim, index)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_index_shape_format_fp32(self, device):
        format_list = [-1]
        shape_list = [(1000, 1280), (32, 3, 3), (1024, 464, 7, 7)]
        shape_format = [
            [[np.float32, i, j], [np.int64, 0, [2]]] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            dim = np.random.randint(0, len(item[0][2]))
            index1 = np.random.uniform(0, item[0][2][dim], item[1][2]).astype(np.int64)
            index = torch.from_numpy(index1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, dim, index)
            npu_output = self.npu_op_exec(npu_input1, dim, index)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_fp16(self, device):
        format_list = [-1]
        shape_list = [(1000, 1280), (32, 3, 3), (1024, 464, 7, 7)]
        shape_format = [
            [[np.float16, i, j], [np.int64, 0, [2]]] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            dim = np.random.randint(0, len(item[0][2]))
            index1 = np.random.uniform(0, item[0][2][dim], item[1][2]).astype(np.int64)
            index = torch.from_numpy(index1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, dim, index)
            npu_output = self.npu_op_exec(npu_input1, dim, index)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestIndex, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
