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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests
import time
import os
import copy
# Need export DYNAMIC_COMPILE_ENABLE=1 and export EXPERIMENTAL_DYNAMIC_PARTITION=1


class BroadcastToNet(torch.nn.Module):
    def __init__(self):
        super(BroadcastToNet, self).__init__()

    def forward(self, x, size):
        out = x.npu_broadcast(size)
        return out


class TestShape(TestCase):
    def create_random_shape_tensor(self, item, min_value, max_value):
        npu_format = item[0]
        dtype = item[1]
        dim = item[2]
        shape = np.random.randint(1, 10, dim)
        input_tensor = np.random.uniform(min_value, max_value, shape).astype(dtype)
        cpu_input = torch.from_numpy(input_tensor)
        npu_input = torch.from_numpy(input_tensor).npu()
        if npu_format not in (-1, 0):
            npu_input = npu_input.npu_format_cast(npu_format)
        return cpu_input, npu_input

    def get_broad_size(self, cpu_tensor):
        shape = list(cpu_tensor.shape)
        addim = np.random.randint(0, 10, 2)
        sizes = list(addim) + shape
        return list(sizes)

    def test_dynamic_threads_support_op(self, device):
        format_list = [0]
        dtype_list = [np.float32, np.float16]
        dim_list = [4]
        net = BroadcastToNet()
        net_npu = copy.deepcopy(net).to("npu")
        items = [
            [i, j, k] for i in format_list for j in dtype_list for k in dim_list
        ]
        for item in items:
            if item[0] == 29 and item[2] == 1:
                continue
            for _ in range(100):
                cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -10, 10)
                sizes = self.get_broad_size(cpu_tensor)
                output = net_npu(npu_tensor, sizes)
                size1 = np.array(output.size(), dtype=np.int32)
                size2 = np.array(sizes, dtype=np.int32)
                self.assertRtolEqual(size1, size2)


instantiate_device_type_tests(TestShape, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
