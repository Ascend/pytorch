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


class GatherV2Net(torch.nn.Module):
    def __init__(self):
        super(GatherV2Net, self).__init__()

    def forward(self, x, dim, idx):
        if x.device == torch.device("cpu") and x.dtype == torch.float16:
            x = x.to(torch.float32)
        out = torch.index_select(x, dim, idx)
        if x.device == torch.device("cpu") and x.dtype == torch.float16:
            out = out.to(torch.float16)
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

    def get_random_dim_index(self, cpu_tensor):
        shape = list(cpu_tensor.shape)
        dim = np.random.randint(0, len(shape))
        the_size_of_dim = shape[dim]
        indices = np.random.randint(0, the_size_of_dim, np.random.randint(the_size_of_dim), dtype=np.int64)
        indices = torch.from_numpy(indices)
        return dim, indices

    def test_dynamic_threads_support_op(self, device):
        format_list = [0]
        dtype_list = [np.float32, np.float16]
        dim_list = [3,4]
        net = GatherV2Net()
        net_npu = copy.deepcopy(net).to("npu")
        items = [
            [i, j, k] for i in format_list for j in dtype_list for k in dim_list
        ]
        for item in items:
            if item[0] == 29 and item[2] == 1:
                continue
            if item[0] in [0,3,29] and item[2] == 5:
                continue
            for _ in range(100):
                cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -10, 10)
                dim, indices = self.get_random_dim_index(cpu_tensor)
                cpu_output = net(cpu_tensor, dim, indices)
                npu_output = net_npu(npu_tensor, dim, indices.to("npu"))
                self.assertRtolEqual(cpu_output.to(npu_output.dtype).numpy(), npu_output.cpu().numpy())


instantiate_device_type_tests(TestShape, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
