# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestScatterDimUpdate(TestCase):

    def generate_data(self, min, max, shape_var, shape_indices, shape_updates, dtype_var,
                      dtype_indices, dtype_updates, dim):
        var = np.random.uniform(min, max, shape_var).astype(dtype_var)
        updates = np.random.uniform(min, max, shape_updates).astype(dtype_updates)
        indices = np.random.randint(0, shape_var[dim], shape_indices).astype(dtype_indices)

        #modify from numpy.ndarray to torch.tensor
        var = torch.from_numpy(var)
        indices = torch.from_numpy(indices)
        updates = torch.from_numpy(updates)

        return var, indices, updates, dim

    def cpu_op_exec(self, var, indices, updates, dim):
        output = var.scatter(dim=dim, index=indices.long(), src=updates)
        return output.numpy()

    def npu_op_exec(self, var, indices, updates, dim):
        var = var.to("npu")
        indices = indices.to("npu")
        updates = updates.to("npu")
        output = torch.scatter(var, dim, indices, updates)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_scatter_dim_update_32_float32(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (32, ), (32, ), (32, ),
                                                   "float32", "int32", "float32", 0)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_dim_update_32_32_float16(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (32, 32), (32, 32), (32, 32),
                                                   "float16", "int32", "float16", 0)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_dim_update_32_32_float32(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (32, 32), (24, 24), (24, 24),
                                                   "float32", "int32", "float32", 1)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_dim_update_32_32_32_int8(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (32, 32, 32), (24, 24, 24), (32, 32, 32),
                                                   "int8", "int32", "int8", 1)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_dim_update_16_16_16_16_float16(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (16, 16, 16, 16), (8, 8, 8, 8), (12, 12, 12, 12),
                                                   "float16", "int32", "float16", 2)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_dim_update_8_8_8_8_8_floa32(self, device):
        var, indices, updates, dim = self.generate_data(-2, 2, (8, 8, 8, 8, 8), (3, 3, 3, 3, 3), (8, 8, 8, 8, 8),
                                                   "float32", "int32", "float32", 3)
        cpu_output = self.cpu_op_exec(var, indices, updates, dim)
        npu_output = self.npu_op_exec(var, indices, updates, dim)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestScatterDimUpdate, globals(), except_for='cpu')
if __name__ == '__main__':
    torch.npu.set_device("npu:2")
    run_tests()