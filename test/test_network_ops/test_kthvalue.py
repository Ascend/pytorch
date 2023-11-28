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

import random
import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestKthvalues(TestCase):
    def generate_data(self, min1, max1, shape, dtype):
        if dtype == np.float32:
            x = np.random.uniform(min1, max1, shape).astype(np.float16)
            x = x.astype(np.float32)
            npu_x = torch.from_numpy(x)
            return npu_x
        x = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_x = torch.from_numpy(x)
        return npu_x

    def generate_int_k(self, max1):
        k = np.random.randint(1, max1 + 1)
        return k

    def generate_int_dim(self, max1):
        dim = np.random.randint(-max1, max1)
        return dim

    def generate_bool_keepdim(self):
        keepdim = random.choice([True, False])
        return keepdim

    def cpu_op_exec(self, x, k, dim, keepdim):
        y, indices = torch.kthvalue(x, k, dim, keepdim)
        y = y.numpy()
        indices = indices.numpy()
        return y, indices

    def npu_op_exec(self, x, k, dim, keepdim):
        x = x.to("npu")
        y, indices = torch.kthvalue(x, k, dim, keepdim)
        y = y.to("cpu")
        y = y.numpy()
        indices = indices.to("cpu")
        indices = indices.numpy()
        return y, indices

    def cpu_op_exec_without_dim(self, x, k, keepdim):
        y, indices = torch.kthvalue(x, k, keepdim=keepdim)
        y = y.numpy()
        indices = indices.numpy()
        return y, indices

    def npu_op_exec_without_dim(self, x, k, keepdim):
        x = x.to("npu")
        y, indices = torch.kthvalue(x, k, keepdim=keepdim)
        y = y.to("cpu")
        y = y.numpy()
        indices = indices.to("cpu")
        indices = indices.numpy()
        return y, indices

    def cpu_op_exec_without_keepdim(self, x, k, dim):
        y, indices = torch.kthvalue(x, k, dim=dim)
        y = y.numpy()
        indices = indices.numpy()
        return y, indices

    def npu_op_exec_without_keepdim(self, x, k, dim):
        x = x.to("npu")
        y, indices = torch.kthvalue(x, k, dim=dim)
        y = y.to("cpu")
        y = y.numpy()
        indices = indices.to("cpu")
        indices = indices.numpy()
        return y, indices

    def test_kthvalues(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.float32)
        k = self.generate_int_k(3)
        dim = self.generate_int_dim(4)
        keepdim = self.generate_bool_keepdim()
        cpu_y, cpu_indices = self.cpu_op_exec(x, k, dim, keepdim)
        npu_y, npu_indices = self.npu_op_exec(x, k, dim, keepdim)
        self.assertRtolEqual(cpu_y, npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    @unittest.skip("skip test_kthvalues_without_dim now")
    def test_kthvalues_without_dim(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.int32)
        k = self.generate_int_k(3)
        keepdim = self.generate_bool_keepdim()
        cpu_y, cpu_indices = self.cpu_op_exec_without_dim(x, k, keepdim)
        npu_y, npu_indices = self.npu_op_exec_without_dim(x, k, keepdim)
        self.assertRtolEqual(cpu_y, npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    def test_kthvalues_without_keepdim(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.float16)
        k = self.generate_int_k(3)
        dim = self.generate_int_dim(4)
        cpu_y, cpu_indices = self.cpu_op_exec_without_keepdim(x.float(), k, dim)
        npu_y, npu_indices = self.npu_op_exec_without_keepdim(x, k, dim)
        self.assertRtolEqual(cpu_y.astype(np.float16), npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    def test_kthvalues_out(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.float32)
        k = self.generate_int_k(3)
        dim = self.generate_int_dim(4)
        keepdim = self.generate_bool_keepdim()
        cpu_y = torch.tensor(0.).float()
        cpu_indices = torch.tensor(0)
        npu_y = torch.tensor(0.).float().to("npu")
        npu_indices = torch.tensor(0).long().to("npu")
        torch.kthvalue(x, k, dim, keepdim, out=(cpu_y, cpu_indices))
        torch.kthvalue(x.to("npu"), k, dim, keepdim, out=(npu_y, npu_indices))
        self.assertRtolEqual(cpu_y.numpy(), npu_y.to("cpu").numpy())
        self.assertRtolEqual(cpu_indices.numpy().astype(np.int32), npu_indices.to("cpu").numpy().astype(np.int32))

    def test_kthvalues_dimname(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.float32)
        x.names = ['A', 'B', 'C', 'D']
        k = self.generate_int_k(3)
        keepdim = self.generate_bool_keepdim()
        cpu_y, cpu_indices = self.cpu_op_exec(x, k, 'B', keepdim)
        npu_y, npu_indices = self.npu_op_exec(x, k, 'B', keepdim)
        self.assertRtolEqual(cpu_y, npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    def test_kthvalues_dimname_without_dim(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.int32)
        x.names = ['A', 'B', 'C', 'D']
        k = self.generate_int_k(3)
        keepdim = self.generate_bool_keepdim()
        cpu_y, cpu_indices = self.cpu_op_exec_without_dim(x, k, keepdim)
        npu_y, npu_indices = self.npu_op_exec_without_dim(x, k, keepdim)
        self.assertRtolEqual(cpu_y, npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    def test_kthvalues_dimname_without_keepdim(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.float32)
        x.names = ['A', 'B', 'C', 'D']
        k = self.generate_int_k(3)
        cpu_y, cpu_indices = self.cpu_op_exec_without_keepdim(x, k, 'B')
        npu_y, npu_indices = self.npu_op_exec_without_keepdim(x, k, 'B')
        self.assertRtolEqual(cpu_y, npu_y)
        self.assertRtolEqual(cpu_indices.astype(np.int32), npu_indices.astype(np.int32))

    @unittest.skip("skip test_kthvalues_dimname_out now")
    def test_kthvalues_dimname_out(self, device="npu"):
        x = self.generate_data(-100, 100, (3, 4, 5, 6), np.int32)
        x.names = ['A', 'B', 'C', 'D']
        k = self.generate_int_k(3)
        dim = 'C'
        keepdim = self.generate_bool_keepdim()
        cpu_y = torch.tensor(0).int()
        cpu_indices = torch.tensor(0)
        npu_y = torch.tensor(0).int().to("npu")
        npu_indices = torch.tensor(0).long().to("npu")
        torch.kthvalue(x, k, dim, keepdim, out=(cpu_y, cpu_indices))
        torch.kthvalue(x.to("npu"), k, dim, keepdim, out=(npu_y, npu_indices))
        self.assertRtolEqual(cpu_y.numpy(), npu_y.to("cpu").numpy())
        self.assertRtolEqual(cpu_indices.numpy().astype(np.int32), npu_indices.to("cpu").numpy().astype(np.int32))


if __name__ == "__main__":
    run_tests()
