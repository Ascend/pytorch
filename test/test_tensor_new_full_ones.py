# Copyright (c) 2026 Huawei Technologies Co., Ltd
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

"""
Add validation cases for torch.Tensor APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.Tensor.new_full and torch.Tensor.new_ones (extendable).
"""
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTensorNewFullNPU(TestCase):
    def test_new_full_basic(self):
        t = torch.randn(2, 3, device=device_type)
        r = t.new_full((2, 2), 5.0)
        self.assertEqual(r.shape, torch.Size([2, 2]))
        self.assertEqual(r.dtype, torch.float32)
        self.assertEqual(r.device.type, device_type)
        self.assertEqual(r.tolist(), [[5.0, 5.0], [5.0, 5.0]])

    def test_new_full_dtype(self):
        t = torch.randn(2, 3, device=device_type)
        r = t.new_full((2,), 3, dtype=torch.int64)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(r.tolist(), [3, 3])

    def test_new_full_1d(self):
        t = torch.randn(4, device=device_type)
        r = t.new_full((3,), 7)
        self.assertEqual(r.shape, torch.Size([3]))
        self.assertEqual(r.tolist(), [7, 7, 7])

    def test_new_full_float_value(self):
        t = torch.randn(2, device=device_type)
        r = t.new_full((2, 2), 2.5)
        self.assertEqual(r.tolist(), [[2.5, 2.5], [2.5, 2.5]])


class TestTensorNewOnesNPU(TestCase):
    def test_new_ones_basic(self):
        t = torch.randn(2, 3, device=device_type)
        o = t.new_ones((3, 3))
        self.assertEqual(o.shape, torch.Size([3, 3]))
        self.assertEqual(o.dtype, torch.float32)
        self.assertEqual(o.device.type, device_type)
        self.assertEqual(o.sum().item(), 9)

    def test_new_ones_dtype(self):
        t = torch.randn(2, device=device_type)
        o = t.new_ones((2, 2), dtype=torch.int32)
        self.assertEqual(o.dtype, torch.int32)
        self.assertEqual(o.tolist(), [[1, 1], [1, 1]])

    def test_new_ones_1d(self):
        t = torch.randn(4, device=device_type)
        o = t.new_ones((5,))
        self.assertEqual(o.shape, torch.Size([5]))
        self.assertEqual(o.tolist(), [1, 1, 1, 1, 1])


if __name__ == "__main__":
    run_tests()
