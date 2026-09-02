# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
# Licensed under the BSD 3-Clause License
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for Tensor.ndim API on Ascend NPU:

PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
This file validates Tensor.ndim (extendable).

Test command:
    python test/npu/test_tensor_ndim.py
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices

assert torch_npu is not None  # NPU backend registration


class TestTensorNdim(TestCase):

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_0d_tensor(self):
        x = torch.tensor(5).npu()
        self.assertEqual(x.ndim, 0)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_1d_tensor(self):
        x = torch.randn(5).npu()
        self.assertEqual(x.ndim, 1)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_2d_tensor(self):
        x = torch.randn(3, 4).npu()
        self.assertEqual(x.ndim, 2)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_3d_tensor(self):
        x = torch.randn(2, 3, 4).npu()
        self.assertEqual(x.ndim, 3)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_4d_tensor(self):
        x = torch.randn(2, 3, 4, 5).npu()
        self.assertEqual(x.ndim, 4)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_cpu_npu_consistency(self):
        for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]:
            cpu_tensor = torch.randn(*shape)
            npu_tensor = cpu_tensor.npu()
            self.assertEqual(cpu_tensor.ndim, npu_tensor.ndim)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_ndim_after_operations(self):
        x = torch.randn(3, 4).npu()
        y = torch.randn(3, 4).npu()
        z = x + y
        self.assertEqual(z.ndim, 2)
        w = z.sum(dim=1)
        self.assertEqual(w.ndim, 1)


if __name__ == "__main__":
    run_tests()
