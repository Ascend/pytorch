# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Add consistency validation cases for torch.Tensor.module_load.

This file adds self-contained, focused validation for torch.Tensor.module_load as requested by the
Ascend for PyTorch API consistency task (#2722). Where tensors are involved
they run on NPU via the accelerator device pattern. Extendable.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


DEVICE = (acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu")


class TestTensorModuleLoad(TestCase):
    def test_module_load_returns_source_values(self):
        dest = torch.zeros(3).to(DEVICE)
        src = torch.tensor([1.0, 2.0, 3.0]).to(DEVICE)
        out = dest.module_load(src)
        self.assertEqual(out.cpu(), src.cpu())

    def test_module_load_keeps_dest_dtype(self):
        dest = torch.zeros(2, dtype=torch.float32).to(DEVICE)
        src = torch.tensor([1.0, 2.0], dtype=torch.float64).to(DEVICE)
        out = dest.module_load(src)
        self.assertEqual(out.dtype, dest.dtype)

    def test_module_load_assign_true(self):
        dest = torch.zeros(3).to(DEVICE)
        src = torch.tensor([4.0, 5.0, 6.0]).to(DEVICE)
        out = dest.module_load(src, assign=True)
        # assign=True should return other.detach(), preserving src values
        self.assertEqual(out.cpu(), src.cpu())

    def test_module_load_updates_destination(self):
        dest = torch.ones(3).to(DEVICE)
        src = torch.tensor([7.0, 8.0, 9.0]).to(DEVICE)
        out = dest.module_load(src, assign=False)
        # assign=False does self.copy_(other).detach(), dest should be updated
        self.assertEqual(dest.cpu(), src.cpu())

    def test_module_load_returns_detached_tensor(self):
        dest = torch.zeros(3, requires_grad=True).to(DEVICE)
        src = torch.tensor([1.0, 2.0, 3.0]).to(DEVICE)
        out = dest.module_load(src)
        # module_load returns detach() result, so out should not require grad
        self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    run_tests()
