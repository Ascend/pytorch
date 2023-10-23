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
from torch_npu.testing.testcase import TestCase, run_tests

import torch_npu

device = 'npu:0'
torch.npu.set_device(device)


class TestLDGComputation(TestCase):
    def test_no_grad(self):
        x = torch.tensor([1], dtype=torch.float32, device=device, requires_grad=True)
        with torch.no_grad():
            y = x * 2
        self.assertFalse(y.requires_grad)

        @torch.no_grad()
        def doubler(x):
            return x * 2
        z = doubler(x)
        self.assertFalse(z.requires_grad)

    def test_enable_grad(self):
        x = torch.tensor([1], dtype=torch.float32, device=device, requires_grad=True)
        with torch.no_grad():
            with torch.enable_grad():
                y = x * 2
        self.assertTrue(y.requires_grad)

        @torch.enable_grad()
        def doubler(x):
            return x * 2
        with torch.no_grad():
            z = doubler(x)
        self.assertTrue(z.requires_grad)

    def test_set_grad_enabled(self):
        x = torch.tensor([1.], device=device, requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)


if __name__ == "__main__":
    run_tests()
