# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
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

"""Add validation cases for gradient-mode APIs on NPU.

This file validates torch.no_grad, torch.enable_grad, torch.set_grad_enabled,
and torch.autograd.grad_mode.set_grad_enabled.clone, including clone object
independence, mode preservation, grad-mode restoration, and decorator behavior
with NPU tensors.
"""

import torch
from torch_npu.testing.testcase import TestCase, run_tests

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

    def test_set_grad_enabled_clone(self):
        initial_grad_enabled = torch.is_grad_enabled()

        with torch.set_grad_enabled(initial_grad_enabled):
            for mode in (False, True):
                ctx = torch.set_grad_enabled(mode)
                cloned = ctx.clone()

                self.assertIsNot(ctx, cloned)
                self.assertIsInstance(
                    cloned,
                    torch.autograd.grad_mode.set_grad_enabled,
                )
                self.assertEqual(cloned.mode, mode)

        self.assertEqual(torch.is_grad_enabled(), initial_grad_enabled)

    def test_set_grad_enabled_clone_decorator(self):
        initial_grad_enabled = torch.is_grad_enabled()
        x = torch.tensor(
            [1.0],
            device="npu:0",
            requires_grad=True,
        )

        with torch.set_grad_enabled(initial_grad_enabled):
            @torch.set_grad_enabled(False).clone()
            def grad_disabled(tensor):
                return tensor * 2

            with torch.enable_grad():
                disabled_result = grad_disabled(x)

            self.assertEqual(disabled_result.device, x.device)
            self.assertFalse(disabled_result.requires_grad)

            @torch.set_grad_enabled(True).clone()
            def grad_enabled(tensor):
                return tensor * 2

            with torch.no_grad():
                enabled_result = grad_enabled(x)

            self.assertEqual(enabled_result.device, x.device)
            self.assertTrue(enabled_result.requires_grad)

        self.assertEqual(torch.is_grad_enabled(), initial_grad_enabled)

if __name__ == "__main__":
    run_tests()
