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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.optim APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.optim.Optimizer.zero_grad (extendable).
"""

import torch
import torch_npu  # noqa: F401

from torch_npu.testing.testcase import TestCase, run_tests


class TestOptimizerZeroGrad(TestCase):
    def _create_optimizer_with_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], device="npu"))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.square().sum().backward()
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.count_nonzero(parameter.grad).item() > 0)
        return parameter, optimizer

    def _assert_gradient_is_none(self, set_to_none):
        parameter, optimizer = self._create_optimizer_with_gradient()
        optimizer.zero_grad(set_to_none)
        self.assertIsNone(parameter.grad)

    def _assert_gradient_is_zero(self, set_to_none):
        parameter, optimizer = self._create_optimizer_with_gradient()
        optimizer.zero_grad(set_to_none)
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_zero_grad_default_sets_gradient_to_none(self):
        parameter, optimizer = self._create_optimizer_with_gradient()

        optimizer.zero_grad()

        self.assertIsNone(parameter.grad)

    def test_zero_grad_boolean_parameter_values(self):
        self._assert_gradient_is_none(True)
        self._assert_gradient_is_zero(False)

    def test_zero_grad_non_boolean_truthy_and_falsy_values(self):
        # The upstream implementation uses Python truthiness for this argument.
        for set_to_none in (1, [1]):
            self._assert_gradient_is_none(set_to_none)
        for set_to_none in (0, [], None):
            self._assert_gradient_is_zero(set_to_none)

    def test_zero_grad_rejects_invalid_call_signatures(self):
        _, optimizer = self._create_optimizer_with_gradient()

        with self.assertRaises(TypeError):
            optimizer.zero_grad(True, False)
        with self.assertRaises(TypeError):
            optimizer.zero_grad(unexpected=True)


if __name__ == "__main__":
    run_tests()
