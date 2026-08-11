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
Add validation cases for torch.autograd.Variable API on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
2. This file validates torch.autograd.Variable (extendable).
"""
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestAutogradVariable(TestCase):
    def test_variable_wraps_npu_tensor(self):
        """Test torch.autograd.Variable wraps an NPU tensor and inherits device."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device_type)
        v = torch.autograd.Variable(x)
        self.assertEqual(v.device, x.device)
        self.assertEqual(v.dtype, x.dtype)
        self.assertEqual(v.shape, x.shape)
        self.assertTrue(torch.equal(v, x))

    def test_variable_isinstance_check(self):
        """Test torch.autograd.Variable instance is both Variable and Tensor."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device_type)
        v = torch.autograd.Variable(x)
        self.assertIsInstance(v, torch.Tensor)
        self.assertIsInstance(v, torch.autograd.Variable)

    def test_variable_arithmetic_on_npu(self):
        """Test basic arithmetic operations on a Variable holding an NPU tensor."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device_type)
        y = torch.tensor([3.0, 4.0], dtype=torch.float32, device=device_type)
        vx = torch.autograd.Variable(x)
        vy = torch.autograd.Variable(y)
        out = vx + vy
        self.assertEqual(out.device.type, device_type)
        self.assertIsInstance(out, torch.autograd.Variable)
        self.assertTrue(
            torch.equal(
                out, torch.tensor([4.0, 6.0], dtype=torch.float32, device=device_type)
            )
        )

    def test_variable_gradient_on_npu(self):
        """Test backward pass on a Variable that requires grad on NPU."""
        v = torch.autograd.Variable(
            torch.tensor([1.0, 2.0], dtype=torch.float32, device=device_type)
        )
        v.requires_grad_(True)
        self.assertTrue(v.requires_grad)
        loss = (v * 2).sum()
        loss.backward()
        self.assertIsNotNone(v.grad)
        self.assertTrue(
            torch.equal(
                v.grad,
                torch.tensor([2.0, 2.0], dtype=torch.float32, device=device_type),
            )
        )

    def test_variable_cpu_to_npu_roundtrip(self):
        """Test Variable created on CPU keeps Variable type after moving to NPU."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        v = torch.autograd.Variable(x).to(device_type)
        self.assertEqual(v.device.type, device_type)
        self.assertIsInstance(v, torch.autograd.Variable)
        self.assertTrue(torch.equal(v.cpu(), x))

    def test_variable_npu_to_cpu_roundtrip(self):
        """Test Variable created on NPU keeps Variable type after moving to CPU."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device_type)
        v = torch.autograd.Variable(x).cpu()
        self.assertEqual(v.device.type, "cpu")
        self.assertIsInstance(v, torch.autograd.Variable)
        self.assertTrue(torch.equal(v, x.cpu()))

    def test_variable_requires_grad_in_constructor(self):
        """Test Variable with requires_grad=True passed directly to constructor."""
        v = torch.autograd.Variable(
            torch.tensor([1.0, 2.0], dtype=torch.float32, device=device_type),
            requires_grad=True,
        )
        self.assertTrue(v.requires_grad)
        loss = (v * 2).sum()
        loss.backward()
        self.assertIsNotNone(v.grad)

    def test_variable_requires_grad_default(self):
        """Test Variable with default requires_grad=False."""
        v = torch.autograd.Variable(
            torch.tensor([3.0, 4.0], dtype=torch.float32, device=device_type)
        )
        self.assertFalse(v.requires_grad)

    def test_variable_with_non_tensor_input(self):
        """Test Variable raises TypeError for non-Tensor inputs."""
        with self.assertRaises(TypeError):
            torch.autograd.Variable([1.0, 2.0])
        with self.assertRaises(TypeError):
            torch.autograd.Variable("not_a_tensor")


if __name__ == "__main__":
    run_tests()
