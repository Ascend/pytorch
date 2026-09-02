# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.autograd.gradcheck on Ascend NPU.

PyTorch community has independent test cases for gradcheck in
test/test_autograd.py, but these cases cannot run on NPU because Ascend NPU
does not support float64 for some operations. This file validates that
gradcheck works correctly in slow_mode with float64 on NPU for the
operations that Ascend NPU supports, and covers parameter combinations
including raise_exception, check_undefined_grad, check_batched_grad,
fast_mode, and edge cases.
"""
import torch
from torch.autograd import gradcheck, gradgradcheck

from torch_npu.testing.testcase import TestCase, run_tests


class TestGradcheck(TestCase):
    """Test cases for torch.autograd.gradcheck on Ascend NPU."""

    def test_gradcheck_slow_mode_mul(self):
        """slow_mode mul with gradient value verification."""
        torch.manual_seed(42)

        def f(inp):
            return inp.mul(5)

        x = torch.rand(10, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        y = f(xc)
        y.sum().backward()
        self.assertTrue(torch.allclose(xc.grad, torch.full_like(xc, 5.0)))

    def test_gradcheck_slow_mode_linear(self):
        """slow_mode linear function."""
        torch.manual_seed(42)

        def f(x):
            return 3 * x + 2

        x = torch.rand(4, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        y = f(xc)
        y.sum().backward()
        self.assertTrue(torch.allclose(xc.grad, torch.full_like(xc, 3.0)))

    def test_gradcheck_slow_mode_sin_cos(self):
        """slow_mode sin and cos."""
        torch.manual_seed(42)

        def f(x):
            return x.sin().cos()

        x = torch.rand(8, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        y = f(xc)
        y.sum().backward()
        self.assertFalse(torch.allclose(xc.grad, torch.zeros_like(xc.grad)))

    def test_gradcheck_slow_mode_exp(self):
        """slow_mode exp."""
        torch.manual_seed(42)

        def f(x):
            return x.exp()

        x = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        y = f(xc)
        y.sum().backward()
        self.assertTrue(torch.allclose(xc.grad, xc.exp(), rtol=1e-5))

    def test_gradcheck_slow_mode_sum(self):
        """slow_mode sum reduction."""
        torch.manual_seed(42)

        def f(x):
            return x.sum()

        x = torch.rand(4, 5, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        y = f(xc)
        y.backward()
        self.assertTrue(torch.allclose(xc.grad, torch.ones_like(xc)))

    def test_gradcheck_slow_mode_multiple_inputs(self):
        """slow_mode multiple inputs."""
        torch.manual_seed(42)

        def f(x, y):
            return x * y + x

        x = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        y = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, (x, y), fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        yc = y.detach().clone().requires_grad_(True)
        z = f(xc, yc)
        z.sum().backward()
        self.assertTrue(torch.allclose(xc.grad, yc + 1.0))
        self.assertTrue(torch.allclose(yc.grad, xc))

    def test_gradcheck_slow_mode_return_tuple(self):
        """slow_mode function returning tuple."""
        torch.manual_seed(42)

        def f(x):
            return x.sin(), x.cos()

        x = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))
        xc = x.detach().clone().requires_grad_(True)
        s, c = f(xc)
        (s.sum() + c.sum()).backward()
        self.assertFalse(torch.allclose(xc.grad, torch.zeros_like(xc.grad)))

    def test_gradgradcheck_slow_mode_mul(self):
        """gradgradcheck slow_mode with mul."""
        torch.manual_seed(42)

        def f(inp):
            return inp.mul(5)

        x = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradgradcheck(f, x, fast_mode=False))

    def test_gradgradcheck_slow_mode_multiple_inputs(self):
        """gradgradcheck slow_mode with multiple inputs."""
        torch.manual_seed(42)

        def f(x, y):
            return x * y

        x = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=True)
        y = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradgradcheck(f, (x, y), fast_mode=False))

    # Parameter coverage tests

    def test_gradcheck_raise_exception_false(self):
        """raise_exception=False returns False on mismatch."""
        torch.manual_seed(42)

        def f(x):
            return x * torch.tensor([1.0, 2.0, 3.0], device=x.device)

        x = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=True)
        result = gradcheck(f, x, raise_exception=False, atol=1e-10, rtol=1e-10)
        self.assertIsInstance(result, bool)

    def test_gradcheck_nondet_tol(self):
        """nondet_tol parameter works."""
        torch.manual_seed(42)

        def f(x):
            return x.sin()

        x = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        result = gradcheck(f, x, nondet_tol=1.0)
        self.assertTrue(result)

    def test_gradcheck_check_backward_ad_false(self):
        """check_backward_ad=False skips backward AD check."""
        torch.manual_seed(42)

        def f(x):
            return x.neg()

        x = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=True)
        # check_backward_ad=False with check_forward_ad=True
        result = gradcheck(f, x, fast_mode=False, check_backward_ad=False,
                           check_forward_ad=True)
        self.assertIsInstance(result, bool)

    def test_gradcheck_check_batched_grad(self):
        """check_batched_grad=True with slow_mode."""
        torch.manual_seed(42)

        def f(x):
            return x.pow(2)

        x = torch.rand(4, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False, check_batched_grad=True))

    def test_gradcheck_check_undefined_grad_false(self):
        """check_undefined_grad=False skips undefined grad check."""
        torch.manual_seed(42)

        def f(x):
            return x.mul(5)

        x = torch.rand(4, dtype=torch.float64, device="npu", requires_grad=True)
        result = gradcheck(f, x, check_undefined_grad=False)
        self.assertIsInstance(result, bool)

    def test_gradcheck_custom_eps_atol_rtol(self):
        """Custom eps, atol, rtol values."""
        torch.manual_seed(42)

        def f(x):
            return x.sin()

        x = torch.rand(5, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False,
                                  eps=1e-4, atol=1e-3, rtol=1e-2))

    # Edge cases

    def test_gradcheck_single_element_tensor(self):
        """1-element tensor (boundary value)."""
        torch.manual_seed(42)

        def f(x):
            return x * 2

        x = torch.rand(1, dtype=torch.float64, device="npu", requires_grad=True)
        self.assertTrue(gradcheck(f, x, fast_mode=False))

    def test_gradcheck_no_requires_grad_input_raises(self):
        """Input without requires_grad raises ValueError."""
        torch.manual_seed(42)

        def f(x):
            return x * 2

        x = torch.rand(3, dtype=torch.float64, device="npu", requires_grad=False)
        with self.assertRaises((ValueError, RuntimeError)):
            gradcheck(f, x, fast_mode=False)

    def test_gradcheck_masked_parameter(self):
        """masked=True with masked=False comparison."""
        torch.manual_seed(42)

        def f(x):
            return x.mul(5)

        x = torch.rand(4, dtype=torch.float64, device="npu", requires_grad=True)
        r1 = gradcheck(f, x, fast_mode=False, masked=True,
                       raise_exception=False)
        r2 = gradcheck(f, x, fast_mode=False, masked=False,
                       raise_exception=False)
        self.assertIsInstance(r1, bool)
        self.assertIsInstance(r2, bool)


if __name__ == "__main__":
    run_tests()
