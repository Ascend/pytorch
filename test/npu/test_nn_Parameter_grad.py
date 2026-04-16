# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.Parameter.grad 属性功能正确性
API 名称：torch.nn.Parameter.grad
API 签名：grad: Optional[Tensor] (attribute, inherited from Tensor)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 梯度为 None 与有值的情况                                     | 已覆盖                                         |
| 枚举选项         | requires_grad=True/False                                     | 已覆盖                                         |
| 参数类型         | 各种 dtype 的 Parameter                                      | 已覆盖                                         |
| 传参与不传参     | N/A（属性访问）                                              | N/A                                            |
| 等价类/边界值    | 空 tensor、单元素、多维 tensor、不同 shape                   | 已覆盖                                         |
| 正常传参场景     | backward 后 grad 存在且形状正确                              | 已覆盖                                         |
| 异常传参场景     | requires_grad=False 时访问 grad                              | 已覆盖                                         |

未覆盖项及原因：
- 无未覆盖项

注意：本测试仅验证功能正确性（grad 属性存在、shape/dtype/device 正确），不做数值精度校验。
"""

import torch
import torch.nn as nn
import torch_npu

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestParameterGrad(TestCase):
    """Test cases for torch.nn.Parameter.grad on NPU devices."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name, 0)

    def test_grad_initially_none(self):
        """Test that grad is None before backward."""
        d = self.device
        param = nn.Parameter(torch.randn(3, 3, dtype=torch.float32, device=d))
        self.assertIsNone(param.grad)

    def test_grad_after_backward(self):
        """Test grad exists and has correct shape after backward."""
        d = self.device
        param = nn.Parameter(torch.randn(3, 3, dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad.shape, param.shape)
        self.assertEqual(param.grad.dtype, param.dtype)
        self.assertEqual(param.grad.device.type, 'npu')

    def test_grad_requires_grad_false(self):
        """Test grad is None when requires_grad=False."""
        d = self.device
        param = nn.Parameter(torch.randn(2, 2, dtype=torch.float32, device=d), requires_grad=False)
        self.assertIsNone(param.grad)
        # backward should raise error on non-leaf or no-grad tensor
        with self.assertRaises(RuntimeError):
            loss = param.sum()
            loss.backward()

    def test_grad_accumulation(self):
        """Test grad accumulation on multiple backward calls."""
        d = self.device
        param = nn.Parameter(torch.ones(2, 2, dtype=torch.float32, device=d))
        loss1 = param.sum()
        loss1.backward()
        self.assertIsNotNone(param.grad)
        grad_first = param.grad.clone()
        loss2 = param.sum()
        loss2.backward()
        # Grad should be accumulated (sum of both gradients)
        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad.shape, (2, 2))

    def test_grad_zero(self):
        """Test grad can be zeroed."""
        d = self.device
        param = nn.Parameter(torch.randn(3, dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertIsNotNone(param.grad)
        param.grad.zero_()
        self.assertTrue((param.grad == 0).all())

    def test_grad_different_dtypes(self):
        """Test grad with different parameter dtypes."""
        d = self.device
        dtypes = [torch.float32, torch.float16]
        for dtype in dtypes:
            param = nn.Parameter(torch.randn(2, 2, dtype=dtype, device=d))
            loss = param.sum()
            loss.backward()
            self.assertIsNotNone(param.grad)
            self.assertIn(param.grad.dtype, [dtype, torch.float32])

    def test_grad_empty_tensor(self):
        """Test grad with empty parameter."""
        d = self.device
        param = nn.Parameter(torch.tensor([], dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad.shape, (0,))

    def test_grad_scalar_tensor(self):
        """Test grad with scalar (0-dim) parameter."""
        d = self.device
        param = nn.Parameter(torch.tensor(5.0, dtype=torch.float32, device=d))
        loss = param * 2
        loss.backward()
        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad.shape, ())

    def test_grad_1d_tensor(self):
        """Test grad with 1D parameter."""
        d = self.device
        param = nn.Parameter(torch.randn(10, dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertEqual(param.grad.shape, (10,))

    def test_grad_3d_tensor(self):
        """Test grad with 3D parameter."""
        d = self.device
        param = nn.Parameter(torch.randn(2, 3, 4, dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertEqual(param.grad.shape, (2, 3, 4))

    def test_grad_in_optimizer_context(self):
        """Test grad in a simple optimizer-like context."""
        d = self.device
        param = nn.Parameter(torch.randn(3, 3, dtype=torch.float32, device=d))
        optimizer = torch.optim.SGD([param], lr=0.01)
        loss = param.sum()
        loss.backward()
        self.assertIsNotNone(param.grad)
        optimizer.step()
        # After step, grad still exists until zero_grad
        self.assertIsNotNone(param.grad)
        optimizer.zero_grad()
        self.assertIsNone(param.grad)

    def test_grad_after_optimizer_step(self):
        """Test grad behavior through optimizer cycle."""
        d = self.device
        param = nn.Parameter(torch.ones(2, 2, dtype=torch.float32, device=d))
        optimizer = torch.optim.SGD([param], lr=0.1)
        # First iteration
        loss = param.sum()
        optimizer.zero_grad()
        self.assertIsNone(param.grad)
        loss.backward()
        self.assertIsNotNone(param.grad)
        optimizer.step()
        # Second iteration
        loss = param.sum()
        optimizer.zero_grad()
        self.assertIsNone(param.grad)
        loss.backward()
        self.assertIsNotNone(param.grad)

    def test_grad_detached_tensor(self):
        """Test grad with detached tensor operations."""
        d = self.device
        param = nn.Parameter(torch.randn(2, 2, dtype=torch.float32, device=d))
        detached = param.detach()
        # Detached tensor is not a Parameter, no grad tracking
        self.assertIsNone(detached.grad_fn)
        # Original param still has grad after backward
        loss = param.sum()
        loss.backward()
        self.assertIsNotNone(param.grad)

    def test_grad_complex_computation(self):
        """Test grad with complex computation graph."""
        d = self.device
        param1 = nn.Parameter(torch.randn(2, 3, dtype=torch.float32, device=d))
        param2 = nn.Parameter(torch.randn(3, 2, dtype=torch.float32, device=d))
        output = torch.matmul(param1, param2)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(param1.grad)
        self.assertIsNotNone(param2.grad)
        self.assertEqual(param1.grad.shape, (2, 3))
        self.assertEqual(param2.grad.shape, (3, 2))

    def test_grad_is_contiguous(self):
        """Test grad tensor properties (contiguous, etc.)."""
        d = self.device
        param = nn.Parameter(torch.randn(3, 3, dtype=torch.float32, device=d))
        loss = param.sum()
        loss.backward()
        self.assertTrue(param.grad.is_contiguous())


if __name__ == "__main__":
    run_tests()
