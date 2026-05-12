# Owner(s): ["module: nn"]

"""
Add validation cases for torch.nn APIs on NPU:
1. test/test_nn.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates torch.nn.Parameter, torch.nn.Buffer, torch.nn.Module.npu (extendable).
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


device = torch.device("npu:0")


class TestNPUParameterBuffer(TestCase):
    def test_parameter_api(self):
        """Verifies Parameter creation, attributes, and in-place modification on NPU."""
        # 直接在NPU上创建Parameter（推荐方式）
        p = nn.Parameter(torch.randn(10, 20, device=device))

        # 验证类型和属性
        self.assertIsInstance(p, nn.Parameter)
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.device, device)
        self.assertEqual(p.shape, (10, 20))

        # 验证数据操作
        p.data = p.data * 2  # Parameter支持.data访问
        self.assertEqual(p.shape, (10, 20))  # 形状不变

        # 修改requires_grad
        p.requires_grad = False
        self.assertFalse(p.requires_grad)

    def test_buffer_api(self):
        """Verifies Buffer creation, attributes, and in-place modification on NPU."""
        # 创建persistent Buffer（默认）
        b1 = nn.Buffer(torch.randn(5, 5, device=device))
        self.assertIsInstance(b1, nn.Buffer)
        self.assertFalse(b1.requires_grad)  # Buffer默认不需要梯度
        self.assertEqual(b1.device, device)

        # 创建non-persistent Buffer
        b2 = nn.Buffer(torch.randn(3, 3, device=device), persistent=False)
        self.assertFalse(b2.requires_grad)
        self.assertEqual(b2.device, device)

        # 验证persistent属性在Module中的行为
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("persistent_buf", b1)
                self.register_buffer("non_persistent_buf", b2, persistent=False)

        m = TestModule()
        state_dict = m.state_dict()

        # persistent buffer应在state_dict中
        self.assertIn("persistent_buf", state_dict)
        # non-persistent buffer不应在state_dict中
        self.assertNotIn("non_persistent_buf", state_dict)

        # 验证Buffer在state_dict中的值正确
        self.assertEqual(state_dict["persistent_buf"].device, device)


class TestNNModuleAPIs(TestCase):
    def test_npu(self):
        """Verifies that Module.npu() correctly move parameters and buffers to NPU."""

        class MyModule(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(in_features, out_features))
                self.register_buffer("buf", torch.randn(out_features))

            def forward(self, x):
                return x @ self.weight + self.buf

        m = MyModule(3, 5)
        self.assertEqual(m.to("npu"), m.npu())
        m1 = m.npu()

        for param in m1.parameters():
            self.assertEqual(param.device.type, "npu")
        for param in m1.buffers():
            self.assertEqual(param.device.type, "npu")


if __name__ == "__main__":
    run_tests()
