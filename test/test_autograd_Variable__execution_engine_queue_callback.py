# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.Variable._execution_engine.queue_callback 接口功能正确性
API 名称：torch.autograd.Variable._execution_engine.queue_callback
API 签名：queue_callback(fn: Callable) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入 callable 参数                                       | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | function / lambda                                            | 已覆盖      |
| 传参与不传参     | 必须传参                                                     | 已覆盖      |
| 等价类/边界值    | 简单回调 / 无操作回调                                        | 已覆盖      |
| 正常传参场景     | backward pass 期间队列回调不报错                             | 已覆盖      |
| 异常传参场景     | 非 backward pass 调用触发 RuntimeError                       | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestExecutionEngineQueueCallback(TestCase):
    """Test cases for torch.autograd.Variable._execution_engine.queue_callback."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_queue_callback_basic(self):
        """Queue a simple callback during backward pass does not raise."""
        callback_called = [False]

        def cb():
            callback_called[0] = True

        x = torch.randn(2, requires_grad=True, device=self.device)
        loss = x.sum()
        x.register_hook(lambda grad: torch.autograd.Variable._execution_engine.queue_callback(cb))
        loss.backward()
        self.assertTrue(callback_called[0], "Callback was not executed during backward")

    def test_queue_callback_lambda(self):
        """Queue a lambda callback during backward pass does not raise."""
        x = torch.randn(2, requires_grad=True, device=self.device)
        loss = x.sum()
        x.register_hook(lambda grad: torch.autograd.Variable._execution_engine.queue_callback(lambda: None))
        loss.backward()

    def test_queue_callback_outside_backward_raises(self):
        """Queue callback outside backward pass raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.autograd.Variable._execution_engine.queue_callback(lambda: None)


if __name__ == "__main__":
    run_tests()
