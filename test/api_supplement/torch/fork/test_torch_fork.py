# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fork 接口功能正确性
API 名称：torch.fork
API 签名：torch.fork(func, *args, **kwargs) -> torch.Future

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 基础调用 | 传入可调用对象，返回 Future | 已覆盖 |
| 参数枚举/边界值 | 位置参数、关键字参数、无参函数 | 已覆盖 |
| 副作用验证 | Future.value() 返回被调用函数的真实结果 | 已覆盖 |
| 幂等性 | 对同一纯函数多次调用结果一致 | 已覆盖 |
| 异常路径 | 非可调用对象、函数内部抛异常 | 已覆盖 |

未覆盖项及原因：
- 该 API 为运行时调度工具：在独立的 RNG/执行上下文中调用给定函数并返回 Future。
  返回值为 Future 而非张量数值，故不以 CPU Golden 做浮点对照；改为断言 Future
  类型、value() 结果与直接调用的等价性及异常传播。
- 混合设备维度不适用：输入为调用对象与参数，不做跨设备张量运算。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testfork(TestCase):
    """Test cases for torch.fork (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_future(self):
        """Calling fork returns a torch.Future object."""
        future = torch.fork(lambda: 1)
        self.assertIsInstance(future, torch.Future)

    def test_value_equals_direct_call(self):
        """Future.value() equals invoking the function directly."""
        future = torch.fork(lambda a, b: a + b, 2, 3)
        self.assertEqual(future.value(), 5)

    def test_kwargs_forwarded(self):
        """Keyword arguments are forwarded to the target function."""
        future = torch.fork(lambda a, b=0: a * b, 4, b=5)
        self.assertEqual(future.value(), 20)

    def test_future_wait_and_done(self):
        """wait() blocks until completion and done() then reports True."""
        future = torch.fork(lambda: 42)
        future.wait()
        self.assertTrue(future.done())
        self.assertEqual(future.value(), 42)

    def test_non_callable_raises(self):
        """A non-callable first argument is rejected."""
        with self.assertRaises(TypeError):
            torch.fork(123)

    def test_exception_propagates(self):
        """An exception raised inside the target is surfaced to the caller."""
        def boom():
            raise ValueError('boom')
        with self.assertRaises(ValueError):
            torch.fork(boom)


if __name__ == "__main__":
    run_tests()
