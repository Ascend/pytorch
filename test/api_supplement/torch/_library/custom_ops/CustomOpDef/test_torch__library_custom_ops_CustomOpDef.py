# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._library.custom_ops.CustomOpDef 接口功能正确性
API 名称：torch._library.custom_ops.CustomOpDef
API 签名：CustomOpDef(namespace, name, schema, fn, tags=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 基础调用 | custom_op 装饰器返回 CustomOpDef 实例 | 已覆盖 |
| 参数枚举/边界值 | 默认内核与设备专用内核的注册/覆盖 | 已覆盖 |
| 副作用验证 | 实例可调用并调度到已注册内核 | 已覆盖 |
| 幂等性 | 同一实例多次调用结果一致 | 已覆盖 |
| 异常路径 | 非法算子名 / 非法 schema 触发异常 | 已覆盖 |

未覆盖项及原因：
- 该 API 为自定义算子定义工具：运行结果取决于注册的实现函数，不做浮点 Golden 对照，
  改为断言实例类型、调度结果与异常路径。
- 设备维度：内核按 device_type 注册，本用例在 CPU 上验证注册与调度契约；
  NPU 侧内核注册不属当前环境内的稳定路径，未覆盖。
- 混合设备维度不适用：调度按输入张量设备分发，无跨设备组合输入。
"""
import os
import unittest

import torch
from torch import Tensor

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

from torch._library.custom_ops import CustomOpDef  # noqa: E402


@torch.library.custom_op("ap_sup_copdef::plain", mutates_args=())
def _plain_op(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


@torch.library.custom_op("ap_sup_copdef::with_cpu", mutates_args=())
def _cpu_op(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


@_cpu_op.register_kernel('cpu')
def _cpu_op_cpu(x: Tensor) -> Tensor:
    return torch.ones_like(x)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Test_library_custom_ops_CustomOpDef(TestCase):
    """Test cases for torch._library.custom_ops.CustomOpDef (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_decorator_returns_customopdef(self):
        """The custom_op decorator produces a CustomOpDef instance."""
        self.assertIsInstance(_plain_op, CustomOpDef)
        self.assertIsInstance(_cpu_op, CustomOpDef)

    def test_call_uses_default_kernel(self):
        """Calling the op runs its default (composition) kernel."""
        out = _plain_op(torch.randn(3))
        torch.testing.assert_close(out, torch.zeros(3))

    def test_register_kernel_overrides_default(self):
        """A device-specific kernel takes precedence over the default one."""
        out = _cpu_op(torch.randn(3))
        torch.testing.assert_close(out, torch.ones(3))

    def test_idempotent_call(self):
        """Repeated calls with the same input give the same result."""
        x = torch.randn(4)
        torch.testing.assert_close(_cpu_op(x), _cpu_op(x))

    def test_invalid_name_raises(self):
        """A fully-qualified name without exactly one '::' is rejected."""
        def body(x: Tensor) -> Tensor:
            return x

        with self.assertRaises(ValueError):
            torch.library.custom_op('noseparator', mutates_args=())(body)

    def test_invalid_schema_raises(self):
        """A malformed explicit schema is rejected."""
        def body(x: Tensor) -> Tensor:
            return x

        with self.assertRaises((RuntimeError, ValueError)):
            torch.library.custom_op('ap_sup_copdef::bad', mutates_args=(), schema='(Tensor x) ->')(body)


if __name__ == "__main__":
    run_tests()
