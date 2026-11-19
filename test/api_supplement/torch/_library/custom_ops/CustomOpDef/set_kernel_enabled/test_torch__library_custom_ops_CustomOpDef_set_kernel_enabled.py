# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._library.custom_ops.CustomOpDef.set_kernel_enabled 接口功能正确性
API 名称：torch._library.custom_ops.CustomOpDef.set_kernel_enabled
API 签名：CustomOpDef.set_kernel_enabled(device_type, enabled=True)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 基础调用 | 作为上下文管理器临时禁用/启用内核 | 已覆盖 |
| 参数枚举/边界值 | enabled=True/False、已禁用/未禁用状态 | 已覆盖 |
| 副作用验证 | 上下文内调度到默认内核、退出后恢复 | 已覆盖 |
| 幂等性 | 重复禁用/启用按文档为 no-op 且行为不变 | 已覆盖 |
| 异常路径 | 未注册内核的设备类型不改变调度行为 | 已覆盖 |

未覆盖项及原因：
- 该 API 为内核调度开关：行为以「上下文内/外调用结果差异」验证，不做浮点 Golden 对照。
- 设备维度：内核按 device_type 注册，本用例在 CPU 上验证禁用/启用与恢复契约；
  NPU 内核注册不属当前环境内的稳定路径，未覆盖。
- 混合设备维度不适用：开关按 device_type 生效，无多张量跨设备输入。
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


@torch.library.custom_op("ap_sup_skc::f", mutates_args=())
def _op(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


@_op.register_kernel('cpu')
def _op_cpu(x: Tensor) -> Tensor:
    return torch.ones_like(x)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Test_library_custom_ops_CustomOpDef_set_kernel_enabled(TestCase):
    """Test cases for CustomOpDef.set_kernel_enabled (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_disable_temporarily_uses_default_kernel(self):
        """Disabling the CPU kernel inside the context falls back to default."""
        x = torch.randn(3)
        torch.testing.assert_close(_op(x), torch.ones(3))
        with _op.set_kernel_enabled('cpu', enabled=False):
            torch.testing.assert_close(_op(x), torch.zeros(3))

    def test_restores_after_context(self):
        """The CPU kernel is re-enabled once the context exits."""
        x = torch.randn(3)
        with _op.set_kernel_enabled('cpu', enabled=False):
            pass
        torch.testing.assert_close(_op(x), torch.ones(3))

    def test_enable_true_is_noop(self):
        """enabled=True inside the context keeps the CPU kernel active."""
        x = torch.randn(3)
        with _op.set_kernel_enabled('cpu', enabled=True):
            torch.testing.assert_close(_op(x), torch.ones(3))

    def test_nested_disabled_then_enabled(self):
        """Nested context can re-enable a kernel disabled by an outer context."""
        x = torch.randn(3)
        with _op.set_kernel_enabled('cpu', enabled=False):
            torch.testing.assert_close(_op(x), torch.zeros(3))
            with _op.set_kernel_enabled('cpu', enabled=True):
                torch.testing.assert_close(_op(x), torch.ones(3))
            torch.testing.assert_close(_op(x), torch.zeros(3))
        torch.testing.assert_close(_op(x), torch.ones(3))

    def test_unregistered_device_is_noop(self):
        """Toggling a device with no registered kernel leaves dispatch unchanged."""
        x = torch.randn(3)
        with _op.set_kernel_enabled('meta', enabled=False):
            torch.testing.assert_close(_op(x), torch.ones(3))


if __name__ == "__main__":
    run_tests()
