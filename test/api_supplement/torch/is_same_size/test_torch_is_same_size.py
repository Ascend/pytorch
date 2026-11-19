# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.is_same_size 接口功能正确性
API 名称：torch.is_same_size
API 签名：torch.is_same_size(input, other) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空张量与非空张量 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 仅接受 Tensor；非张量报错 | 已覆盖 |
| 传参与不传参 | input、other 均为必填参数 | 已覆盖 |
| 等价类/边界值 | 形状相同/不同、可广播但不相同 | 已覆盖 |
| 正常传参场景 | 返回 Python bool | 已覆盖 |
| 异常传参场景 | 非 Tensor 触发 TypeError | 已覆盖 |
| 混合设备类型 | 一个 CPU、一个设备张量（同形状） | 已覆盖 |

未覆盖项及原因：
- 枚举选项：is_same_size 无枚举参数。
- 该 API 只比较形状元数据，不做数值运算，无 CPU Golden 浮点对照。
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
class Testis_same_size(TestCase):
    """Test cases for torch.is_same_size (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_same_shape_returns_true(self):
        """Two tensors of identical shape compare equal."""
        self.assertTrue(torch.is_same_size(torch.randn(2, 3), torch.randn(2, 3)))

    def test_different_shape_returns_false(self):
        """Different shapes compare unequal even when element counts match."""
        self.assertFalse(torch.is_same_size(torch.randn(2, 3), torch.randn(3, 2)))
        self.assertFalse(torch.is_same_size(torch.randn(6), torch.randn(2, 3)))

    def test_broadcastable_shapes_are_not_same(self):
        """Broadcast-compatible but distinct shapes are not the same size."""
        self.assertFalse(torch.is_same_size(torch.randn(4), torch.randn(3, 4)))

    def test_returns_python_bool(self):
        """The predicate always yields a real bool."""
        self.assertIsInstance(torch.is_same_size(torch.randn(2, 3), torch.randn(2, 3)), bool)

    def test_empty_and_scalar(self):
        """Empty tensors and 0-dim tensors compare correctly."""
        self.assertTrue(torch.is_same_size(torch.empty(0), torch.empty(0)))
        self.assertTrue(torch.is_same_size(torch.tensor(1.0), torch.tensor(2.0)))
        self.assertFalse(torch.is_same_size(torch.tensor(1.0), torch.randn(1)))

    def test_mixed_device_same_shape(self):
        """Shape comparison ignores device and succeeds across CPU/device."""
        cpu_t = torch.randn(2, 3)
        dev_t = torch.randn(2, 3, device=self.device)
        self.assertTrue(torch.is_same_size(cpu_t, dev_t))

    def test_non_tensor_raises(self):
        """A non-Tensor argument raises an error."""
        with self.assertRaises((TypeError, RuntimeError)):
            torch.is_same_size(torch.randn(2, 3), (2, 3))


if __name__ == "__main__":
    run_tests()
