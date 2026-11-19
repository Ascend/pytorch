# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.get_device 接口功能正确性
API 名称：torch.get_device
API 签名：torch.get_device(input: Tensor) -> int

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入非空 Tensor | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | Tensor 入参，非 Tensor 触发 TypeError | 已覆盖 |
| 传参与不传参 | input 为必填参数 | 已覆盖 |
| 等价类/边界值 | CPU 返回 -1、加速器返回设备索引 | 已覆盖 |
| 正常传参场景 | 返回设备序号整型 | 已覆盖 |
| 异常传参场景 | 非 Tensor 入参抛 TypeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入组合 | 未覆盖（单输入 API） |

未覆盖项及原因：
- 本 API 仅接收单个 Tensor，不存在混合设备输入，故该维度不适用。
- 加速器索引分支仅在存在 NPU/CUDA 设备时执行，否则跳过并说明原因。
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
class Testget_device(TestCase):
    """Test cases for torch.get_device (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_int(self):
        """The getter returns a Python int."""
        self.assertIsInstance(torch.get_device(torch.randn(3)), int)

    def test_cpu_tensor_returns_minus_one(self):
        """A CPU tensor reports the sentinel index -1."""
        self.assertEqual(torch.get_device(torch.randn(3)), -1)

    def test_accel_tensor_matches_tensor_get_device(self):
        """For an accelerator tensor the index equals Tensor.get_device()."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accelerator device')
        t = torch.randn(3, device=self.device_name)
        self.assertGreaterEqual(torch.get_device(t), 0)
        self.assertEqual(torch.get_device(t), t.get_device())

    def test_non_tensor_raises(self):
        """A non-Tensor argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch.get_device(1)


if __name__ == "__main__":
    run_tests()
