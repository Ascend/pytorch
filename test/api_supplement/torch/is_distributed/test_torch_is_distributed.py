# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.is_distributed 接口功能正确性
API 名称：torch.is_distributed
API 签名：torch.is_distributed(input: Tensor) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入非空 CPU/加速器 Tensor | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | Tensor 入参，非 Tensor 触发 TypeError | 已覆盖 |
| 传参与不传参 | input 为必填参数 | 已覆盖 |
| 等价类/边界值 | 无 ProcessGroup 时返回 False | 已覆盖 |
| 正常传参场景 | 返回布尔分布式归属判定 | 已覆盖 |
| 异常传参场景 | 非 Tensor 入参抛 TypeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入组合 | 未覆盖（单输入 API） |

未覆盖项及原因：
- 本 API 仅接收单个 Tensor，不存在混合设备输入，故该维度不适用。
- 返回 True 的分支需要跨进程 ProcessGroup 环境，本用例不做多进程启动，
  故该分支未覆盖（无可用分布式后端时不伪造该路径）。
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
class Testis_distributed(TestCase):
    """Test cases for torch.is_distributed (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_bool(self):
        """The predicate yields a real bool."""
        self.assertIsInstance(torch.is_distributed(torch.randn(3)), bool)

    def test_cpu_tensor_not_distributed(self):
        """A plain CPU tensor has no ProcessGroup, so the result is False."""
        self.assertFalse(torch.is_distributed(torch.randn(3)))

    def test_accel_tensor_not_distributed(self):
        """An accelerator tensor outside a group still reports False."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accelerator device')
        self.assertFalse(torch.is_distributed(torch.randn(3, device=self.device_name)))

    def test_non_tensor_raises(self):
        """A non-Tensor argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch.is_distributed(1)


if __name__ == "__main__":
    run_tests()
