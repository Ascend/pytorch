# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.is_deterministic_algorithms_warn_only_enabled 接口功能正确性
API 名称：torch.is_deterministic_algorithms_warn_only_enabled
API 签名：torch.is_deterministic_algorithms_warn_only_enabled() -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无入参 | 未覆盖（无入参） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | warn_only 开/关两种状态均可观测 | 已覆盖 |
| 正常传参场景 | 返回 warn_only 布尔状态 | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 无入参、不接收 Tensor，参数与混合设备维度不适用。
- 异常传参场景：无入参，未找到稳定异常路径。
- 本 API 仅查询确定性算法配置，不产生浮点输出，故不做跨设备数值对照；
  全局状态在 tearDown 中还原。
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
class Testis_deterministic_algorithms_warn_only_enabled(TestCase):
    """Test cases for torch.is_deterministic_algorithms_warn_only_enabled."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._enabled = torch.are_deterministic_algorithms_enabled()
        self._warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def tearDown(self):
        try:
            torch.use_deterministic_algorithms(self._enabled, warn_only=self._warn_only)
        finally:
            super().tearDown()

    def test_returns_bool(self):
        """The query yields a real bool."""
        self.assertIsInstance(torch.is_deterministic_algorithms_warn_only_enabled(), bool)

    def test_true_when_warn_only_set(self):
        """Enabling deterministic mode with warn_only=True is reported."""
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_false_when_warn_only_unset(self):
        """The enforce (non warn-only) mode reports False."""
        torch.use_deterministic_algorithms(True, warn_only=False)
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


if __name__ == "__main__":
    run_tests()
