# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_autocast_cpu_enabled 接口功能正确性
API 名称：torch.set_autocast_cpu_enabled
API 签名：torch.set_autocast_cpu_enabled(enabled) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | enabled 为必填布尔 | 未覆盖（不适用） |
| 枚举选项 | True/False 两个取值 | 已覆盖 |
| 参数类型 | 布尔入参 | 已覆盖 |
| 传参与不传参 | enabled 为必填参数 | 已覆盖 |
| 等价类/边界值 | 开/关状态切换可被查询 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收布尔值、不接收 Tensor，空值与混合设备维度不适用。
- 异常传参场景：CPU 开关对入参做真值语义处理，未找到稳定异常路径。
- 本 API 面向 CPU autocast 开关，NPU/CUDA 路径由 set_autocast_enabled 覆盖；
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
class Testset_autocast_cpu_enabled(TestCase):
    """Test cases for torch.set_autocast_cpu_enabled (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved = torch.is_autocast_enabled('cpu')

    def tearDown(self):
        try:
            torch.set_autocast_enabled('cpu', self._saved)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """The setter returns None."""
        self.assertIsNone(torch.set_autocast_cpu_enabled(True))

    def test_enable_then_query_true(self):
        """Enabling CPU autocast makes the query return True."""
        torch.set_autocast_cpu_enabled(True)
        self.assertTrue(torch.is_autocast_cpu_enabled())

    def test_disable_then_query_false(self):
        """Disabling CPU autocast makes the query return False."""
        torch.set_autocast_cpu_enabled(True)
        torch.set_autocast_cpu_enabled(False)
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def test_toggle_roundtrip(self):
        """The flag tracks the last set value."""
        torch.set_autocast_cpu_enabled(False)
        self.assertFalse(torch.is_autocast_cpu_enabled())
        torch.set_autocast_cpu_enabled(True)
        self.assertTrue(torch.is_autocast_cpu_enabled())


if __name__ == "__main__":
    run_tests()
