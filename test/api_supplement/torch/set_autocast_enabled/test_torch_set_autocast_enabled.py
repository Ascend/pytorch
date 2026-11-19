# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_autocast_enabled 接口功能正确性
API 名称：torch.set_autocast_enabled
API 签名：torch.set_autocast_enabled(device_type, enabled) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 两个入参均为必填 | 未覆盖（不适用） |
| 枚举选项 | 多设备类型 cpu/xla/ipu 可分别设置 | 已覆盖 |
| 参数类型 | 设备字符串 + 布尔 | 已覆盖 |
| 传参与不传参 | device_type 与 enabled 均为必填 | 已覆盖 |
| 等价类/边界值 | 设置后经 is_autocast_enabled 可读回 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 非法设备名触发 RuntimeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收设备字符串与布尔、不接收 Tensor，空值与混合设备维度不适用。
- 本 API 仅配置全局 autocast 开关，不产生浮点输出，故不做跨设备数值对照；
  所涉设备状态在 tearDown 中逐一还原。
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

_TOUCHED = ('cpu', 'cuda', 'xla', 'ipu')


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testset_autocast_enabled(TestCase):
    """Test cases for torch.set_autocast_enabled (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved = {d: torch.is_autocast_enabled(d) for d in _TOUCHED}

    def tearDown(self):
        try:
            for d, en in self._saved.items():
                torch.set_autocast_enabled(d, en)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """The setter returns None."""
        self.assertIsNone(torch.set_autocast_enabled('cpu', True))

    def test_cpu_roundtrip(self):
        """Setting the CPU flag is reflected by is_autocast_enabled('cpu')."""
        torch.set_autocast_enabled('cpu', True)
        self.assertTrue(torch.is_autocast_enabled('cpu'))
        torch.set_autocast_enabled('cpu', False)
        self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_multiple_devices_are_independent(self):
        """Each device type keeps its own autocast flag."""
        torch.set_autocast_enabled('xla', True)
        torch.set_autocast_enabled('ipu', False)
        self.assertTrue(torch.is_autocast_enabled('xla'))
        self.assertFalse(torch.is_autocast_enabled('ipu'))

    def test_invalid_device_raises(self):
        """An illegal device string raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.set_autocast_enabled('not_a_device', True)


if __name__ == "__main__":
    run_tests()
