# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.QInt32Storage 接口功能正确性
API 名称：torch.QInt32Storage
API 签名：QInt32Storage(size=0) -> torch.QInt32Storage

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | size=0 空存储与 size=4 非空存储 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | size 为 int；非法 size 报错 | 已覆盖 |
| 传参与不传参 | 省略 size 与显式传入 | 已覆盖 |
| 等价类/边界值 | size=0 边界、负 size 报错 | 已覆盖 |
| 正常传参场景 | 构造出对应 dtype 的存储 | 已覆盖 |
| 异常传参场景 | 负 size 触发异常 | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- QInt32Storage 为类型化存储类（纯框架状态，A 类），不接收 Tensor、无设备执行路径，
  不做 CPU/NPU 数值对照；本类的 deprecation 告警为进程级 once，不做稳定断言。
- 混合设备类型：本 API 只涉及 CPU 存储对象，不涉及 NPU/CPU 混合输入。
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
class TestQInt32Storage(TestCase):
    """Test cases for torch.QInt32Storage (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_qint32storage_default_size_zero(self):
        """Omitting the size yields an empty (size 0) storage."""
        self.assertEqual(torch.QInt32Storage().size(), 0)

    def test_qint32storage_construct_size(self):
        """An explicit size is honored."""
        self.assertEqual(torch.QInt32Storage(4).size(), 4)

    def test_qint32storage_dtype(self):
        """The constructed storage carries the class dtype."""
        self.assertEqual(torch.QInt32Storage(4).dtype, torch.qint32)

    def test_qint32storage_element_size(self):
        """Element size matches the dtype byte width."""
        self.assertEqual(torch.QInt32Storage(4).element_size(), 4)

    def test_qint32storage_negative_size_raises(self):
        """A negative size must raise."""
        with self.assertRaises((RuntimeError, ValueError)):
            torch.QInt32Storage(-1)


if __name__ == "__main__":
    run_tests()
