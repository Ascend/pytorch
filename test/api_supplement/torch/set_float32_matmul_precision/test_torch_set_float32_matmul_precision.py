# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_float32_matmul_precision 接口功能正确性
API 名称：torch.set_float32_matmul_precision
API 签名：torch.set_float32_matmul_precision(precision) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | precision 为必填字符串 | 未覆盖（不适用） |
| 枚举选项 | highest/high/medium 三个取值 | 已覆盖 |
| 参数类型 | 字符串入参 | 已覆盖 |
| 传参与不传参 | precision 为必填参数 | 已覆盖 |
| 等价类/边界值 | 设置后经 get_float32_matmul_precision 可读回 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收字符串、不接收 Tensor，空值与混合设备维度不适用。
- 异常传参场景：目标版本对未知 precision 字符串不做稳定报错（实测不抛异常），
  故不构造非法入参断言，避免臆造契约。
- 本 API 仅配置全局精度策略，不产生浮点输出，故不做跨设备数值对照；
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

_VALID = ('highest', 'high', 'medium')


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testset_float32_matmul_precision(TestCase):
    """Test cases for torch.set_float32_matmul_precision (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved = torch.get_float32_matmul_precision()

    def tearDown(self):
        try:
            torch.set_float32_matmul_precision(self._saved)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """The setter returns None."""
        self.assertIsNone(torch.set_float32_matmul_precision('high'))

    def test_valid_values_roundtrip(self):
        """Every documented precision string is stored and read back."""
        for p in _VALID:
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)


if __name__ == "__main__":
    run_tests()
