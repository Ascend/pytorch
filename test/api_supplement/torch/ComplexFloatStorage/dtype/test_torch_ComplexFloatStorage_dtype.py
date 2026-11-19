# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ComplexFloatStorage.dtype 接口功能正确性
API 名称：torch.ComplexFloatStorage.dtype
API 签名：ComplexFloatStorage.dtype -> torch.dtype（类属性）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 类型属性，无输入 | 未覆盖（属性无入参） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 与 torch.complex64 相等、实例 dtype 一致 | 已覆盖 |
| 正常传参场景 | 返回该存储对应的 dtype | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- ComplexFloatStorage.dtype 为类属性（纯框架状态，A 类），无入参与设备执行路径，不做 CPU/NPU 数值对照。
- 异常传参/混合设备：属性访问无入参、不接收 Tensor，相关维度不适用。
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
class TestComplexFloatStoragedtype(TestCase):
    """Test cases for torch.ComplexFloatStorage.dtype (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_complexfloatstorage_dtype_value(self):
        """The storage class exposes its documented dtype."""
        self.assertEqual(torch.ComplexFloatStorage.dtype, torch.complex64)

    def test_complexfloatstorage_dtype_is_torch_dtype(self):
        """The attribute is a torch.dtype instance."""
        self.assertIsInstance(torch.ComplexFloatStorage.dtype, torch.dtype)

    def test_complexfloatstorage_dtype_str(self):
        """The dtype string form is stable and matches torch.complex64."""
        self.assertEqual(str(torch.ComplexFloatStorage.dtype), 'torch.complex64')

    def test_complexfloatstorage_instance_reports_dtype(self):
        """A constructed storage instance reports the same dtype."""
        self.assertEqual(torch.ComplexFloatStorage(4).dtype, torch.ComplexFloatStorage.dtype)

    def test_complexfloatstorage_tensor_dtype_matches(self):
        """A tensor built with this dtype keeps it."""
        self.assertEqual(torch.zeros(3, dtype=torch.ComplexFloatStorage.dtype).dtype, torch.complex64)


if __name__ == "__main__":
    run_tests()
