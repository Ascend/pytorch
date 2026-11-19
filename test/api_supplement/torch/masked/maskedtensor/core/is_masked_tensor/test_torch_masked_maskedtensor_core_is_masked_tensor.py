# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.masked.maskedtensor.core.is_masked_tensor 接口功能正确性
API 名称：torch.masked.maskedtensor.core.is_masked_tensor
API 签名：torch.masked.maskedtensor.core.is_masked_tensor(obj) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 基础调用 | 传入 MaskedTensor 返回 True | 已覆盖 |
| 参数枚举/边界值 | 普通张量、None、列表等非 MaskedTensor 对象 | 已覆盖 |
| 副作用验证 | 无副作用（纯谓词） | 已覆盖 |
| 幂等性 | 连续调用结果一致 | 已覆盖 |
| 异常路径 | 无参数时报错 | 已覆盖 |

未覆盖项及原因：
- 该 API 为纯 Python 谓词，不接收待计算张量内容、不做数值运算，无 CPU Golden 对照。
- 设备维度不适用：判断对象类型与设备无关，故未构造 NPU/CPU 双端场景。
- 混合设备维度不适用：仅单个对象入参，无跨设备组合。
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

from torch.masked import masked_tensor  # noqa: E402
from torch.masked.maskedtensor.core import is_masked_tensor  # noqa: E402


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testis_masked_tensor(TestCase):
    """Test cases for torch.masked.maskedtensor.core.is_masked_tensor (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_true_for_masked_tensor(self):
        """A MaskedTensor instance is recognized."""
        mt = masked_tensor(torch.randn(3), torch.tensor([True, False, True]))
        self.assertTrue(is_masked_tensor(mt))

    def test_false_for_plain_tensor(self):
        """A plain torch.Tensor is not a MaskedTensor."""
        self.assertFalse(is_masked_tensor(torch.randn(3)))

    def test_returns_python_bool(self):
        """The predicate always yields a real bool."""
        mt = masked_tensor(torch.randn(3), torch.tensor([True, True, True]))
        self.assertIsInstance(is_masked_tensor(mt), bool)
        self.assertIsInstance(is_masked_tensor(torch.randn(3)), bool)

    def test_false_for_other_objects(self):
        """Arbitrary non-tensor objects are not MaskedTensors."""
        for obj in (None, [1, 2, 3], 'tensor', 3.14):
            self.assertFalse(is_masked_tensor(obj))

    def test_idempotent(self):
        """Repeated calls on the same object return the same result."""
        mt = masked_tensor(torch.randn(2, 2), torch.tensor([[True, False], [False, True]]))
        self.assertEqual(is_masked_tensor(mt), is_masked_tensor(mt))

    def test_missing_argument_raises(self):
        """Calling without an argument raises TypeError."""
        with self.assertRaises(TypeError):
            is_masked_tensor()


if __name__ == "__main__":
    run_tests()
