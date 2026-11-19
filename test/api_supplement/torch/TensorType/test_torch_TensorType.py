# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.TensorType 接口功能正确性
API 名称：torch.TensorType
API 签名：torch.TensorType（类型对象，不可实例化）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 类型对象，无输入 | 未覆盖（类型对象无入参） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 与 torch.Tensor 区分、父类 torch.Type | 已覆盖 |
| 正常传参场景 | name/module/父类关系符合契约 | 已覆盖 |
| 异常传参场景 | 直接实例化报 TypeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- torch.TensorType 为遗留类型对象（纯框架状态，A 类），不接收 Tensor、无设备执行路径，
  故不做 CPU/NPU 数值对照，也不构造张量执行路径。
- 空/参数/传参/混合设备：类型对象无入参、不接收 Tensor，相关维度不适用。
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
class TesttorchTensorType(TestCase):
    """Test cases for torch.TensorType (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_tensortype_is_class(self):
        """torch.TensorType is exported as a class object."""
        self.assertIsInstance(torch.TensorType, type)

    def test_tensortype_name_and_module(self):
        """The type keeps its public name and torch module."""
        self.assertEqual(torch.TensorType.__name__, 'TensorType')
        self.assertEqual(torch.TensorType.__module__, 'torch')

    def test_tensortype_parent_is_torch_type(self):
        """torch.TensorType derives from the torch.Type base."""
        self.assertTrue(issubclass(torch.TensorType, torch.Type))

    def test_tensortype_distinct_from_tensor(self):
        """TensorType is not Tensor; a plain tensor is not a TensorType instance."""
        self.assertIsNot(torch.Tensor, torch.TensorType)
        self.assertFalse(issubclass(torch.Tensor, torch.TensorType))
        self.assertFalse(isinstance(torch.zeros(3), torch.TensorType))

    def test_tensortype_not_instantiable(self):
        """The legacy type object has no Python constructor."""
        with self.assertRaises(TypeError):
            torch.TensorType()


if __name__ == "__main__":
    run_tests()
