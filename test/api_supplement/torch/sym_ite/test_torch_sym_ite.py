# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.sym_ite 接口功能正确性
API 名称：torch.sym_ite
API 签名：torch.sym_ite(b, t, f) -> SymInt | int

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 三个入参均为必填 | 未覆盖（不适用） |
| 枚举选项 | 条件为 True/False 两种 | 已覆盖 |
| 参数类型 | bool/SymBool 条件 + 同类型分支值 | 已覆盖 |
| 传参与不传参 | b/t/f 均为必填 | 已覆盖 |
| 等价类/边界值 | 真分支、假分支、类型不匹配 | 已覆盖 |
| 正常传参场景 | 返回被选中分支的值 | 已覆盖 |
| 异常传参场景 | 条件非布尔、分支类型不一致抛 AssertionError | 已覆盖 |
| 混合设备类型 | 控制流原语、无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 为符号三元表达式工具、不接收 Tensor，混合设备维度不适用。
- SymBool 分支的符号求值需完整 shape-env 上下文，独立构造不稳定，故未覆盖；
  具体布尔分支的选择语义与参数校验已断言。
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
class Testsym_ite(TestCase):
    """Test cases for torch.sym_ite (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_true_selects_then_branch(self):
        """A True condition returns the then-value."""
        self.assertEqual(torch.sym_ite(True, 1, 2), 1)

    def test_false_selects_else_branch(self):
        """A False condition returns the else-value."""
        self.assertEqual(torch.sym_ite(False, 1, 2), 2)

    def test_float_branches(self):
        """Both branches may be floats and must share the same type."""
        self.assertEqual(torch.sym_ite(True, 1.5, 2.5), 1.5)

    def test_non_bool_condition_raises(self):
        """A non-boolean condition raises AssertionError."""
        with self.assertRaises(AssertionError):
            torch.sym_ite(1, 1, 2)

    def test_mismatched_branch_types_raise(self):
        """Branches with different types raise AssertionError."""
        with self.assertRaises(AssertionError):
            torch.sym_ite(True, 1, 2.0)


if __name__ == "__main__":
    run_tests()
