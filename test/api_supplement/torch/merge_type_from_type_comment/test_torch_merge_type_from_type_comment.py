# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.merge_type_from_type_comment 接口功能正确性
API 名称：torch.merge_type_from_type_comment
API 签名：torch.merge_type_from_type_comment(decl, type_comment_decl, is_method) -> Decl

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入函数声明与类型注释声明 | 已覆盖 |
| 枚举选项 | is_method 为 True/False 两种 | 已覆盖 |
| 参数类型 | Decl/Decl/bool 三参数 | 已覆盖 |
| 传参与不传参 | 三个参数均为必填 | 已覆盖 |
| 等价类/边界值 | 注释数等于参数数时合并成功 | 已覆盖 |
| 正常传参场景 | 合并后返回带类型标注的 Decl | 已覆盖 |
| 异常传参场景 | 注释数与参数数不一致抛 RuntimeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只处理 JIT 源码树结构、不接收 Tensor，混合设备维度不适用。
- 合并结果的字符串结构随版本可变，故只断言合并成功、方法首参被跳过及异常契约，
  不对具体树形文本做硬编码断言。
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
class Testmerge_type_from_type_comment(TestCase):
    """Test cases for torch.merge_type_from_type_comment (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_merge_function_returns_decl(self):
        """A matching type comment merges into a typed Decl."""
        func_def = torch._C._parse_source_def('def f(x, y):\n    pass\n')
        type_comment = torch.parse_type_comment('# type: (int, float) -> int')
        merged = torch.merge_type_from_type_comment(func_def.decl(), type_comment, False)
        self.assertEqual(type(merged).__name__, 'Decl')

    def test_merge_method_skips_self(self):
        """With is_method=True the leading self parameter is not annotated."""
        method_def = torch._C._parse_source_def('def m(self, x):\n    pass\n')
        type_comment = torch.parse_type_comment('# type: (int) -> int')
        merged = torch.merge_type_from_type_comment(method_def.decl(), type_comment, True)
        self.assertEqual(type(merged).__name__, 'Decl')

    def test_annotation_count_mismatch_raises(self):
        """Too few annotations for the parameters raises RuntimeError."""
        func_def = torch._C._parse_source_def('def f(x, y):\n    pass\n')
        type_comment = torch.parse_type_comment('# type: (int) -> int')
        with self.assertRaises(RuntimeError):
            torch.merge_type_from_type_comment(func_def.decl(), type_comment, False)


if __name__ == "__main__":
    run_tests()
