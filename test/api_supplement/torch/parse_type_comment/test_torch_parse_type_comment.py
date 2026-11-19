# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.parse_type_comment 接口功能正确性
API 名称：torch.parse_type_comment
API 签名：torch.parse_type_comment(arg0: str) -> torch._C._jit_tree_views.Decl

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入非空类型注释字符串 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 字符串入参 | 已覆盖 |
| 传参与不传参 | arg0 为必填参数 | 已覆盖 |
| 等价类/边界值 | 无参/有参/有返回值三种注释 | 已覆盖 |
| 正常传参场景 | 返回解析后的 Decl 结构 | 已覆盖 |
| 异常传参场景 | 非法注释文本抛 RuntimeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只解析类型注释文本、不接收 Tensor，混合设备维度不适用。
- Decl 的文本表示随版本可变，故只断言返回类型与异常契约，不硬编码树形文本。
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
class Testparse_type_comment(TestCase):
    """Test cases for torch.parse_type_comment (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_decl_for_typed_params(self):
        """A parameter/return type comment parses into a Decl."""
        decl = torch.parse_type_comment('# type: (int, float) -> int')
        self.assertEqual(type(decl).__name__, 'Decl')

    def test_no_arg_type_comment(self):
        """An empty-parameter type comment still parses."""
        decl = torch.parse_type_comment('# type: () -> ()')
        self.assertEqual(type(decl).__name__, 'Decl')

    def test_invalid_comment_raises(self):
        """A comment without the type-comment marker raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.parse_type_comment('(int) -> int')


if __name__ == "__main__":
    run_tests()
