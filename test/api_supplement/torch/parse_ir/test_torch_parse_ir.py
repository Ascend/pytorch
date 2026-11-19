# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.parse_ir 接口功能正确性
API 名称：torch.parse_ir
API 签名：torch.parse_ir(input: str, parse_tensor_constants: bool = False) -> torch._C.Graph

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入非空图 IR 字符串 | 已覆盖 |
| 枚举选项 | parse_tensor_constants 为 True/False | 已覆盖 |
| 参数类型 | 字符串 + 可选布尔 | 已覆盖 |
| 传参与不传参 | 默认不传 parse_tensor_constants | 已覆盖 |
| 等价类/边界值 | 单算子图与含返回图均可解析 | 已覆盖 |
| 正常传参场景 | 返回可枚举节点的 Graph | 已覆盖 |
| 异常传参场景 | 非法 IR 抛 RuntimeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只解析文本 IR、不接收 Tensor，混合设备维度不适用。
- parse_tensor_constants=True 的常量张量还原路径需要含张量常量的 IR，未构造，
  仅验证其在普通 IR 上不报错。
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

_ADD_IR = 'graph(%a : int, %b : int):\n  %c : int = aten::add(%a, %b)\n  return (%c)'


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testparse_ir(TestCase):
    """Test cases for torch.parse_ir (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_graph_with_expected_node(self):
        """Parsing a two-input add graph yields one aten::add node."""
        graph = torch.parse_ir(_ADD_IR)
        self.assertEqual(type(graph).__name__, 'Graph')
        kinds = [node.kind() for node in graph.nodes()]
        self.assertIn('aten::add', kinds)

    def test_parse_tensor_constants_flag(self):
        """The optional flag is accepted and still returns a Graph."""
        graph = torch.parse_ir(_ADD_IR, True)
        self.assertEqual(type(graph).__name__, 'Graph')

    def test_roundtrip_string_is_stable(self):
        """str(graph) reproduces a reparsable IR."""
        graph = torch.parse_ir(_ADD_IR)
        reparsed = torch.parse_ir(str(graph))
        self.assertEqual(len(list(graph.nodes())), len(list(reparsed.nodes())))

    def test_invalid_ir_raises(self):
        """Malformed IR text raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.parse_ir('this is not a graph')


if __name__ == "__main__":
    run_tests()
