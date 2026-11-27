# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.all_node_args_except_first 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.all_node_args_except_first
API 签名：all_node_args_except_first(node)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | args 为空/单元素/多元素 | 已覆盖：test_multi_args_returns_tail 等 |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | 入参为 FX Node | 已覆盖：test_returns_list_of_ints |
| 传参与不传参 | node 为唯一必填参数 | 不适用 |
| 等价类/边界值 | args 长度为 0/1/2/3 的边界 | 已覆盖：test_index_count_equals_args_minus_one |
| 正常传参场景 | 由真实图节点取全部尾随索引 | 已覆盖：test_multi_args_returns_tail |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯索引计算，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：该函数仅读取 node.args 长度并返回区间列表，任何 Node 输入均可返回结果，无稳定异常路径。
- 枚举选项 / 传参与不传参：该 API 无枚举参数、无可选参数。
- 混合设备类型：纯 Python 索引计算，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图结构工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回列表的内容、类型与长度断言。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

import torch.fx
from torch.ao.quantization.fx.utils import all_node_args_except_first


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAllNodeArgsExceptFirstUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.all_node_args_except_first."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _make_node_with_args(self, count):
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        args = (inp,) + tuple(0 for _ in range(count - 1))
        node = graph.call_method("view", args)
        graph.output(node)
        return node

    def test_no_args_returns_empty(self):
        node = self._make_node_with_args(0)
        self.assertEqual(all_node_args_except_first(node), [])

    def test_single_arg_returns_empty(self):
        node = self._make_node_with_args(1)
        self.assertEqual(all_node_args_except_first(node), [])

    def test_multi_args_returns_tail(self):
        node = self._make_node_with_args(3)
        self.assertEqual(all_node_args_except_first(node), [1, 2])

    def test_returned_values_are_consecutive_ints(self):
        node = self._make_node_with_args(4)
        result = all_node_args_except_first(node)
        self.assertEqual(result, [1, 2, 3])
        for index in result:
            self.assertIsInstance(index, int)

    def test_index_count_equals_args_minus_one(self):
        for count in (0, 1, 2, 5):
            node = self._make_node_with_args(count)
            result = all_node_args_except_first(node)
            self.assertEqual(len(result), max(len(node.args) - 1, 0))

    def test_returns_list_of_ints(self):
        node = self._make_node_with_args(2)
        result = all_node_args_except_first(node)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1])


if __name__ == "__main__":
    run_tests()
