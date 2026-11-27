# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_non_observable_arg_indexes_and_types 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_non_observable_arg_indexes_and_types
API 签名：get_non_observable_arg_indexes_and_types(node) -> dict

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 命中 NON_OBSERVABLE_ARG_DICT 与未命中两种 | 已覆盖：test_known_size_node / test_unknown_node_returns_empty_dict |
| 枚举选项 | size/permute/masked_fill 等 NodeInfo 键 | 已覆盖：test_permute_returns_all_except_first 等 |
| 参数类型 | 入参为 FX Node | 已覆盖：test_known_size_node_returns_int_mapping |
| 传参与不传参 | node 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 返回的函数按 node.args 长度过滤索引 | 已覆盖：test_returned_func_filters_by_node_args |
| 正常传参场景 | 已知节点返回类型->取值函数映射 | 已覆盖：test_known_size_node_returns_int_mapping |
| 异常传参场景 | 未命中返回空字典（非异常） | 已覆盖：test_unknown_node_returns_empty_dict |
| 混合设备类型 | 纯图结构查询，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：未命中时返回共享空字典而非抛异常，故无 assertRaises 路径。
- 枚举选项全覆盖：NON_OBSERVABLE_ARG_DICT 中的键随版本变化，仅覆盖代表性的 size/permute/masked_fill。
- 混合设备类型：纯图结构查询，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回字典的键与取值函数结果断言。
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

import operator

import torch.fx
from torch.ao.quantization.fx.utils import (
    EMPTY_ARG_DICT,
    get_non_observable_arg_indexes_and_types,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetNonObservableArgIndexesAndTypesUt(TestCase):
    """Functional tests for get_non_observable_arg_indexes_and_types."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _placeholder_graph(self):
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        return graph, inp

    def test_known_size_node_returns_int_mapping(self):
        graph, inp = self._placeholder_graph()
        size = graph.call_method("size", (inp, 0))
        result = get_non_observable_arg_indexes_and_types(size)
        self.assertIn(int, result)
        self.assertEqual(result[int](size), [1])

    def test_unknown_node_returns_empty_dict(self):
        graph, inp = self._placeholder_graph()
        add = graph.call_function(operator.add, (inp, inp))
        result = get_non_observable_arg_indexes_and_types(add)
        self.assertEqual(result, {})
        self.assertIs(result, EMPTY_ARG_DICT)

    def test_permute_returns_all_except_first(self):
        graph, inp = self._placeholder_graph()
        permute = graph.call_method("permute", (inp, 0, 1, 2))
        result = get_non_observable_arg_indexes_and_types(permute)
        self.assertIn(int, result)
        self.assertEqual(result[int](permute), [1, 2, 3])

    def test_masked_fill_has_bool_and_float_keys(self):
        graph, inp = self._placeholder_graph()
        mask = graph.call_method("size", (inp,))
        fill = graph.call_method("masked_fill", (inp, mask, 0.0))
        result = get_non_observable_arg_indexes_and_types(fill)
        self.assertIn(torch.bool, result)
        self.assertIn(float, result)
        self.assertEqual(result[torch.bool](fill), [1])
        self.assertEqual(result[float](fill), [2])

    def test_returned_func_filters_by_node_args(self):
        graph, inp = self._placeholder_graph()
        size = graph.call_method("size", (inp,))
        result = get_non_observable_arg_indexes_and_types(size)
        # The size node has only one arg, so index 1 must be filtered out.
        self.assertEqual(result[int](size), [])


if __name__ == "__main__":
    run_tests()
