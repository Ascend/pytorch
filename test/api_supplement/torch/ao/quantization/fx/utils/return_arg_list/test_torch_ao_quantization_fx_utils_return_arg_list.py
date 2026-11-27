# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.return_arg_list 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.return_arg_list
API 签名：return_arg_list(arg_indices) -> Callable[[Node], list[int]]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | arg_indices 为空列表与多个索引 | 已覆盖：test_empty_indices_returns_empty / test_keeps_indices_within_args |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | 入参为 list[int]，返回 Callable，内部入参为 FX Node | 已覆盖：test_returns_callable |
| 传参与不传参 | arg_indices 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 索引等于 node.args 长度（越界） | 已覆盖：test_filters_indices_exceeding_args |
| 正常传参场景 | 返回按 node.args 长度过滤的索引列表 | 已覆盖：test_keeps_indices_within_args |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯索引过滤，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：返回的函数对任意 Node 均返回列表，无稳定异常路径。
- 枚举选项 / 传参与不传参：该 API 无枚举参数、无可选参数。
- 混合设备类型：纯索引过滤工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回列表的内容与顺序断言。
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
from torch.ao.quantization.fx.utils import return_arg_list


def _make_node(num_args):
    graph = torch.fx.Graph()
    inp = graph.placeholder("inp")
    args = (inp,) + tuple(0 for _ in range(num_args - 1))
    node = graph.call_method("view", args)
    graph.output(node)
    return node


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestReturnArgListUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.return_arg_list."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_callable(self):
        func = return_arg_list([1, 2])
        self.assertTrue(callable(func))

    def test_keeps_indices_within_args(self):
        func = return_arg_list([0, 1, 2])
        node = _make_node(2)
        self.assertEqual(func(node), [0, 1])

    def test_filters_indices_exceeding_args(self):
        func = return_arg_list([1, 5])
        node = _make_node(3)
        self.assertEqual(func(node), [1])

    def test_empty_indices_returns_empty(self):
        func = return_arg_list([])
        node = _make_node(3)
        self.assertEqual(func(node), [])

    def test_preserves_order(self):
        func = return_arg_list([2, 1, 0])
        node = _make_node(3)
        self.assertEqual(func(node), [2, 1, 0])

    def test_index_equal_to_args_length_is_excluded(self):
        func = return_arg_list([2])
        node = _make_node(2)
        # node.args has length 2, so index 2 is out of range and must be dropped.
        self.assertEqual(func(node), [])


if __name__ == "__main__":
    run_tests()
