# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.all_node_args_have_no_tensors 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.all_node_args_have_no_tensors
API 签名：all_node_args_have_no_tensors(node, modules, cache)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | cache 传空 dict 与预填充 dict | 已覆盖：test_cache_is_populated |
| 枚举选项 | 覆盖 placeholder/get_attr/call_method/call_function/call_module 分支 | 已覆盖：多个 test_* |
| 参数类型 | node 为 Node、modules 为 dict、cache 为 dict | 已覆盖：test_non_node_argument_returns_true |
| 传参与不传参 | 三参数均为必填 | 不适用 |
| 等价类/边界值 | 纯 int 与含张量两条等价类 | 已覆盖：test_pure_int_call_function_returns_true 等 |
| 正常传参场景 | 真实图节点上判定是否含张量 | 已覆盖：test_placeholder_returns_false 等 |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 仅做图结构判定，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：该函数对任意输入均返回布尔值，无稳定异常路径。
- 枚举选项全覆盖：分支与激活后处理模块（observer）耦合，未穷举所有模块类型。
- 混合设备类型：纯图结构判定，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图结构工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回布尔值与缓存副作用断言。
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
import torch.nn as nn
from torch.ao.quantization.fx.utils import all_node_args_have_no_tensors


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAllNodeArgsHaveNoTensorsUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.all_node_args_have_no_tensors."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _graph(self):
        return torch.fx.Graph()

    def test_placeholder_returns_false(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        self.assertFalse(all_node_args_have_no_tensors(inp, {}, {}))

    def test_get_attr_returns_false(self):
        graph = self._graph()
        attr = graph.get_attr("buf")
        self.assertFalse(all_node_args_have_no_tensors(attr, {}, {}))

    def test_size_call_method_returns_true(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        size = graph.call_method("size", (inp, 0))
        self.assertTrue(all_node_args_have_no_tensors(size, {}, {}))

    def test_ndim_getattr_returns_true(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        dim = graph.call_function(getattr, (inp, "ndim"))
        self.assertTrue(all_node_args_have_no_tensors(dim, {}, {}))

    def test_pure_int_call_function_returns_true(self):
        graph = self._graph()
        node = graph.call_function(operator.add, (1, 2))
        self.assertTrue(all_node_args_have_no_tensors(node, {}, {}))

    def test_call_function_with_tensor_returns_false(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        add = graph.call_function(operator.add, (inp, inp))
        self.assertFalse(all_node_args_have_no_tensors(add, {}, {}))

    def test_call_module_non_observer_returns_false(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        call = graph.call_module("relu", (inp,))
        modules = {"relu": nn.ReLU()}
        self.assertFalse(all_node_args_have_no_tensors(call, modules, {}))

    def test_non_node_argument_returns_true(self):
        # The function short-circuits to True for anything that is not a Node.
        self.assertTrue(all_node_args_have_no_tensors(3, {}, {}))

    def test_cache_is_populated(self):
        graph = self._graph()
        inp = graph.placeholder("inp")
        size = graph.call_method("size", (inp, 0))
        # The cache is only consulted/written when it is already non-empty, so
        # seed it with an unrelated entry.
        cache = {"seed": True}
        result = all_node_args_have_no_tensors(size, {}, cache)
        self.assertTrue(result)
        self.assertIn(size, cache)
        self.assertEqual(cache[size], result)


if __name__ == "__main__":
    run_tests()
