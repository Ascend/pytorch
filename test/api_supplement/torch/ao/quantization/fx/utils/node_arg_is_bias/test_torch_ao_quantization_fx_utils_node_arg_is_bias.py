# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.node_arg_is_bias 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.node_arg_is_bias
API 签名：node_arg_is_bias(node, arg) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无 target_dtype_info 与含该 meta 两种 | 已覆盖：test_no_meta_returns_false / test_true_for_bias_index_meta |
| 枚举选项 | 命中索引路径与命中 kwargs 路径 | 已覆盖：test_true_for_kwargs_bias |
| 参数类型 | arg 为 FX Node | 已覆盖：test_false_for_non_bias_arg |
| 传参与不传参 | node、arg 均为必填 | 不适用 |
| 等价类/边界值 | bias_index 越界（大于 args 长度） | 已覆盖：test_index_out_of_range_falls_back |
| 正常传参场景 | 按 meta 的 bias_index 命中偏置 | 已覆盖：test_true_for_bias_index_meta |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯图结构判定，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：函数对任意输入返回布尔值，无稳定异常路径。
- 枚举选项全覆盖：target_dtype_info 的键随量化后端变化，仅覆盖 bias_index 与 kwargs 两条路径。
- 混合设备类型：纯图结构判定，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回布尔值断言。
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
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import node_arg_is_bias


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestNodeArgIsBiasUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.node_arg_is_bias."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _graph_with_linear(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        b = graph.placeholder("b")
        linear = graph.call_function(F.linear, (x, b))
        return graph, x, b, linear

    def test_true_for_bias_index_meta(self):
        _, x, b, linear = self._graph_with_linear()
        linear.meta["target_dtype_info"] = {"bias_index": 1}
        self.assertTrue(node_arg_is_bias(linear, b))

    def test_false_for_non_bias_arg(self):
        _, x, b, linear = self._graph_with_linear()
        linear.meta["target_dtype_info"] = {"bias_index": 1}
        self.assertFalse(node_arg_is_bias(linear, x))

    def test_true_for_kwargs_bias(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        b = graph.placeholder("b")
        linear = graph.call_function(F.linear, (x,), {"bias": b})
        self.assertTrue(node_arg_is_bias(linear, b))

    def test_index_out_of_range_falls_back(self):
        _, x, b, linear = self._graph_with_linear()
        linear.meta["target_dtype_info"] = {"bias_index": 5}
        self.assertFalse(node_arg_is_bias(linear, b))

    def test_no_meta_returns_false(self):
        _, x, _, linear = self._graph_with_linear()
        self.assertFalse(node_arg_is_bias(linear, x))

    def test_non_node_arg_returns_false(self):
        _, _, _, linear = self._graph_with_linear()
        linear.meta["target_dtype_info"] = {"bias_index": 1}
        self.assertFalse(node_arg_is_bias(linear, 1))


if __name__ == "__main__":
    run_tests()
