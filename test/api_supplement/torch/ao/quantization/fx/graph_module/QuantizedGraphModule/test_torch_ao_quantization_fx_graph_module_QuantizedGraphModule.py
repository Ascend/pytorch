# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.graph_module.QuantizedGraphModule 接口功能正确性
API 名称：torch.ao.quantization.fx.graph_module.QuantizedGraphModule
API 签名：QuantizedGraphModule(root, graph, preserved_attr_names)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | preserved_attr_names 取空集与含属性两种 | 已覆盖：test_empty_preserved_names_ok / test_preserves_named_attributes |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | root 为 nn.Module、graph 为 Graph、preserved 为 set | 已覆盖：test_constructor_returns_graph_module_subclass |
| 传参与不传参 | 三个位置参数均为必填 | 不适用 |
| 等价类/边界值 | 单节点图；preserved 属性名不存在于 root | 已覆盖：test_missing_preserved_attr_ignored |
| 正常传参场景 | 前向执行、state_dict 往返 | 已覆盖：test_forward_executes_graph 等 |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 无多 Tensor 设备输入路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：构造函数对非法图结构不做显式校验，异常均来自 GraphModule 基类内部，非本 API 稳定契约，故不单列。
- 枚举选项 / 传参与不传参：该 API 无枚举参数、无可选参数。
- 混合设备类型：纯图结构与状态字典工具，不涉及设备迁移，无 CPU/NPU 混合输入路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注意：本 API 为纯 Python 图模块工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回类型、结构、状态字典内容与异常路径断言。
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

import copy

import torch.fx
import torch.nn as nn
from torch.ao.quantization.fx import graph_module as graph_module_mod
from torch.ao.quantization.fx.graph_module import QuantizedGraphModule


def _build_graph():
    graph = torch.fx.Graph()
    inp = graph.placeholder("inp")
    neg = graph.call_function(torch.neg, (inp,))
    graph.output(neg)
    return graph


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestQuantizedGraphModuleUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.graph_module.QuantizedGraphModule."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_constructor_returns_graph_module_subclass(self):
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        self.assertIsInstance(module, QuantizedGraphModule)
        self.assertIsInstance(module, torch.fx.GraphModule)

    def test_empty_preserved_names_ok(self):
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        self.assertEqual(module.preserved_attr_names, set())

    def test_preserves_named_attributes(self):
        root = nn.Module()
        root.preserved_attr = 123
        module = QuantizedGraphModule(root, _build_graph(), {"preserved_attr"})
        self.assertEqual(module.preserved_attr, 123)
        # The attribute must be carried over by reference, not copied.
        self.assertIs(module.preserved_attr, root.preserved_attr)

    def test_missing_preserved_attr_ignored(self):
        root = nn.Module()
        module = QuantizedGraphModule(root, _build_graph(), {"not_there"})
        self.assertFalse(hasattr(module, "not_there"))

    def test_forward_executes_graph(self):
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        out = module(torch.tensor([1.0, -2.0, 3.0]))
        self.assertTrue(torch.equal(out, torch.tensor([-1.0, 2.0, -3.0])))

    def test_state_dict_hook_registered(self):
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        hooks = list(module._state_dict_hooks.values())
        self.assertIn(graph_module_mod._save_packed_weight, hooks)

    def test_packed_weight_state_dict_round_trip(self):
        packed = torch.ops.quantized.linear_prepack_fp16(torch.randn(4, 4))
        self.assertIsInstance(packed, torch._C.ScriptObject)
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        module._packed_weight_linear = packed
        state = module.state_dict()
        self.assertIn("_packed_weight_linear", state)
        restored = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        restored.load_state_dict(state)
        self.assertTrue(hasattr(restored, "_packed_weight_linear"))
        self.assertIsInstance(restored._packed_weight_linear, torch._C.ScriptObject)

    def test_deepcopy_returns_quantized_graph_module(self):
        module = QuantizedGraphModule(nn.Module(), _build_graph(), set())
        cloned = copy.deepcopy(module)
        self.assertIsInstance(cloned, QuantizedGraphModule)
        self.assertIsInstance(cloned, torch.fx.GraphModule)


if __name__ == "__main__":
    run_tests()
