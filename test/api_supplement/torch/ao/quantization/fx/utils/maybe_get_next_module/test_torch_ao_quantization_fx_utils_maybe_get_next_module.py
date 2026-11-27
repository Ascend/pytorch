# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.maybe_get_next_module 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.maybe_get_next_module
API 签名：maybe_get_next_module(node, modules, target_module_type=None, target_functional_type=None) -> Node | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | node 无后继用户与有后继用户 | 已覆盖：test_no_users_returns_none / test_returns_functional_user_node |
| 枚举选项 | 按 module 类型匹配 / 按 functional 匹配 | 已覆盖：test_returns_module_user_node / test_returns_functional_user_node |
| 参数类型 | modules 为 dict，target_* 为类型/函数 | 已覆盖：test_returns_module_user_node |
| 传参与不传参 | 两 target 参数均可省略（默认 None） | 已覆盖：test_no_target_returns_none |
| 等价类/边界值 | 匹配命中与未命中两类 | 已覆盖：test_returns_none_when_type_mismatch |
| 正常传参场景 | 返回匹配的首个用户节点 | 已覆盖：test_returns_functional_user_node |
| 异常传参场景 | 未命中返回 None（非异常） | 已覆盖：test_returns_none_when_type_mismatch |
| 混合设备类型 | 纯图结构遍历，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：未命中返回 None 而非抛异常，故无 assertRaises 路径。
- 混合设备类型：纯图结构遍历工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回用户节点的同一性与 None 边界断言。
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
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import maybe_get_next_module


def _build_functional_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    obs = graph.call_module("obs", (x,))
    relu = graph.call_function(F.relu, (obs,))
    graph.output(relu)
    root = nn.Module()
    root.obs = nn.ReLU()
    gm = torch.fx.GraphModule(root, graph)
    return gm, obs, relu


def _build_module_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu = graph.call_module("relu", (x,))
    linear = graph.call_module("linear", (relu,))
    graph.output(linear)
    root = nn.Module()
    root.relu = nn.ReLU()
    root.linear = nn.Linear(3, 3)
    gm = torch.fx.GraphModule(root, graph)
    return gm, relu, linear


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestMaybeGetNextModuleUt(TestCase):
    """Functional tests for maybe_get_next_module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_functional_user_node(self):
        gm, obs, relu = _build_functional_graph()
        modules = dict(gm.named_modules())
        result = maybe_get_next_module(obs, modules, target_functional_type=F.relu)
        self.assertIs(result, relu)

    def test_returns_module_user_node(self):
        gm, relu, linear = _build_module_graph()
        modules = dict(gm.named_modules())
        result = maybe_get_next_module(relu, modules, target_module_type=nn.Linear)
        self.assertIs(result, linear)

    def test_returns_none_when_type_mismatch(self):
        gm, relu, _ = _build_module_graph()
        modules = dict(gm.named_modules())
        self.assertIsNone(maybe_get_next_module(relu, modules, target_module_type=nn.Conv2d))

    def test_returns_none_when_no_functional_match(self):
        gm, obs, _ = _build_functional_graph()
        modules = dict(gm.named_modules())
        self.assertIsNone(maybe_get_next_module(obs, modules, target_functional_type=F.gelu))

    def test_no_target_returns_none(self):
        gm, obs, _ = _build_functional_graph()
        modules = dict(gm.named_modules())
        self.assertIsNone(maybe_get_next_module(obs, modules))

    def test_no_users_returns_none(self):
        gm, _, relu = _build_functional_graph()
        modules = dict(gm.named_modules())
        # The relu node has no users that match a requested type.
        self.assertIsNone(maybe_get_next_module(relu, modules, target_functional_type=F.relu))


if __name__ == "__main__":
    run_tests()
