# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.graph_module.reduce_graph_module 接口功能正确性
API 名称：torch.fx.graph_module.reduce_graph_module
API 签名：reduce_graph_module(body, import_block) -> torch.nn.Module

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：旧 code 字段及零/负标量输入 |
| 正常传参场景 | 已覆盖：源码重建图、执行和额外属性保持 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

import importlib
from torch.fx.graph_module import reduce_graph_module


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _reduction(self):
        module = torch.fx.symbolic_trace(lambda value: value * 2 + 1)
        module.user_tag = "preserved"
        module_type = importlib.import_module("torch.fx.graph_module")._CodeOnlyModule
        original = module_type.forward
        self.addCleanup(setattr, module_type, "forward", original)
        function, arguments = module.__reduce__()
        self.assertIs(function, reduce_graph_module)
        return arguments

    def test_reduce_graph_module_generated_code_reconstruction_preserves_execution_and_attributes(self):
        body, imports = self._reduction()
        restored = reduce_graph_module(body, imports)
        restored.graph.lint()
        self.assertEqual(restored(3), 7)
        self.assertEqual(restored(-2), -3)
        self.assertEqual(restored.user_tag, "preserved")

    def test_reduce_graph_module_legacy_code_key_reconstructs_equivalent_graph(self):
        body, imports = self._reduction()
        body["code"] = body.pop("_code")
        restored = reduce_graph_module(body, imports)
        self.assertEqual(restored(0), 1)
        self.assertEqual(restored(5), 11)
        self.assertTrue(any(node.op == "call_function" for node in restored.graph.nodes))


if __name__ == "__main__":
    run_tests()
