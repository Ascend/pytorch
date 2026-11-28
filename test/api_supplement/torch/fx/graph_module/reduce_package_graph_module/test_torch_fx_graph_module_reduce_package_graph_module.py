# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.graph_module.reduce_package_graph_module 接口功能正确性
API 名称：torch.fx.graph_module.reduce_package_graph_module
API 签名：reduce_package_graph_module(importer, body, generated_module_name) -> torch.nn.Module

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：有/无节点栈和额外属性 |
| 正常传参场景 | 已覆盖：真实内存包导出/导入后重建及执行 |
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
import io
from torch.package import PackageExporter, PackageImporter
from torch.fx.graph_module import reduce_package_graph_module


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _roundtrip(self, with_stack):
        module = torch.fx.symbolic_trace(lambda value: value * 3 - 2)
        module.user_tag = "packaged"
        if with_stack:
            for node in module.graph.nodes:
                if node.op == "call_function":
                    node.meta["stack_trace"] = "ut original graph source"
        module_type = importlib.import_module("torch.fx.graph_module")._CodeOnlyModule
        original = module_type.forward
        self.addCleanup(setattr, module_type, "forward", original)
        buffer = io.BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.extern("**")
            function, arguments = module.__reduce_package__(exporter)
        self.assertIs(function, reduce_package_graph_module)
        body, generated_module = arguments
        buffer.seek(0)
        restored = reduce_package_graph_module(PackageImporter(buffer), body, generated_module)
        return restored

    def test_reduce_package_graph_module_real_package_importer_reconstructs_executable_graph(self):
        restored = self._roundtrip(False)
        self.assertEqual(restored(4), 10)
        self.assertEqual(restored(-1), -5)
        self.assertEqual(restored.user_tag, "packaged")
        restored.graph.lint()

    def test_reduce_package_graph_module_recorded_node_stack_traces_survive_package_reconstruction(self):
        restored = self._roundtrip(True)
        nodes = [node for node in restored.graph.nodes if node.op == "call_function"]
        self.assertTrue(nodes)
        for node in nodes:
            self.assertEqual(node.meta["stack_trace"], "ut original graph source")
        self.assertEqual(restored(0), -2)


if __name__ == "__main__":
    run_tests()
