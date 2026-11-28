# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.operator_support.create_op_support 接口功能正确性
API 名称：torch.fx.passes.operator_support.create_op_support
API 签名：create_op_support(is_node_supported) -> torch.fx.passes.operator_support.OperatorSupportBase

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空支持链、初始状态或无迭代边界 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 节点或带可观测状态的 Python pass 回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：包装实例独立和拒绝结果 |
| 正常传参场景 | 已覆盖：节点及子模块映射原样传入回调 |
| 异常传参场景 | 已覆盖：回调异常不被包装器吞掉 |
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

from torch.fx.passes.operator_support import create_op_support


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_create_op_support_forwards_node_and_submodule_mapping(self):
        graph = torch.fx.Graph()
        node = graph.call_module("activation", (graph.placeholder("x"),))
        modules = {"activation": torch.nn.ReLU()}
        calls = []

        def predicate(actual_modules, actual_node):
            calls.append((actual_modules, actual_node))
            return isinstance(actual_modules[actual_node.target], torch.nn.ReLU)

        support = create_op_support(predicate)
        self.assertTrue(support.is_node_supported(modules, node))
        self.assertIs(calls[0][0], modules)
        self.assertIs(calls[0][1], node)
        self.assertFalse(support.is_node_supported({"activation": torch.nn.Identity()}, node))

    def test_create_op_support_independent_wrappers_and_callback_exception(self):
        node = torch.fx.Graph().placeholder("x")
        accept = create_op_support(lambda modules, current: current.op == "placeholder")
        reject = create_op_support(lambda modules, current: False)
        self.assertTrue(accept.is_node_supported({}, node))
        self.assertFalse(reject.is_node_supported({}, node))

        def fail(modules, current):
            raise ValueError("predicate rejected input")

        with self.assertRaisesRegex(ValueError, "predicate rejected input"):
            create_op_support(fail).is_node_supported({}, node)


if __name__ == "__main__":
    run_tests()
