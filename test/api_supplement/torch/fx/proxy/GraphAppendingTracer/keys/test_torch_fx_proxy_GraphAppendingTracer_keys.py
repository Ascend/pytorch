# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.proxy.GraphAppendingTracer.keys 接口功能正确性
API 名称：torch.fx.proxy.GraphAppendingTracer.keys
API 签名：GraphAppendingTracer.keys(self, obj) -> Proxy

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的新图、空字典或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Graph/Node/Proxy 和 Python 标量/容器 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空字典与重复调用独立节点 |
| 正常传参场景 | 已覆盖：keys 方法生成 IR 并返回实际字典键序列 |
| 异常传参场景 | 不适用：有效 FX 图和 Proxy |
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

from torch.fx.proxy import GraphAppendingTracer


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_keys_keys_call_is_recorded_and_evaluates_dictionary_keys(self):
        graph = torch.fx.Graph()
        tracer = GraphAppendingTracer(graph)
        value = tracer.create_proxy("placeholder", "mapping", (), {})
        result = tracer.keys(value)
        self.assertEqual(result.node.op, "call_method")
        self.assertEqual(result.node.target, "keys")
        self.assertEqual(result.node.args, (value.node,))
        graph.output(result.node)
        graph.lint()
        module = torch.fx.GraphModule({}, graph)
        self.assertEqual(list(module({"left": 1, "right": 2})), ["left", "right"])

    def test_keys_empty_dictionary_and_repeated_calls_make_distinct_nodes(self):
        graph = torch.fx.Graph()
        tracer = GraphAppendingTracer(graph)
        value = tracer.create_proxy("placeholder", "mapping", (), {})
        first = tracer.keys(value)
        second = tracer.keys(value)
        self.assertIsNot(first.node, second.node)
        graph.output((first.node, second.node))
        module = torch.fx.GraphModule({}, graph)
        self.assertEqual(tuple(list(item) for item in module({})), ([], []))


if __name__ == "__main__":
    run_tests()
