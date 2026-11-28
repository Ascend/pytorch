# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.proxy.GraphAppendingTracer 接口功能正确性
API 名称：torch.fx.proxy.GraphAppendingTracer
API 签名：GraphAppendingTracer(graph)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的新图、空字典或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Graph/Node/Proxy 和 Python 标量/容器 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：两个 tracer/graph 实例独立 |
| 正常传参场景 | 已覆盖：在已有图追加算术节点与 users 关系 |
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
import operator


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_graphappendingtracer_appends_to_supplied_graph_without_losing_existing_nodes(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        tracer = GraphAppendingTracer(graph)
        proxy = tracer.proxy(source)
        result = proxy + 2
        self.assertIs(tracer.graph, graph)
        self.assertEqual(list(graph.nodes), [source, result.node])
        self.assertIs(result.node.target, operator.add)
        self.assertEqual(result.node.args, (source, 2))
        self.assertIn(result.node, source.users)

    def test_graphappendingtracer_tracers_and_graphs_remain_independent(self):
        first_graph = torch.fx.Graph()
        second_graph = torch.fx.Graph()
        first = GraphAppendingTracer(first_graph)
        second = GraphAppendingTracer(second_graph)
        value = first.create_proxy("placeholder", "value", (), {})
        result = value * 3
        self.assertEqual(list(second_graph.nodes), [])
        self.assertEqual(len(list(first_graph.nodes)), 2)
        self.assertIs(result.tracer, first)
        self.assertIs(second.graph, second_graph)


if __name__ == "__main__":
    run_tests()
