# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.split_utils.move_non_tensor_nodes_on_boundary 接口功能正确性
API 名称：torch.fx.passes.split_utils.move_non_tensor_nodes_on_boundary
API 签名：move_non_tensor_nodes_on_boundary(subgraphs) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：非 acc 源图不可移动与空输入 |
| 正常传参场景 | 已覆盖：跨分区标量链移动且张量节点保留 |
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

import operator
from torch.fx.passes.split_utils import move_non_tensor_nodes_on_boundary
from torch.fx.passes.splitter_base import Subgraph


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _fixture(self):
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        size = graph.call_function(len, (value,))
        dimension = graph.call_function(operator.add, (size, 1))
        consumer = graph.call_function(operator.mul, (dimension, 2))
        tensor = graph.call_function(torch.neg, (value,))
        size.meta["type"] = int
        dimension.meta["type"] = int
        consumer.meta["type"] = int
        tensor.meta["type"] = torch.Tensor
        return size, dimension, consumer, tensor

    def test_move_non_tensor_nodes_on_boundary_non_tensor_chain_moves_to_its_unique_consumer_partition(self):
        size, dimension, consumer, tensor = self._fixture()
        groups = [Subgraph(True, [size, dimension, tensor]), Subgraph(False, [consumer])]
        move_non_tensor_nodes_on_boundary(groups)
        self.assertEqual(set(groups[0].nodes), {tensor})
        self.assertEqual(set(groups[1].nodes), {size, dimension, consumer})
        self.assertEqual(len(groups[1].nodes), 3)

    def test_move_non_tensor_nodes_on_boundary_non_acc_source_and_empty_partitions_are_unchanged(self):
        size, dimension, consumer, tensor = self._fixture()
        groups = [Subgraph(False, [size, dimension, tensor]), Subgraph(True, [consumer])]
        before = [tuple(group.nodes) for group in groups]
        move_non_tensor_nodes_on_boundary(groups)
        self.assertEqual([tuple(group.nodes) for group in groups], before)
        self.assertIsNone(move_non_tensor_nodes_on_boundary([]))


if __name__ == "__main__":
    run_tests()
