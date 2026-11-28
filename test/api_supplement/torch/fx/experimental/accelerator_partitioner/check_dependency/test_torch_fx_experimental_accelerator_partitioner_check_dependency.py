# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.accelerator_partitioner.check_dependency 接口功能正确性
API 名称：torch.fx.experimental.accelerator_partitioner.check_dependency
API 签名：check_dependency(partition) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空图、空约束或无依赖状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX Graph/Node、类型对象及对应接口参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：自环及不经过根的下游环 |
| 正常传参场景 | 已覆盖：无环链与返回根分区的环 |
| 异常传参场景 | 不适用：有效图和元数据输入 |
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

from torch.fx.experimental.accelerator_partitioner import Partition, check_dependency
from torch.fx.passes.graph_manipulation import size_bytes


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _partitions(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        first = graph.call_function(torch.neg, (source,))
        second = graph.call_function(torch.abs, (first,))
        last = graph.call_function(torch.neg, (second,))
        independent = graph.call_function(torch.abs, (source,))
        for node in graph.nodes:
            node.size_bytes = size_bytes(output_size=16, total_size=16)
        partitions = [Partition(index + 10) for index in range(3)]
        for partition, nodes in zip(partitions, ({source, first}, {second}, {last})):
            partition.nodes = nodes
            partition.recalculate_mem_size()
        return partitions, (source, first, second, last, independent)


    def test_check_dependency_acyclic_chain_and_root_cycle(self):
        partitions, _ = self._partitions()
        first, second, third = partitions
        first.children = {second}
        second.children = {third}
        self.assertFalse(check_dependency(first))
        third.children = {first}
        self.assertTrue(check_dependency(first))

    def test_check_dependency_self_cycle_and_cycle_not_reaching_root(self):
        partitions, _ = self._partitions()
        first, second, third = partitions
        first.children = {first}
        self.assertTrue(check_dependency(first))
        first.children = {second}
        second.children = {third}
        third.children = {second}
        self.assertFalse(check_dependency(first))


if __name__ == "__main__":
    run_tests()
