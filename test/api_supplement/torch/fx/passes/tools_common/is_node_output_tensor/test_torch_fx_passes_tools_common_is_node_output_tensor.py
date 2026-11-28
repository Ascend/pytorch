# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.tools_common.is_node_output_tensor 接口功能正确性
API 名称：torch.fx.passes.tools_common.is_node_output_tensor
API 签名：is_node_output_tensor(node) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空集合、缺失元数据或无依赖节点 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Graph/Node 和相关结构参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无元数据和显式非 Tensor 类型 |
| 正常传参场景 | 已覆盖：真实 ShapeProp 的 Tensor、标量和 tuple 输出 |
| 异常传参场景 | 不适用：使用有效图或辅助对象 |
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

from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.tools_common import is_node_output_tensor


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_is_node_output_tensor_shape_propagation_distinguishes_tensor_and_scalar(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        result = graph.call_function(torch.neg, (source,))
        rank = graph.call_method("dim", (result,))
        graph.output((result, rank))
        module = torch.fx.GraphModule({}, graph)
        ShapeProp(module).propagate(torch.empty(2, 3, device="meta"))
        self.assertTrue(is_node_output_tensor(source))
        self.assertTrue(is_node_output_tensor(result))
        self.assertFalse(is_node_output_tensor(rank))
        self.assertFalse(is_node_output_tensor(graph.output_node()))

    def test_is_node_output_tensor_missing_and_explicit_non_tensor_metadata(self):
        node = torch.fx.Graph().placeholder("value")
        self.assertFalse(is_node_output_tensor(node))
        node.meta["type"] = int
        self.assertFalse(is_node_output_tensor(node))
        node.meta["type"] = torch.Tensor
        self.assertTrue(is_node_output_tensor(node))


if __name__ == "__main__":
    run_tests()
