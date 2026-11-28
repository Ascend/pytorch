# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.graph_manipulation.get_size_of_node 接口功能正确性
API 名称：torch.fx.passes.graph_manipulation.get_size_of_node
API 签名：get_size_of_node(fx_module, node) -> torch.fx.passes.graph_manipulation.size_bytes

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空节点集合或无匹配状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 图、模块、schema 及接口默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：float32/float16 与零 batch |
| 正常传参场景 | 已覆盖：参数加输出的内存字节计算 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 参数类型：未覆盖量化元数据分支。
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

from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.passes.shape_prop import ShapeProp


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _linear(self, shape, dtype):
        root = torch.nn.Sequential(torch.nn.Linear(3, 4, device="meta", dtype=dtype))
        module = torch.fx.symbolic_trace(root)
        ShapeProp(module).propagate(torch.empty(shape, dtype=dtype, device="meta"))
        node = next(node for node in module.graph.nodes if node.op == "call_module")
        return get_size_of_node(module, node)

    def test_get_size_of_node_parameter_and_output_byte_counts_for_two_dtypes(self):
        for dtype, itemsize in ((torch.float32, 4), (torch.float16, 2)):
            with self.subTest(dtype=dtype):
                result = self._linear((2, 3), dtype)
                self.assertEqual(result.output_size, 8 * itemsize)
                self.assertEqual(result.total_size, (12 + 4 + 8) * itemsize)

    def test_get_size_of_node_empty_output_still_counts_parameters(self):
        result = self._linear((0, 3), torch.float32)
        self.assertEqual(result.output_size, 0)
        self.assertEqual(result.total_size, 64)


if __name__ == "__main__":
    run_tests()
