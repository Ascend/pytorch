# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.graph.traverse_dps 接口功能正确性
API 名称：torch.utils.data.graph.traverse_dps
API 签名：torch.utils.data.graph.traverse_dps(datapipe)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 单节点空子图与多级管道非空嵌套子图         | 已覆盖（single_datapipe、chained_datapipe、three_stage_pipeline） |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | IterDataPipe 实例入参                      | 已覆盖（single_datapipe、chained_datapipe）            |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（datapipe 为唯一必填参数，无可选默认参数）      |
| 等价类/边界值    | 单级、两级与三级链式 DataPipe 拓扑         | 已覆盖（single_datapipe、chained_datapipe、three_stage_pipeline） |
| 正常传参场景     | 正确构建嵌套 DataPipe 图字典               | 已覆盖（single_datapipe、chained_datapipe、three_stage_pipeline） |
| 异常传参场景     | 非 DataPipe 输入抛 RuntimeError 与缺参     | 已覆盖（invalid_input_raises、missing_arg_raises）     |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（DataPipe 图结构遍历函数，不接收 Tensor 输入）  |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：datapipe 为唯一必填位置参数，无可选默认参数。
- 混合设备类型：接口用于遍历 DataPipe 管道图结构，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.graph import traverse_dps

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _double(x):
    return x * 2


def _add_one(x):
    return x + 1


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDataGraphTraverseDps(TestCase):
    """Test cases for torch.utils.data.graph.traverse_dps."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_single_datapipe(self):
        """Traverses a single source DataPipe into a one-entry graph with empty subgraph."""
        src = IterableWrapper([10, 20, 30])
        graph = traverse_dps(src)
        self.assertIsInstance(graph, dict)
        self.assertEqual(len(graph), 1)
        self.assertIn(id(src), graph)
        dp_obj, sub_graph = graph[id(src)]
        self.assertIs(dp_obj, src)
        self.assertEqual(sub_graph, {})

    def test_chained_datapipe(self):
        """Traverses a two-stage DataPipe pipeline and records upstream dependency."""
        src = IterableWrapper([1, 2, 3])
        mapped = src.map(_add_one)
        graph = traverse_dps(mapped)
        self.assertIn(id(mapped), graph)
        root_dp, sub_graph = graph[id(mapped)]
        self.assertIs(root_dp, mapped)
        self.assertIn(id(src), sub_graph)
        src_dp, src_sub = sub_graph[id(src)]
        self.assertIs(src_dp, src)
        self.assertEqual(src_sub, {})

    def test_three_stage_pipeline(self):
        """Traverses a three-stage DataPipe pipeline preserving nested graph hierarchy."""
        src = IterableWrapper([1, 2])
        mid = src.map(_double)
        end = mid.map(_add_one)
        graph = traverse_dps(end)
        self.assertIn(id(end), graph)
        mid_graph = graph[id(end)][1]
        self.assertIn(id(mid), mid_graph)
        src_graph = mid_graph[id(mid)][1]
        self.assertIn(id(src), src_graph)
        self.assertIs(src_graph[id(src)][0], src)

    def test_invalid_input_raises(self):
        """Passing a non-DataPipe object to traverse_dps raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            traverse_dps("not_a_datapipe")

    def test_missing_arg_raises(self):
        """Calling traverse_dps without datapipe raises TypeError."""
        with self.assertRaises(TypeError):
            traverse_dps()


if __name__ == "__main__":
    run_tests()
