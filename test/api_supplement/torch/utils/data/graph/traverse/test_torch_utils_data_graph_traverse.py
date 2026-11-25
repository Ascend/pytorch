# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.graph.traverse 接口功能正确性
API 名称：torch.utils.data.graph.traverse
API 签名：torch.utils.data.graph.traverse(datapipe, only_datapipe=None)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 单节点空子图与链式 DataPipe 非空子图       | 已覆盖（single_datapipe、chained_datapipe）            |
| 枚举选项         | only_datapipe 布尔选项（True/False）       | 已覆盖（only_datapipe_options）                        |
| 参数类型         | IterDataPipe 对象与 bool 类型参数          | 已覆盖（single_datapipe、only_datapipe_options）       |
| 传参与不传参     | 省略 only_datapipe 与显式传入布尔值        | 已覆盖（single_datapipe、only_datapipe_options）       |
| 等价类/边界值    | 单节点、链式节点与 FutureWarning 弃用告警  | 已覆盖（single_datapipe、chained_datapipe、emits_future_warning） |
| 正常传参场景     | 正确返回 {id: (dp, subgraph)} 图结构       | 已覆盖（single_datapipe、chained_datapipe、only_datapipe_options） |
| 异常传参场景     | 非 DataPipe 输入抛 RuntimeError 与缺参     | 已覆盖（invalid_input_raises、missing_arg_raises）     |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（DataPipe 图遍历函数，不接收 Tensor 输入）      |

未覆盖项及原因：
- 混合设备类型：接口用于遍历 DataPipe 依赖图，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import operator
import os
import unittest
import warnings

import torch
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.graph import traverse

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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDataGraphTraverse(TestCase):
    """Test cases for torch.utils.data.graph.traverse."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_single_datapipe(self):
        """Traverses a single source DataPipe and returns an empty subgraph."""
        src = IterableWrapper([1, 2, 3])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            graph = traverse(src)
        self.assertIsInstance(graph, dict)
        self.assertIn(id(src), graph)
        dp_obj, sub_graph = graph[id(src)]
        self.assertIs(dp_obj, src)
        self.assertEqual(sub_graph, {})

    def test_chained_datapipe(self):
        """Traverses a chained DataPipe and captures upstream DataPipe in subgraph."""
        src = IterableWrapper([1, 2, 3])
        mapped = src.map(operator.neg)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            graph = traverse(mapped)
        self.assertIn(id(mapped), graph)
        root_dp, sub_graph = graph[id(mapped)]
        self.assertIs(root_dp, mapped)
        self.assertIn(id(src), sub_graph)
        self.assertIs(sub_graph[id(src)][0], src)

    def test_only_datapipe_options(self):
        """Accepts both True and False for the only_datapipe option."""
        src = IterableWrapper([1, 2])
        mapped = src.map(operator.neg)
        for flag in (True, False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                graph = traverse(mapped, only_datapipe=flag)
            self.assertIn(id(mapped), graph)
            self.assertIn(id(src), graph[id(mapped)][1])

    def test_emits_future_warning(self):
        """Emits a FutureWarning when deprecated traverse is called."""
        src = IterableWrapper([1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            traverse(src, only_datapipe=True)
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in caught))

    def test_invalid_input_raises(self):
        """Passing a non-DataPipe object to traverse raises RuntimeError."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError):
                traverse([1, 2, 3])

    def test_missing_arg_raises(self):
        """Calling traverse without datapipe raises TypeError."""
        with self.assertRaises(TypeError):
            traverse()


if __name__ == "__main__":
    run_tests()
