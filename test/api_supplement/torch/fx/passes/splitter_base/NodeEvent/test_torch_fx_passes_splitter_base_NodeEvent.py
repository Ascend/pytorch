# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.splitter_base.NodeEvent 接口功能正确性
API 名称：torch.fx.passes.splitter_base.NodeEvent
API 签名：NodeEvent(source, desc, dep=None)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的无依赖事件、空 tracker 或未知节点 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Node、事件描述和可选依赖节点 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空描述和依赖节点重命名 |
| 正常传参场景 | 已覆盖：有/无依赖的事件内容和字符串序列化 |
| 异常传参场景 | 不适用：使用有效、无环的事件依赖 |
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

from torch.fx.passes.splitter_base import NodeEvent


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _nodes(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        negative = graph.call_function(torch.neg, (source,))
        return source, negative


    def test_nodeevent_default_and_explicit_dependency(self):
        source, negative = self._nodes()
        root = NodeEvent(source, "create|input")
        dependent = NodeEvent(negative, "move|consumer", source)
        self.assertEqual(root.to_str(), f"{source.name}: create|input #")
        self.assertEqual(dependent.to_str(), f"{negative.name}: move|consumer {source.name}")
        self.assertIs(dependent.source, negative)
        self.assertIs(dependent.dep, source)

    def test_nodeevent_empty_description_and_current_dependency_name(self):
        source, negative = self._nodes()
        event = NodeEvent(negative, "", source)
        source.name = "renamed_input"
        self.assertEqual(event.to_str(), f"{negative.name}:  renamed_input")
        self.assertEqual(event.desc, "")


if __name__ == "__main__":
    run_tests()
