# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.splitter_base.NodeEventTracker.dump 接口功能正确性
API 名称：torch.fx.passes.splitter_base.NodeEventTracker.dump
API 签名：NodeEventTracker.dump(self) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的无依赖事件、空 tracker 或未知节点 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Node、事件描述和可选依赖节点 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：mode 0/1/2/3 和单/多事件节点 |
| 正常传参场景 | 已覆盖：不落文件、全量文件及选定节点递归文件 |
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

from torch.fx.passes.splitter_base import NodeEventTracker

from pathlib import Path
import tempfile
from unittest import mock
from torch.fx.passes.splitter_base import (
    ALL_SUFFIX, NODES_SUFFIX, ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES,
)


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


    def test_dump_mode_zero_and_all_events_file(self):
        source, negative = self._nodes()
        with tempfile.TemporaryDirectory(prefix="node-event-dump-") as folder:
            prefix = str(Path(folder) / "events")
            tracker = NodeEventTracker(0, prefix)
            tracker.add(source, "origin")
            tracker.add(negative, "consume", source)
            tracker.dump()
            self.assertEqual(list(Path(folder).iterdir()), [])
            tracker.tracker_mode = 1
            tracker.dump()
            expected = (f"Node: {source.name}:\n  {source.name}: origin #\n"
                        f"Node: {negative.name}:\n  {negative.name}: consume {source.name}\n")
            self.assertEqual(Path(prefix + ALL_SUFFIX).read_text(), expected)
            self.assertFalse(Path(prefix + NODES_SUFFIX).exists())

    def test_dump_explicit_and_multi_event_node_selection(self):
        source, negative = self._nodes()
        with tempfile.TemporaryDirectory(prefix="node-event-selected-") as folder:
            prefix = str(Path(folder) / "events")
            tracker = NodeEventTracker(2, prefix)
            tracker.add(source, "origin")
            tracker.add(negative, "consume", source)
            tracker.add(negative, "revisit")
            with mock.patch.dict(os.environ, {ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES: source.name}):
                tracker.dump()
            selected = Path(prefix + NODES_SUFFIX).read_text()
            self.assertIn(f"===== Tracking node {source.name} =====", selected)
            self.assertIn(f"|-{source.name}: origin #", selected)
            self.assertNotIn(f"===== Tracking node {negative.name} =====", selected)
            tracker.tracker_mode = 3
            tracker.dump()
            selected = Path(prefix + NODES_SUFFIX).read_text()
            self.assertIn(f"===== Tracking node {negative.name} =====", selected)
            self.assertIn(f"| |-{source.name}: origin #", selected)
            self.assertIn(f"|-{negative.name}: revisit #", selected)
            self.assertNotIn(f"===== Tracking node {source.name} =====", selected)


if __name__ == "__main__":
    run_tests()
