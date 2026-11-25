# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.key_averages 接口功能正确性
API 名称：torch.profiler.profile.key_averages
API 签名：profile.key_averages(self, group_by_input_shape=False, group_by_stack_n=0, group_by_overload_name=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 未启动状态与已采集非空事件聚合 | 已覆盖（before_start_raises、default_key_averages） |
| 枚举选项 | 布尔分组开关与 include_python_functions | 已覆盖（group_by_input_shape、group_by_overload_name、include_python_functions） |
| 参数类型 | bool 分组标志与 int 栈深度 group_by_stack_n | 已覆盖（group_by_input_shape、group_by_stack_n） |
| 传参与不传参 | 全默认参数与各分组参数显式传入 | 已覆盖（default_key_averages、group_by_input_shape、group_by_stack_n、group_by_overload_name） |
| 等价类/边界值 | 分组与不分组数量关系及栈帧截断上限 | 已覆盖（group_by_input_shape、group_by_stack_n） |
| 正常传参场景 | EventList 按形状/栈/重载聚合 | 已覆盖（default_key_averages、group_by_input_shape、group_by_stack_n） |
| 异常传参场景 | 未启动 profiler 调用触发 AssertionError | 已覆盖（before_start_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为事件统计聚合方法，不接受 Tensor 输入 |

未覆盖项及原因：
- 混合设备类型：该接口仅对已采集的 profiler 事件做分组统计，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import inspect
import os
import unittest

import torch
from torch.autograd.profiler_util import EventList
from torch.profiler import ProfilerActivity

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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
class TestTorchProfilerProfileKeyAverages(TestCase):
    """Test cases for torch.profiler.profile.key_averages."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_workload(self):
        x1 = torch.randn(16, 16, device=self.device)
        x2 = torch.randn(32, 32, device=self.device)
        return (x1 @ x1).sum() + (x2 @ x2).sum()

    def test_default_key_averages(self):
        """key_averages() returns a non-empty EventList with positive event counts."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
        averages = prof.key_averages()
        self.assertIsInstance(averages, EventList)
        self.assertGreater(len(averages), 0)
        self.assertGreater(sum(evt.count for evt in averages), 0)

    def test_group_by_input_shape(self):
        """group_by_input_shape=True preserves input shapes and groups accordingly."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            self._run_workload()
        grouped = prof.key_averages(group_by_input_shape=True)
        ungrouped = prof.key_averages(group_by_input_shape=False)
        self.assertGreaterEqual(len(grouped), len(ungrouped))
        self.assertTrue(any(len(evt.input_shapes) > 0 for evt in grouped))

    def test_group_by_stack_n(self):
        """group_by_stack_n caps stack depth for each aggregated event."""
        exp_cfg = torch._C._profiler._ExperimentalConfig(verbose=True)
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], with_stack=True, experimental_config=exp_cfg
        ) as prof:
            self._run_workload()
        grouped = prof.key_averages(group_by_stack_n=2)
        non_empty_stacks = [evt.stack for evt in grouped if len(evt.stack) > 0]
        self.assertGreater(len(non_empty_stacks), 0)
        for stack in non_empty_stacks:
            self.assertLessEqual(len(stack), 2)

    def test_group_by_overload_name(self):
        """group_by_overload_name=True returns a valid EventList for overloaded ops."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            x = torch.randn(16, 16, device=self.device)
            _ = torch.add(x, x)
            _ = torch.add(x, 1.0)
        grouped = prof.key_averages(group_by_overload_name=True)
        self.assertIsInstance(grouped, EventList)
        self.assertGreater(len(grouped), 0)

    def test_include_python_functions(self):
        """include_python_functions toggles Python function aggregation when supported."""
        if "include_python_functions" not in inspect.signature(torch.profiler.profile.key_averages).parameters:
            self.skipTest("include_python_functions parameter is not present in this torch version")
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            self._run_workload()
        self.assertGreater(len(prof.key_averages(include_python_functions=False)), 0)
        self.assertGreater(len(prof.key_averages(include_python_functions=True)), 0)

    def test_before_start_raises(self):
        """Calling key_averages() on an unstarted profiler raises AssertionError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(AssertionError):
            prof.key_averages()


if __name__ == "__main__":
    run_tests()
