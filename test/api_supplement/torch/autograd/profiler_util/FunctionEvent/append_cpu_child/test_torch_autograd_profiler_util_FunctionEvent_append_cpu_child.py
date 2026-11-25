# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.FunctionEvent.append_cpu_child 接口功能正确性
API 名称：torch.autograd.profiler_util.FunctionEvent.append_cpu_child
API 签名：FunctionEvent.append_cpu_child(self, child: FunctionEvent) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 初始空子列表追加单个与多个子事件 | 已覆盖（append_single_child、append_multiple_children） |
| 枚举选项 | DeviceType.CPU 与非 CPU 设备事件校验 | 已覆盖（append_single_child、error_cases_raise） |
| 参数类型 | 合法 FunctionEvent、str、None 与缺参 | 已覆盖（append_single_child、error_cases_raise） |
| 传参与不传参 | 必选 child 传入与缺参报错 | 已覆盖（append_single_child、error_cases_raise） |
| 等价类/边界值 | 单子事件与多子事件累计扣减 self_cpu_time | 已覆盖（append_single_child、append_multiple_children） |
| 正常传参场景 | 追加 CPU 子事件并更新子列表与自耗时 | 已覆盖（append_single_child、append_multiple_children） |
| 异常传参场景 | 非 CPU 父/子事件、非法类型与缺参报错 | 已覆盖（error_cases_raise） |
| 混合设备类型 | CPU 与非 CPU 事件混搭触发 AssertionError | 已覆盖（error_cases_raise） |

未覆盖项及原因：
- 无

注意：本接口维护 FunctionEvent 调用树子事件列表，无浮点 Tensor 输出，
     按 API 语义使用子列表引用、self_cpu_time_total 及异常断言验证。
"""
import os
import unittest

import torch
from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent

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


def _make_event(event_id, name, start_us=0, end_us=100, device_type=DeviceType.CPU):
    return FunctionEvent(id=event_id, name=name, thread=1, start_us=start_us, end_us=end_us, device_type=device_type)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFunctionEventAppendCpuChild(TestCase):
    """Test cases for torch.autograd.profiler_util.FunctionEvent.append_cpu_child."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_append_single_child(self):
        """Appending a CPU child records it in cpu_children and updates self_cpu_time_total."""
        parent = _make_event(1, "parent", 0, 100)
        child = _make_event(2, "child", 10, 40)
        self.assertEqual(len(parent.cpu_children), 0)
        self.assertEqual(parent.self_cpu_time_total, 100)
        self.assertIsNone(parent.append_cpu_child(child))
        self.assertEqual(len(parent.cpu_children), 1)
        self.assertIs(parent.cpu_children[0], child)
        self.assertEqual(parent.self_cpu_time_total, 70)

    def test_append_multiple_children(self):
        """Appending multiple CPU children preserves order and subtracts all child times."""
        parent = _make_event(1, "parent", 0, 100)
        children = [
            _make_event(2, "child_1", 5, 25),
            _make_event(3, "child_2", 30, 65),
            _make_event(4, "child_3", 70, 80),
        ]
        for child in children:
            parent.append_cpu_child(child)
        self.assertEqual(len(parent.cpu_children), 3)
        for idx, child in enumerate(children):
            self.assertIs(parent.cpu_children[idx], child)
        self.assertEqual(parent.self_cpu_time_total, 35)

    def test_error_cases_raise(self):
        """Non-CPU parent/child, invalid child types, and missing arg raise expected errors."""
        gpu_parent = _make_event(1, "gpu_parent", 0, 100, device_type=DeviceType.CUDA)
        with self.assertRaises(AssertionError):
            gpu_parent.append_cpu_child(_make_event(2, "cpu_child", 10, 20))
        self.assertEqual(len(gpu_parent.cpu_children), 0)

        cpu_parent = _make_event(1, "cpu_parent", 0, 100)
        for bad_child in ("not_an_event", None, _make_event(2, "gpu_child", 10, 20, device_type=DeviceType.CUDA)):
            with self.assertRaises(AssertionError):
                cpu_parent.append_cpu_child(bad_child)
        with self.assertRaises(TypeError):
            cpu_parent.append_cpu_child()
        self.assertEqual(len(cpu_parent.cpu_children), 0)


if __name__ == "__main__":
    run_tests()
