# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.FunctionEvent.set_cpu_parent 接口功能正确性
API 名称：torch.autograd.profiler_util.FunctionEvent.set_cpu_parent
API 签名：FunctionEvent.set_cpu_parent(self, parent: FunctionEvent) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 初始父指针为 None 与设置/覆盖父事件 | 已覆盖（set_cpu_parent_basic、overwrite_cpu_parent） |
| 枚举选项 | DeviceType.CPU 与非 CPU 设备事件校验 | 已覆盖（set_cpu_parent_basic、non_cpu_child_raises、non_cpu_parent_raises） |
| 参数类型 | 合法 FunctionEvent、str、None 与缺参 | 已覆盖（set_cpu_parent_basic、non_event_parent_raises、missing_arg_raises） |
| 传参与不传参 | 必选 parent 传入与缺参报错 | 已覆盖（set_cpu_parent_basic、missing_arg_raises） |
| 等价类/边界值 | 首次绑定、覆盖重绑及失败时状态不变 | 已覆盖（set_cpu_parent_basic、overwrite_cpu_parent、non_event_parent_raises） |
| 正常传参场景 | 绑定与覆盖 CPU 父事件并保持对象引用 | 已覆盖（set_cpu_parent_basic、overwrite_cpu_parent） |
| 异常传参场景 | 非 CPU/非事件/缺参 | 已覆盖（non_cpu_child_raises、non_event_parent_raises、non_cpu_parent_raises 等） |
| 混合设备类型 | CPU 与非 CPU 事件混搭触发 AssertionError | 已覆盖（non_cpu_child_raises、non_cpu_parent_raises） |

未覆盖项及原因：
- 无

注意：本接口设置 FunctionEvent 的 CPU 父事件引用，无浮点 Tensor 输出，
     按 API 语义使用返回值 None、cpu_parent 对象引用及异常断言验证。
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
    return FunctionEvent(
        id=event_id, name=name, thread=1, start_us=start_us, end_us=end_us, device_type=device_type,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFunctionEventSetCpuParent(TestCase):
    """Test cases for torch.autograd.profiler_util.FunctionEvent.set_cpu_parent."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_set_cpu_parent_basic(self):
        """Setting a CPU parent assigns cpu_parent by reference and returns None."""
        parent = _make_event(1, "parent", 0, 100)
        child = _make_event(2, "child", 10, 50)
        self.assertIsNone(child.cpu_parent)
        ret = child.set_cpu_parent(parent)
        self.assertIsNone(ret)
        self.assertIs(child.cpu_parent, parent)

    def test_overwrite_cpu_parent(self):
        """Calling set_cpu_parent again overwrites the existing cpu_parent reference."""
        p1 = _make_event(1, "parent_1", 0, 100)
        p2 = _make_event(2, "parent_2", 0, 200)
        child = _make_event(3, "child", 10, 50)
        child.set_cpu_parent(p1)
        self.assertIs(child.cpu_parent, p1)
        child.set_cpu_parent(p2)
        self.assertIs(child.cpu_parent, p2)

    def test_non_cpu_child_raises(self):
        """Calling set_cpu_parent on a non-CPU child raises AssertionError."""
        parent = _make_event(1, "cpu_parent", 0, 100, device_type=DeviceType.CPU)
        child = _make_event(2, "gpu_child", 10, 50, device_type=DeviceType.CUDA)
        with self.assertRaises(AssertionError):
            child.set_cpu_parent(parent)
        self.assertIsNone(child.cpu_parent)

    def test_non_event_parent_raises(self):
        """Passing a non-FunctionEvent parent raises AssertionError without mutating state."""
        p1 = _make_event(1, "cpu_parent", 0, 100)
        child = _make_event(2, "cpu_child", 10, 50)
        child.set_cpu_parent(p1)
        with self.assertRaises(AssertionError):
            child.set_cpu_parent("not_an_event")
        with self.assertRaises(AssertionError):
            child.set_cpu_parent(None)
        self.assertIs(child.cpu_parent, p1)

    def test_non_cpu_parent_raises(self):
        """Passing a non-CPU FunctionEvent parent raises AssertionError without mutating state."""
        p1 = _make_event(1, "cpu_parent", 0, 100)
        gpu_parent = _make_event(2, "gpu_parent", 0, 100, device_type=DeviceType.CUDA)
        child = _make_event(3, "cpu_child", 10, 50)
        child.set_cpu_parent(p1)
        with self.assertRaises(AssertionError):
            child.set_cpu_parent(gpu_parent)
        self.assertIs(child.cpu_parent, p1)

    def test_missing_arg_raises(self):
        """Calling set_cpu_parent without parent argument raises TypeError."""
        child = _make_event(1, "cpu_child", 10, 50)
        with self.assertRaises(TypeError):
            child.set_cpu_parent()
        self.assertIsNone(child.cpu_parent)


if __name__ == "__main__":
    run_tests()
