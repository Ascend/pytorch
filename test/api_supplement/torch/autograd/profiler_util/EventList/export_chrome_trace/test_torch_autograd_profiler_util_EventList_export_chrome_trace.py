# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.export_chrome_trace 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.export_chrome_trace
API 签名：EventList.export_chrome_trace(self, path) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 EventList 与非空事件导出 | 已覆盖（export_empty、export_cpu_events） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口仅接收路径参数 path） |
| 参数类型 | 合法路径字符串与缺参调用 | 已覆盖（export_cpu_events、missing_arg_raises） |
| 传参与不传参 | 必选 path 传入与不传参报错 | 已覆盖（export_cpu_events、missing_arg_raises） |
| 等价类/边界值 | 含 kernel 流事件、trace_name 为 None 过滤 | 已覆盖（export_with_kernels、export_skips_none_trace_name） |
| 正常传参场景 | 导出各类事件 JSON | 已覆盖（export_cpu_events、export_with_kernels、export_skips_none_trace_name、export_empty） |
| 异常传参场景 | 不存在目录写文件报错、缺参报错 | 已覆盖（invalid_dir_raises、missing_arg_raises） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（接口仅序列化事件列表到 JSON，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收输出路径参数 path，无枚举选项。
- 混合设备类型：接口将事件元数据导出为 Chrome Trace JSON，不接收 Tensor 参数。

注意：本接口将 EventList 导出为 Chrome Trace JSON 文件，无浮点 Tensor 输出，
     按 API 语义使用 JSON 内容、事件字段、返回值及异常断言验证。
"""
import json
import os
import tempfile
import unittest

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent

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


def _make_event(event_id, name, start_us=0, end_us=10, thread=1, trace_name=None):
    return FunctionEvent(id=event_id, name=name, trace_name=trace_name, thread=thread, start_us=start_us, end_us=end_us)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListExportChromeTrace(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.export_chrome_trace."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _export_and_load(self, event_list, filename):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, filename)
            ret = event_list.export_chrome_trace(trace_path)
            with open(trace_path, "r", encoding="utf-8") as f:
                return ret, json.load(f)

    def test_export_cpu_events(self):
        """Exporting CPU FunctionEvents writes valid Chrome Trace X events."""
        e1 = _make_event(1, "aten::add", 100, 160, thread=7, trace_name="aten::add")
        e2 = _make_event(2, "aten::mul", 200, 350, thread=8, trace_name="aten::mul")
        ret, data = self._export_and_load(EventList([e1, e2]), "trace.json")
        self.assertIsNone(ret)
        self.assertEqual(len(data), 2)
        self.assertEqual((data[0]["name"], data[0]["ph"], data[0]["ts"], data[0]["dur"]), ("aten::add", "X", 100, 60))
        self.assertEqual((data[1]["name"], data[1]["ts"], data[1]["dur"], data[1]["tid"]), ("aten::mul", 200, 150, 8))

    def test_export_with_kernels(self):
        """Events with appended kernels emit cpu_to_device flow start entries."""
        e1 = _make_event(1, "aten::mm", 50, 120, thread=3, trace_name="aten::mm")
        e1.append_kernel("kernel_a", 0, 30)
        e1.append_kernel("kernel_b", 0, 40)
        el = EventList([e1], use_device=self.device_name, profile_memory=True)
        _, data = self._export_and_load(el, "kernels.json")
        flows = [entry for entry in data if entry["ph"] == "s"]
        self.assertEqual(len(flows), 2)
        self.assertEqual([flows[0]["id"], flows[1]["id"]], [0, 1])
        self.assertEqual(flows[0]["cat"], f"cpu_to_{self.device_name}")

    def test_export_skips_none_trace_name(self):
        """Events whose trace_name is None are omitted from the exported JSON."""
        e_visible = _make_event(1, "visible_op", 10, 25, trace_name="visible_op")
        e_hidden = _make_event(2, "memory_op", 15, 15, trace_name=None)
        _, data = self._export_and_load(EventList([e_visible, e_hidden]), "skip.json")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "visible_op")

    def test_export_empty(self):
        """Exporting an empty EventList produces an empty JSON array."""
        _, data = self._export_and_load(EventList(), "empty.json")
        self.assertEqual(data, [])

    def test_invalid_dir_raises(self):
        """Exporting to a path in a non-existent directory raises OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(OSError):
                EventList().export_chrome_trace(os.path.join(tmpdir, "no_such_dir", "trace.json"))

    def test_missing_arg_raises(self):
        """Calling export_chrome_trace() without path raises TypeError."""
        with self.assertRaises(TypeError):
            EventList().export_chrome_trace()


if __name__ == "__main__":
    run_tests()
