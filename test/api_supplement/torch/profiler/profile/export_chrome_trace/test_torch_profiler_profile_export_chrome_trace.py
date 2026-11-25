# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.export_chrome_trace 接口功能正确性
API 名称：torch.profiler.profile.export_chrome_trace
API 签名：profile.export_chrome_trace(self, path: str, ...) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 未启动空状态与已采集非空 trace 导出 | 已覆盖（error_cases_raise、export_json_and_gzip） |
| 枚举选项 | .json/.json.gz 后缀及 graph_lanes 选项 | 已覆盖（export_json_and_gzip、use_python_export_and_graph_lanes） |
| 参数类型 | 字符串路径及布尔标志 use_python_export | 已覆盖（export_json_and_gzip、use_python_export_and_graph_lanes） |
| 传参与不传参 | 仅传 path 与传入可选关键字参数及缺参 | 已覆盖（export_json_and_gzip、use_python_export_and_graph_lanes、error_cases_raise） |
| 等价类/边界值 | 普通 JSON、gzip 压缩及 C++/Python 导出路径 | 已覆盖（export_json_and_gzip、use_python_export_and_graph_lanes） |
| 正常传参场景 | 正常导出 JSON/GZ 文件并校验 traceEvents | 已覆盖（export_json_and_gzip、use_python_export_and_graph_lanes） |
| 异常传参场景 | 未启动断言失败、无效路径异常及缺参 | 已覆盖（error_cases_raise） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为 trace 文件导出方法，不接受 Tensor 输入 |

未覆盖项及原因：
- 混合设备类型：该接口接收文件路径与导出选项，不涉及多 Tensor 混合设备输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import gzip
import inspect
import json
import os
import tempfile
import unittest

import torch
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
class TestTorchProfilerProfileExportChromeTrace(TestCase):
    """Test cases for torch.profiler.profile.export_chrome_trace."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_profiled_workload(self):
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            x = torch.randn(16, 16, device=self.device)
            _ = (x @ x).sum()
        return prof

    def _assert_valid_trace_data(self, data):
        if isinstance(data, dict):
            self.assertIn("traceEvents", data)
        events = data["traceEvents"] if isinstance(data, dict) else data
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0)

    def test_export_json_and_gzip(self):
        """export_chrome_trace writes valid trace payloads for both .json and .json.gz paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            self._run_profiled_workload().export_chrome_trace(trace_path)
            self.assertGreater(os.path.getsize(trace_path), 0)
            with open(trace_path, "r", encoding="utf-8") as fp:
                self._assert_valid_trace_data(json.load(fp))
            gz_path = os.path.join(tmpdir, "trace.json.gz")
            self._run_profiled_workload().export_chrome_trace(gz_path)
            with gzip.open(gz_path, "rt", encoding="utf-8") as fp:
                self._assert_valid_trace_data(json.load(fp))

    def test_use_python_export_and_graph_lanes(self):
        """Both C++ and Python export paths produce valid traces when supported."""
        params = inspect.signature(torch.profiler.profile.export_chrome_trace).parameters
        if "use_python_export" not in params:
            self.skipTest("use_python_export parameter is not present in this torch version")
        extra = {"graph_lanes": "none"} if "graph_lanes" in params else {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for flag, name in ((False, "cpp.json"), (True, "py.json")):
                path = os.path.join(tmpdir, name)
                self._run_profiled_workload().export_chrome_trace(path, use_python_export=flag, **extra)
                with open(path, "r", encoding="utf-8") as fp:
                    self._assert_valid_trace_data(json.load(fp))

    def test_error_cases_raise(self):
        """Unstarted profiler, invalid directory path, and missing path raise expected errors."""
        unstarted = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        prof = self._run_profiled_workload()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(AssertionError):
                unstarted.export_chrome_trace(os.path.join(tmpdir, "never.json"))
            if "use_python_export" in inspect.signature(torch.profiler.profile.export_chrome_trace).parameters:
                with self.assertRaises((RuntimeError, OSError)):
                    prof.export_chrome_trace(os.path.join(tmpdir, "missing_subdir", "trace.json"),
                                             use_python_export=True)
        with self.assertRaises(TypeError):
            prof.export_chrome_trace()


if __name__ == "__main__":
    run_tests()
