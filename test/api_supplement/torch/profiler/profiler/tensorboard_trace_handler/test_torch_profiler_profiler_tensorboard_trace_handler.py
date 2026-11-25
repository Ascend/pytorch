# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profiler.tensorboard_trace_handler 接口功能正确性
API 名称：torch.profiler.profiler.tensorboard_trace_handler
API 签名：tensorboard_trace_handler(dir_name, worker_name=None, use_gzip=False, use_python_export=False)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 嵌套目录创建与导出非空 trace 文件    | 已覆盖（default_json_and_nested_dir）              |
| 枚举选项         | 布尔导出选项（gzip / python_export） | 已覆盖（custom_worker_name_and_gzip、use_python_export_true） |
| 参数类型         | 目录路径、字符串名称、布尔开关类型   | 已覆盖（default_json_and_nested_dir、custom_worker_name_and_gzip） |
| 传参与不传参     | 可选参数默认与显式传入、缺必选参数   | 已覆盖（default_json_and_nested_dir、missing_dir_name_raises） |
| 等价类/边界值    | 多级不存在目录自动创建、gzip 压缩    | 已覆盖（default_json_and_nested_dir、custom_worker_name_and_gzip） |
| 正常传参场景     | JSON / gzip / Python 导出均正常落盘  | 已覆盖（default_json_and_nested_dir、use_python_export_true 等） |
| 异常传参场景     | 缺少 dir_name 触发 TypeError         | 已覆盖（missing_dir_name_raises）                  |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：接口仅接收目录路径与导出配置，不涉及多 Tensor 混合设备输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import gzip
import json
import os
import tempfile
import unittest

import torch
from torch.profiler import ProfilerActivity
from torch.profiler.profiler import tensorboard_trace_handler

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
class TestProfilerTensorboardTraceHandler(TestCase):
    """Test cases for torch.profiler.profiler.tensorboard_trace_handler."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_default_json_and_nested_dir(self):
        """Default handler creates nested directories and writes valid JSON trace output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "subdir")
            handler = tensorboard_trace_handler(nested_dir)
            self.assertTrue(callable(handler))
            x = torch.ones((4, 4), device=self.device)
            with torch.profiler.profile(activities=[ProfilerActivity.CPU], on_trace_ready=handler):
                _ = x + x
            files = os.listdir(nested_dir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith(".pt.trace.json"))
            with open(os.path.join(nested_dir, files[0]), "r", encoding="utf-8") as fp:
                data = json.load(fp)
            self.assertIn("traceEvents", data)
            self.assertGreater(len(data["traceEvents"]), 0)

    def test_custom_worker_name_and_gzip(self):
        """Custom worker_name and use_gzip=True produce a valid gzipped trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = tensorboard_trace_handler(tmpdir, worker_name="worker0", use_gzip=True)
            x = torch.ones((4, 4), device=self.device)
            with torch.profiler.profile(activities=[ProfilerActivity.CPU], on_trace_ready=handler):
                _ = x * 2
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("worker0."))
            self.assertTrue(files[0].endswith(".pt.trace.json.gz"))
            with gzip.open(os.path.join(tmpdir, files[0]), "rt", encoding="utf-8") as fp:
                data = json.load(fp)
            self.assertIn("traceEvents", data)
            self.assertGreater(len(data["traceEvents"]), 0)

    def test_use_python_export_true(self):
        """Setting use_python_export=True exports a valid JSON trace via Python exporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = tensorboard_trace_handler(
                tmpdir, worker_name="pyexp", use_python_export=True
            )
            with torch.profiler.profile(activities=[ProfilerActivity.CPU], on_trace_ready=handler):
                _ = torch.ones((4, 4), device=self.device) + 1
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("pyexp.") and files[0].endswith(".pt.trace.json"))
            with open(os.path.join(tmpdir, files[0]), "r", encoding="utf-8") as fp:
                data = json.load(fp)
            self.assertIn("traceEvents", data)
            self.assertGreater(len(data["traceEvents"]), 0)

    def test_missing_dir_name_raises(self):
        """Calling tensorboard_trace_handler without dir_name raises TypeError."""
        with self.assertRaises(TypeError):
            tensorboard_trace_handler()


if __name__ == "__main__":
    run_tests()
