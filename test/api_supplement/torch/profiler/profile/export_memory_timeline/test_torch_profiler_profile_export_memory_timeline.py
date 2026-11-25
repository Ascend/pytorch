# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.export_memory_timeline 接口功能正确性
API 名称：torch.profiler.profile.export_memory_timeline
API 签名：profile.export_memory_timeline(self, path: str, device: str | None = None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 未启动/未开启内存采集与正常采集后导出 | 已覆盖（error_cases_raise、export_json_and_gz_cpu） |
| 枚举选项 | 输出文件格式（.json / .json.gz） | 已覆盖（export_json_and_gz_cpu） |
| 参数类型 | 路径字符串与 device 字符串/None | 已覆盖（export_json_and_gz_cpu、export_json_default_device） |
| 传参与不传参 | device 显式传入、省略及缺少 path 参数 | 已覆盖（export_json_and_gz_cpu、export_json_default_device、error_cases_raise） |
| 等价类/边界值 | JSON 与 GZ 导出、默认设备与显式 cpu 设备 | 已覆盖（export_json_and_gz_cpu、export_json_default_device） |
| 正常传参场景 | 导出时间线并校验两段等长数据结构 | 已覆盖（export_json_and_gz_cpu、export_json_default_device） |
| 异常传参场景 | 未开启 profile_memory、未启动及缺参异常 | 已覆盖（error_cases_raise） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为内存时间线导出方法，不接受 Tensor 输入 |

未覆盖项及原因：
- 混合设备类型：该接口接收输出路径和设备名字符串，不涉及多 Tensor 混合设备输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
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
import warnings

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
class TestTorchProfilerProfileExportMemoryTimeline(TestCase):
    """Test cases for torch.profiler.profile.export_memory_timeline."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_memory_profile(self, activities=None):
        acts = activities or [ProfilerActivity.CPU]
        with torch.profiler.profile(activities=acts, record_shapes=True, profile_memory=True, with_stack=True) as prof:
            x = torch.randn(32, 32, device="cpu")
            _ = (x @ x).sum()
            if self.device_name != "cpu":
                xd = torch.randn(32, 32, device=self.device)
                _ = (xd @ xd).sum()
        return prof

    def _export_and_load(self, prof, path, device=None, is_gz=False):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            kwargs = {"device": device} if device is not None else {}
            prof.export_memory_timeline(path, **kwargs)
        open_fn = gzip.open if is_gz else open
        with open_fn(path, "rt" if is_gz else "r", encoding="utf-8") as fp:
            return json.load(fp)

    def test_export_json_and_gz_cpu(self):
        """export_memory_timeline exports valid .json and .json.gz payloads for cpu."""
        prof = self._run_memory_profile()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "mem_timeline.json")
            data = self._export_and_load(prof, json_path, device="cpu")
            self.assertTrue(os.path.isfile(json_path))
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 2)
            self.assertEqual(len(data[0]), len(data[1]))
            gz_path = os.path.join(tmpdir, "mem_timeline.json.gz")
            gz_data = self._export_and_load(prof, gz_path, device="cpu", is_gz=True)
            self.assertIsInstance(gz_data, list)
            self.assertEqual(len(gz_data), 2)

    def test_export_json_default_device(self):
        """Omitting device exports a valid two-element memory timeline list."""
        prof = self._run_memory_profile(activities=list(torch.profiler.profiler.supported_activities()))
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export_and_load(prof, os.path.join(tmpdir, "mem_default.json"))
            self.assertEqual((type(data), len(data)), (list, 2))

    def test_error_cases_raise(self):
        """Missing profile_memory, unstarted profiler, and missing path raise errors."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            x = torch.randn(16, 16, device=self.device)
            _ = (x @ x).sum()
        unstarted = torch.profiler.profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                                           profile_memory=True, with_stack=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                self._export_and_load(prof, os.path.join(tmpdir, "fail.json"), device="cpu")
            with self.assertRaises(AssertionError):
                self._export_and_load(unstarted, os.path.join(tmpdir, "unstarted.json"), device="cpu")
        with self.assertRaises(TypeError):
            unstarted.export_memory_timeline()


if __name__ == "__main__":
    run_tests()
