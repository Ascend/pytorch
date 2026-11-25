# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.preset_metadata_json 接口功能正确性
API 名称：torch.profiler.profile.preset_metadata_json
API 签名：profile.preset_metadata_json(self, key: str, value: str) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空 JSON 对象与非空 JSON 字符串 | 已覆盖（empty_json_value、basic_preset_and_export） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | JSON 对象、字符串、整数与数组 | 已覆盖（basic_preset_and_export、multiple_keys） |
| 传参与不传参 | 必填参数完整传入与缺参异常 | 已覆盖（basic_preset_and_export、missing_args_raises） |
| 等价类/边界值 | 同名键覆盖与空 JSON 对象边界 | 已覆盖（overwrite_existing_key、empty_json_value） |
| 正常传参场景 | 预置元数据并在启动/导出 trace 后校验 | 已覆盖（basic_preset_and_export、multiple_keys、overwrite_existing_key） |
| 异常传参场景 | 缺少必填参数触发 TypeError | 已覆盖（missing_args_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口仅预置 JSON 元数据字符串，不涉及设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口仅向 profiler 预置 JSON 字符串键值，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
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
class TestTorchProfilerProfilePresetMetadataJson(TestCase):
    """Test cases for torch.profiler.profile.preset_metadata_json."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_and_export(self, prof):
        with prof:
            x = torch.randn(16, 16, device=self.device)
            _ = (x @ x).sum()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(trace_path)
            with open(trace_path, "r", encoding="utf-8") as fp:
                return json.load(fp)

    def test_basic_preset_and_export(self):
        """preset_metadata_json stores JSON metadata and exports it into the trace."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        ret = prof.preset_metadata_json("preset_cfg", '{"mode": "eval", "batch": 8}')
        self.assertIsNone(ret)
        self.assertIn("preset_cfg", prof.preset_metadata)
        self.assertEqual(prof.preset_metadata["preset_cfg"], '{"mode": "eval", "batch": 8}')
        data = self._run_and_export(prof)
        self.assertEqual(data.get("preset_cfg"), {"mode": "eval", "batch": 8})

    def test_multiple_keys(self):
        """Multiple preset JSON keys of different JSON types are exported accurately."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        prof.preset_metadata_json("p_str", '"worker_0"')
        prof.preset_metadata_json("p_num", "99")
        prof.preset_metadata_json("p_list", "[1, 2, 3]")
        self.assertEqual(len(prof.preset_metadata), 3)
        data = self._run_and_export(prof)
        self.assertEqual(data.get("p_str"), "worker_0")
        self.assertEqual(data.get("p_num"), 99)
        self.assertEqual(data.get("p_list"), [1, 2, 3])

    def test_overwrite_existing_key(self):
        """Calling preset_metadata_json with an existing key overwrites its value."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        prof.preset_metadata_json("rev", '"v1"')
        prof.preset_metadata_json("rev", '"v2"')
        self.assertEqual(prof.preset_metadata["rev"], '"v2"')
        data = self._run_and_export(prof)
        self.assertEqual(data.get("rev"), "v2")

    def test_empty_json_value(self):
        """An empty JSON object string is preserved in preset_metadata and exported trace."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        prof.preset_metadata_json("empty_meta", "{}")
        self.assertEqual(prof.preset_metadata["empty_meta"], "{}")
        data = self._run_and_export(prof)
        self.assertEqual(data.get("empty_meta"), {})

    def test_missing_args_raises(self):
        """Calling preset_metadata_json with fewer than two arguments raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.preset_metadata_json()
        with self.assertRaises(TypeError):
            prof.preset_metadata_json("only_key")


if __name__ == "__main__":
    run_tests()
