# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.add_metadata_json 接口功能正确性
API 名称：torch.profiler.profile.add_metadata_json
API 签名：profile.add_metadata_json(self, key: str, value: str) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空 JSON 对象/数组与非空 JSON 值 | 已覆盖（empty_and_overwrite、json_object_and_list） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | JSON 对象、数组、数值、字符串及布尔值 | 已覆盖（json_object_and_list、json_primitives） |
| 传参与不传参 | 必填参数完整传入与缺失/多余参数 | 已覆盖（json_object_and_list、missing_args_raises、extra_arg_raises） |
| 等价类/边界值 | 空对象/数组及同名键覆写 | 已覆盖（empty_and_overwrite） |
| 正常传参场景 | 写入各类 JSON 元数据并在导出 trace 中校验 | 已覆盖（json_object_and_list、json_primitives、empty_and_overwrite） |
| 异常传参场景 | 缺参或多余参数触发 TypeError | 已覆盖（missing_args_raises、extra_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口仅接收 JSON 字符串键值，不涉及多 Tensor 设备 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口仅向 profiler trace 附加 JSON 字符串元数据，不接收 Tensor 输入。

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
class TestTorchProfilerProfileAddMetadataJson(TestCase):
    """Test cases for torch.profiler.profile.add_metadata_json."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_workload(self):
        x = torch.randn(16, 16, device=self.device)
        return (x @ x).sum()

    def _export_and_load(self, prof):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(trace_path)
            with open(trace_path, "r", encoding="utf-8") as fp:
                return json.load(fp)

    def test_json_object_and_list(self):
        """JSON object and list strings are embedded and parsed in the exported trace."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            self.assertIsNone(prof.add_metadata_json("cfg_obj", '{"a": 1, "b": "x"}'))
            self.assertIsNone(prof.add_metadata_json("cfg_list", "[1, 2, 3]"))
        data = self._export_and_load(prof)
        self.assertEqual(data.get("cfg_obj"), {"a": 1, "b": "x"})
        self.assertEqual(data.get("cfg_list"), [1, 2, 3])

    def test_json_primitives(self):
        """JSON number, string, and boolean primitives round-trip through trace export."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            prof.add_metadata_json("num_key", "42")
            prof.add_metadata_json("str_key", '"hello"')
            prof.add_metadata_json("bool_key", "true")
        data = self._export_and_load(prof)
        self.assertEqual(data.get("num_key"), 42)
        self.assertEqual(data.get("str_key"), "hello")
        self.assertIs(data.get("bool_key"), True)

    def test_empty_and_overwrite(self):
        """Empty JSON containers are preserved and repeated keys overwrite earlier values."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            prof.add_metadata_json("empty_obj", "{}")
            prof.add_metadata_json("empty_arr", "[]")
            prof.add_metadata_json("dup_key", "1")
            prof.add_metadata_json("dup_key", '{"updated": true}')
        data = self._export_and_load(prof)
        self.assertEqual(data.get("empty_obj"), {})
        self.assertEqual(data.get("empty_arr"), [])
        self.assertEqual(data.get("dup_key"), {"updated": True})

    def test_missing_args_raises(self):
        """Calling add_metadata_json with missing arguments raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.add_metadata_json()
        with self.assertRaises(TypeError):
            prof.add_metadata_json("only_key")

    def test_extra_arg_raises(self):
        """Calling add_metadata_json with extra arguments raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.add_metadata_json("k", "1", "extra")


if __name__ == "__main__":
    run_tests()
