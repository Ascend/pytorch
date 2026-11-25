# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.add_metadata 接口功能正确性
API 名称：torch.profiler.profile.add_metadata
API 签名：profile.add_metadata(self, key: str, value: str) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空字符串值与非空字符串值 | 已覆盖（empty_and_quoted_values、basic_add_metadata） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 普通字符串与含引号特殊字符字符串 | 已覆盖（basic_add_metadata、empty_and_quoted_values） |
| 传参与不传参 | 必填参数完整传入与缺失/多余参数 | 已覆盖（basic_add_metadata、missing_args_raises、extra_arg_raises） |
| 等价类/边界值 | 多键写入、同名键覆盖及空值/引号值 | 已覆盖（multiple_keys_and_overwrite、empty_and_quoted_values） |
| 正常传参场景 | 写入元数据并在导出 trace 中校验结果 | 已覆盖（basic_add_metadata、multiple_keys_and_overwrite） |
| 异常传参场景 | 缺参或多余参数触发 TypeError | 已覆盖（missing_args_raises、extra_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口仅接收字符串键值，不涉及多 Tensor 混合设备输入 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口仅向 profiler trace 附加字符串元数据，不接收 Tensor 输入。

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

# Probe device only when RUN_FLAG is enabled.
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
class TestTorchProfilerProfileAddMetadata(TestCase):
    """Test cases for torch.profiler.profile.add_metadata."""

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

    def test_basic_add_metadata(self):
        """add_metadata attaches a key/value pair visible in the exported trace."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            self.assertIsNone(prof.add_metadata("custom_tag", "run_alpha"))
        data = self._export_and_load(prof)
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("custom_tag"), "run_alpha")

    def test_multiple_keys_and_overwrite(self):
        """Multiple keys are preserved and repeated keys overwrite prior values."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            prof.add_metadata("meta_k1", "val_one")
            prof.add_metadata("meta_k2", "val_two")
            prof.add_metadata("version", "v1")
            prof.add_metadata("version", "v2")
        data = self._export_and_load(prof)
        self.assertEqual(data.get("meta_k1"), "val_one")
        self.assertEqual(data.get("meta_k2"), "val_two")
        self.assertEqual(data.get("version"), "v2")

    def test_empty_and_quoted_values(self):
        """Empty strings and quoted strings are serialized safely into trace metadata."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            prof.add_metadata("empty_key", "")
            prof.add_metadata("quoted_key", 'say "hello"')
        data = self._export_and_load(prof)
        self.assertEqual(data.get("empty_key"), "")
        self.assertIn("hello", data.get("quoted_key", ""))

    def test_missing_args_raises(self):
        """Calling add_metadata with fewer than two arguments raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.add_metadata()
        with self.assertRaises(TypeError):
            prof.add_metadata("only_key")

    def test_extra_arg_raises(self):
        """Calling add_metadata with extra arguments raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.add_metadata("k", "v", "extra")


if __name__ == "__main__":
    run_tests()
