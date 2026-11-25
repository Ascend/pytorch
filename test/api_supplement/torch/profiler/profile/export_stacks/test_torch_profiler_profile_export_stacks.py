# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.export_stacks 接口功能正确性
API 名称：torch.profiler.profile.export_stacks
API 签名：profile.export_stacks(self, path: str, metric: str = "self_cpu_time_total") -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 未开启 with_stack 与正常采集非空调用栈 | 已覆盖（without_with_stack_raises、export_default_metric） |
| 枚举选项 | metric 合法值与非法值 | 已覆盖（export_explicit_cpu_metric、invalid_metric_and_missing_path_raises） |
| 参数类型 | 路径字符串与 metric 字符串 | 已覆盖（export_default_metric、export_explicit_cpu_metric） |
| 传参与不传参 | metric 省略/传入及缺 path | 已覆盖（export_default_metric、export_explicit_cpu_metric） |
| 等价类/边界值 | 默认/显式 metric 及无栈状态 | 已覆盖（export_default_metric、without_with_stack_raises） |
| 正常传参场景 | 导出火焰图折叠栈文件并校验行格式与正耗时 | 已覆盖（export_default_metric、export_explicit_cpu_metric） |
| 异常传参场景 | 未开启 with_stack、非法 metric 及缺参异常 | 已覆盖（without_with_stack_raises、invalid_metric_and_missing_path_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为调用栈文本导出方法，不接受 Tensor 输入 |

未覆盖项及原因：
- 混合设备类型：该接口接收文件路径与 metric 名称，不涉及多 Tensor 混合设备输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
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
class TestTorchProfilerProfileExportStacks(TestCase):
    """Test cases for torch.profiler.profile.export_stacks."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_workload(self):
        x = torch.randn(64, 64, device=self.device)
        out = x
        for _ in range(5):
            out = out @ x
        return out.sum()

    def _assert_valid_stacks_file(self, path):
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as fp:
            lines = [line.strip() for line in fp if line.strip()]
        self.assertGreater(len(lines), 0)
        for line in lines:
            parts = line.rsplit(" ", 1)
            self.assertEqual(len(parts), 2)
            self.assertGreater(int(parts[1]), 0)

    def test_export_default_metric(self):
        """export_stacks writes non-empty folded stack lines with the default metric."""
        exp_cfg = torch._C._profiler._ExperimentalConfig(verbose=True)
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], with_stack=True, experimental_config=exp_cfg
        ) as prof:
            self._run_workload()
        with tempfile.TemporaryDirectory() as tmpdir:
            stacks_path = os.path.join(tmpdir, "stacks_default.txt")
            prof.export_stacks(stacks_path)
            self._assert_valid_stacks_file(stacks_path)

    def test_export_explicit_cpu_metric(self):
        """export_stacks writes valid stack lines when metric='self_cpu_time_total'."""
        exp_cfg = torch._C._profiler._ExperimentalConfig(verbose=True)
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], with_stack=True, experimental_config=exp_cfg
        ) as prof:
            self._run_workload()
        with tempfile.TemporaryDirectory() as tmpdir:
            stacks_path = os.path.join(tmpdir, "stacks_cpu.txt")
            prof.export_stacks(stacks_path, metric="self_cpu_time_total")
            self._assert_valid_stacks_file(stacks_path)

    def test_without_with_stack_raises(self):
        """Calling export_stacks when with_stack=False raises AssertionError."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], with_stack=False) as prof:
            self._run_workload()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(AssertionError):
                prof.export_stacks(os.path.join(tmpdir, "no_stack.txt"))

    def test_invalid_metric_and_missing_path_raises(self):
        """Invalid metric raises ValueError and missing path raises TypeError."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            self._run_workload()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                prof.export_stacks(os.path.join(tmpdir, "bad.txt"), metric="invalid_metric_name")
        with self.assertRaises(TypeError):
            prof.export_stacks()


if __name__ == "__main__":
    run_tests()
