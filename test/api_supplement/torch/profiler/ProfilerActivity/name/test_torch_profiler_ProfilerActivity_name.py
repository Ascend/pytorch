# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.ProfilerActivity.name 属性功能正确性
API 名称：torch.profiler.ProfilerActivity.name
API 签名：ProfilerActivity.name -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 所有标准成员 .name 均为非空字符串 | 已覆盖（name_type_is_str） |
| 枚举选项 | CPU/CUDA/XPU/MTIA/HPU/PrivateUse1 成员名称 | 已覆盖（standard_member_names、distinct_names） |
| 参数类型 | .name 返回类型 str 及与 str(member) 一致性 | 已覆盖（name_type_is_str、name_consistent_with_str_and_roundtrip） |
| 传参与不传参 | 只读属性访问，无调用参数 | 未覆盖，.name 为无参只读属性 |
| 等价类/边界值 | 名称唯一性及按值重建后名称一致性 | 已覆盖（distinct_names、name_consistent_with_str_and_roundtrip） |
| 正常传参场景 | 标准枚举成员 .name 读取与往返校验 | 已覆盖（standard_member_names、name_consistent_with_str_and_roundtrip） |
| 异常传参场景 | 对只读属性赋值或删除触发 AttributeError | 已覆盖（name_is_readonly） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该属性为纯枚举元信息，不涉及 Tensor 与设备迁移 |

未覆盖项及原因：
- 传参与不传参：ProfilerActivity.name 为无参只读属性，不存在参数省略/传入分支。
- 混合设备类型：该属性仅返回枚举成员名称字符串，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
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
class TestProfilerActivityName(TestCase):
    """Test cases for torch.profiler.ProfilerActivity.name."""

    EXPECTED_NAMES = ("CPU", "CUDA", "XPU", "MTIA", "HPU", "PrivateUse1")

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_standard_member_names(self):
        """Each standard ProfilerActivity member exposes its exact expected .name string."""
        self.assertEqual(ProfilerActivity.CPU.name, "CPU")
        self.assertEqual(ProfilerActivity.CUDA.name, "CUDA")
        self.assertEqual(ProfilerActivity.XPU.name, "XPU")
        self.assertEqual(ProfilerActivity.MTIA.name, "MTIA")
        self.assertEqual(ProfilerActivity.HPU.name, "HPU")
        self.assertIn(ProfilerActivity.PrivateUse1.name, ("PrivateUse1", "NPU"))

    def test_name_type_is_str(self):
        """The .name property returns a non-empty str for every standard member."""
        for expected_name in self.EXPECTED_NAMES:
            member = getattr(ProfilerActivity, expected_name)
            self.assertIsInstance(member.name, str)
            self.assertGreater(len(member.name), 0)

    def test_distinct_names(self):
        """All standard ProfilerActivity members have distinct .name values."""
        names = [getattr(ProfilerActivity, n).name for n in self.EXPECTED_NAMES]
        self.assertEqual(len(set(names)), len(self.EXPECTED_NAMES))

    def test_name_consistent_with_str_and_roundtrip(self):
        """Reconstructed members preserve .name and str(member) includes member.name."""
        for expected_name in self.EXPECTED_NAMES:
            member = getattr(ProfilerActivity, expected_name)
            reconstructed = ProfilerActivity(member.value)
            self.assertEqual(reconstructed.name, member.name)
            self.assertIn(member.name, str(member))

    def test_name_is_readonly(self):
        """Assigning to or deleting ProfilerActivity.name raises AttributeError."""
        with self.assertRaises(AttributeError):
            ProfilerActivity.CPU.name = "X"
        self.assertEqual(ProfilerActivity.CPU.name, "CPU")

        with self.assertRaises(AttributeError):
            del ProfilerActivity.CPU.name
        self.assertEqual(ProfilerActivity.CPU.name, "CPU")


if __name__ == "__main__":
    run_tests()
