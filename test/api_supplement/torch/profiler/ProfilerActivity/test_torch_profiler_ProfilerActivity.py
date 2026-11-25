# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.ProfilerActivity 枚举接口功能正确性
API 名称：torch.profiler.ProfilerActivity
API 签名：class torch.profiler.ProfilerActivity

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 无参构造及有效枚举成员存在性 | 已覆盖（invalid_constructor_raises、standard_members_exist） |
| 枚举选项 | CPU/CUDA/XPU/MTIA/HPU/PrivateUse1 标准成员 | 已覆盖（standard_members_exist、distinct_values） |
| 参数类型 | 整数枚举值构造与非法字符串构造 | 已覆盖（value_roundtrip、invalid_constructor_raises） |
| 传参与不传参 | 传入合法/非法值与无参调用构造 | 已覆盖（value_roundtrip、invalid_constructor_raises） |
| 等价类/边界值 | 成员值唯一性、相等/不等、集合去重与字典键 | 已覆盖（distinct_values、equality_and_inequality、hashability_and_containers） |
| 正常传参场景 | 成员访问、比较、容器使用及值往返重建 | 已覆盖（standard_members_exist、equality_and_inequality、value_roundtrip） |
| 异常传参场景 | 非法参数或无参构造触发 TypeError/ValueError | 已覆盖（invalid_constructor_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，ProfilerActivity 为纯枚举定义，不涉及 Tensor 输入 |

未覆盖项及原因：
- 混合设备类型：ProfilerActivity 仅表示性能分析活动枚举，不接受 Tensor 输入。

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
class TestProfilerActivity(TestCase):
    """Test cases for torch.profiler.ProfilerActivity."""

    STANDARD_MEMBERS = ("CPU", "CUDA", "XPU", "MTIA", "HPU", "PrivateUse1")

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_standard_members_exist(self):
        """Standard ProfilerActivity members exist with expected types, names, and int values."""
        for member_name in self.STANDARD_MEMBERS:
            self.assertTrue(hasattr(ProfilerActivity, member_name))
            member = getattr(ProfilerActivity, member_name)
            self.assertIsInstance(member, ProfilerActivity)
            if member_name == "PrivateUse1":
                self.assertIn(member.name, ("PrivateUse1", "NPU"))
            else:
                self.assertEqual(member.name, member_name)
            self.assertIsInstance(member.value, int)
        if hasattr(ProfilerActivity, "NPU"):
            self.assertIsInstance(ProfilerActivity.NPU, ProfilerActivity)

    def test_distinct_values(self):
        """All standard ProfilerActivity members have distinct integer values."""
        members = [getattr(ProfilerActivity, name) for name in self.STANDARD_MEMBERS]
        values = [m.value for m in members]
        self.assertEqual(len(set(values)), len(self.STANDARD_MEMBERS))

    def test_equality_and_inequality(self):
        """Identical members compare equal and distinct members compare unequal."""
        self.assertEqual(ProfilerActivity.CPU, ProfilerActivity.CPU)
        self.assertEqual(ProfilerActivity.CUDA, ProfilerActivity.CUDA)
        self.assertEqual(ProfilerActivity.PrivateUse1, ProfilerActivity.PrivateUse1)
        self.assertNotEqual(ProfilerActivity.CPU, ProfilerActivity.CUDA)
        self.assertNotEqual(ProfilerActivity.CPU, ProfilerActivity.PrivateUse1)
        self.assertNotEqual(ProfilerActivity.CUDA, ProfilerActivity.PrivateUse1)

    def test_hashability_and_containers(self):
        """ProfilerActivity members are hashable and work in sets and dicts."""
        activities = {ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.PrivateUse1, ProfilerActivity.CPU}
        self.assertEqual(len(activities), 3)
        self.assertIn(ProfilerActivity.CPU, activities)
        self.assertIn(ProfilerActivity.CUDA, activities)
        self.assertIn(ProfilerActivity.PrivateUse1, activities)
        labels = {
            ProfilerActivity.CPU: "cpu_activity",
            ProfilerActivity.CUDA: "cuda_activity",
        }
        self.assertEqual(labels[ProfilerActivity.CPU], "cpu_activity")
        self.assertEqual(labels[ProfilerActivity.CUDA], "cuda_activity")

    def test_value_roundtrip(self):
        """Constructing ProfilerActivity from a member's value reproduces that member."""
        for member_name in self.STANDARD_MEMBERS:
            member = getattr(ProfilerActivity, member_name)
            reconstructed = ProfilerActivity(member.value)
            self.assertEqual(reconstructed, member)
            self.assertEqual(reconstructed.name, member.name)

    def test_invalid_constructor_raises(self):
        """Passing invalid or missing arguments to ProfilerActivity raises TypeError or ValueError."""
        with self.assertRaises((TypeError, ValueError)):
            ProfilerActivity("invalid_activity")
        with self.assertRaises((TypeError, ValueError)):
            ProfilerActivity()


if __name__ == "__main__":
    run_tests()
