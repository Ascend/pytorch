# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.ChannelShuffle.extra_repr 接口功能正确性
API 名称：torch.nn.ChannelShuffle.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对 groups=1/4/8 多个非空参数值分别精确断言 | 已覆盖（test_extra_repr 参数化 groups=1/4/8） |
| 枚举选项 | groups 取多个正整数值逐个断言字符串 | 已覆盖（test_extra_repr 参数化 groups=1/4/8） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr 精确字符串断言隐含 str 类型） |
| 传参与不传参 | extra_repr 无参；构造参数 groups 为必填，缺省构造断言 TypeError | 已覆盖（test_constructor_groups_required） |
| 等价类/边界值 | groups=1 最小合法值；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr/test_extra_repr_reflects_mutated_groups） |
| 正常传参场景 | repr(module) 以 "ChannelShuffle(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺参触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_groups_required） |
| 混合设备类型 | 设备侧构造的 ChannelShuffle 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；异常场景覆盖构造缺参。
- groups 非正整数仅在 forward/functional 层报错，不属于 extra_repr 契约，未覆盖。
- 本文件同时看护别名路径 torch.nn.modules.channelshuffle.ChannelShuffle.extra_repr：
  torch.nn.ChannelShuffle 与 torch.nn.modules.channelshuffle.ChannelShuffle 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     设备侧仅验证构造与表示行为，无浮点数值输出，不涉及浮点 Golden 对照。
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

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
class TestChannelShuffleExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.ChannelShuffle.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.ChannelShuffle and torch.nn.modules.channelshuffle.ChannelShuffle
        are the same class."""
        self.assertIs(
            torch.nn.ChannelShuffle, torch.nn.modules.channelshuffle.ChannelShuffle
        )

    @parametrize(
        "groups, expected",
        [(1, "groups=1"), (4, "groups=4"), (8, "groups=8")],
    )
    def test_extra_repr(self, groups, expected):
        """extra_repr reports groups exactly for position and keyword construction."""
        self.assertEqual(torch.nn.ChannelShuffle(groups).extra_repr(), expected)
        self.assertEqual(
            torch.nn.ChannelShuffle(groups=groups).extra_repr(), expected
        )

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.ChannelShuffle(4)
        self.assertEqual(repr(m), "ChannelShuffle(groups=4)")

    def test_extra_repr_reflects_mutated_groups(self):
        """extra_repr reads the current groups attribute at call time."""
        m = torch.nn.ChannelShuffle(4)
        m.groups = 16
        self.assertEqual(m.extra_repr(), "groups=16")

    def test_constructor_groups_required(self):
        """ChannelShuffle requires the groups argument (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.ChannelShuffle()

    def test_extra_repr_device_consistency(self):
        """Device-constructed ChannelShuffle has an identical representation."""
        cpu_m = torch.nn.ChannelShuffle(4)
        device_m = torch.nn.ChannelShuffle(4).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "groups=4")


instantiate_parametrized_tests(TestChannelShuffleExtraRepr)


if __name__ == "__main__":
    run_tests()
