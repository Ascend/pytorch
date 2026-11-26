# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.PixelUnshuffle.extra_repr 接口功能正确性
API 名称：torch.nn.PixelUnshuffle.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对 downscale_factor=1/3/4 多个非空参数值分别精确断言 | 已覆盖（test_extra_repr 参数化 factor=1/3/4） |
| 枚举选项 | downscale_factor 取多个正整数值逐个断言字符串 | 已覆盖（test_extra_repr 参数化 factor=1/3/4） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr 精确字符串断言隐含 str 类型） |
| 传参与不传参 | extra_repr 无参；构造参数 downscale_factor 必填，缺省构造断言 TypeError | 已覆盖（test_constructor_factor_required） |
| 等价类/边界值 | downscale_factor=1 最小合法值；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr/test_extra_repr_reflects_mutated_factor） |
| 正常传参场景 | repr(module) 以 "PixelUnshuffle(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺参触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_factor_required） |
| 混合设备类型 | 设备侧构造的 PixelUnshuffle 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；异常场景覆盖构造缺参。
- downscale_factor 非正整数仅在 forward/functional 层报错，不属于 extra_repr 契约，未覆盖。
- 字段名说明：构造参数为 downscale_factor，extra_repr 输出 downscale_factor=<值>。
- 本文件同时看护别名路径 torch.nn.modules.pixelshuffle.PixelUnshuffle.extra_repr：
  torch.nn.PixelUnshuffle 与 torch.nn.modules.pixelshuffle.PixelUnshuffle 为同一类对象，
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
class TestPixelUnshuffleExtraRepr(TestCase):
    """Cases for torch.nn.PixelUnshuffle.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.PixelUnshuffle and torch.nn.modules.pixelshuffle.PixelUnshuffle
        are the same class."""
        self.assertIs(
            torch.nn.PixelUnshuffle,
            torch.nn.modules.pixelshuffle.PixelUnshuffle,
        )

    @parametrize(
        "factor, expected",
        [(1, "downscale_factor=1"), (3, "downscale_factor=3"),
         (4, "downscale_factor=4")],
    )
    def test_extra_repr(self, factor, expected):
        """extra_repr reports downscale_factor exactly for position/keyword construction."""
        self.assertEqual(torch.nn.PixelUnshuffle(factor).extra_repr(), expected)
        self.assertEqual(
            torch.nn.PixelUnshuffle(downscale_factor=factor).extra_repr(), expected
        )

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.PixelUnshuffle(3)
        self.assertEqual(repr(m), "PixelUnshuffle(downscale_factor=3)")

    def test_extra_repr_reflects_mutated_factor(self):
        """extra_repr reads the current downscale_factor attribute at call time."""
        m = torch.nn.PixelUnshuffle(3)
        m.downscale_factor = 4
        self.assertEqual(m.extra_repr(), "downscale_factor=4")

    def test_constructor_factor_required(self):
        """PixelUnshuffle requires downscale_factor (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.PixelUnshuffle()

    def test_extra_repr_device_consistency(self):
        """Device-constructed PixelUnshuffle has an identical representation."""
        cpu_m = torch.nn.PixelUnshuffle(3)
        device_m = torch.nn.PixelUnshuffle(3).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "downscale_factor=3")


instantiate_parametrized_tests(TestPixelUnshuffleExtraRepr)


if __name__ == "__main__":
    run_tests()
