# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Unfold.extra_repr 接口功能正确性
API 名称：torch.nn.Unfold.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；默认构造与显式参数构造两种状态精确断言 | 已覆盖（default_args/explicit_args 两用例） |
| 枚举选项 | kernel_size/dilation/padding/stride 的 int 与 tuple 形态 | 已覆盖（default_args/int_kernel/explicit_args） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_returns_str） |
| 传参与不传参 | kernel_size 必填；dilation/padding/stride 默认值 1/0/1 | 已覆盖（test_extra_repr_default_args/test_constructor_kernel_required） |
| 等价类/边界值 | int 入参按原样存储不归一化（2.15 行为）；字段顺序固定 | 已覆盖（test_extra_repr_int_kernel/test_extra_repr_field_order） |
| 正常传参场景 | repr(module) 以 "Unfold(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺参触发 TypeError | 已覆盖（test_constructor_kernel_required） |
| 混合设备类型 | 设备侧构造的 Unfold 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；异常场景覆盖构造缺参。
- 非法 kernel_size 等仅在 forward/functional 层报错，不属于 extra_repr 契约，未覆盖。
- 版本行为说明：2.15 的 Unfold.__init__ 将 dilation/padding/stride 按原样存储（默认 int
  1/0/1），不做 _pair 归一化，extra_repr 按存储值精确输出；字段顺序为
  kernel_size, dilation, padding, stride（不含 output_size）。
- 本文件同时看护别名路径 torch.nn.modules.fold.Unfold.extra_repr：torch.nn.Unfold 与
  torch.nn.modules.fold.Unfold 为同一类对象，test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     设备侧仅验证构造与表示行为，无浮点数值输出，不涉及浮点 Golden 对照。
"""
import os
import unittest

import torch

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
class TestUnfoldExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Unfold.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Unfold and torch.nn.modules.fold.Unfold are the same class."""
        self.assertIs(torch.nn.Unfold, torch.nn.modules.fold.Unfold)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for default and explicit constructions."""
        self.assertIsInstance(torch.nn.Unfold((2, 3)).extra_repr(), str)
        self.assertIsInstance(
            torch.nn.Unfold((2, 3), dilation=(2, 2), padding=(1, 1),
                            stride=(2, 2)).extra_repr(),
            str,
        )

    def test_extra_repr_default_args(self):
        """Default dilation/padding/stride are reported as stored ints 1/0/1."""
        m = torch.nn.Unfold((2, 3))
        self.assertEqual(
            m.extra_repr(),
            "kernel_size=(2, 3), dilation=1, padding=0, stride=1",
        )

    def test_extra_repr_int_kernel(self):
        """Int kernel_size is stored as-is and reported without pair expansion."""
        m = torch.nn.Unfold(3)
        self.assertEqual(
            m.extra_repr(), "kernel_size=3, dilation=1, padding=0, stride=1"
        )

    def test_extra_repr_explicit_args(self):
        """Explicit tuple dilation/padding/stride are reported exactly."""
        m = torch.nn.Unfold((2, 3), dilation=(2, 2), padding=(1, 1),
                            stride=(2, 2))
        self.assertEqual(
            m.extra_repr(),
            "kernel_size=(2, 3), dilation=(2, 2), padding=(1, 1), stride=(2, 2)",
        )

    def test_extra_repr_field_order(self):
        """Fields appear in the fixed order kernel_size, dilation, padding,
        stride (no output_size field)."""
        m = torch.nn.Unfold((2, 3), dilation=(2, 2), padding=(1, 1),
                            stride=(2, 2))
        text = m.extra_repr()
        self.assertNotIn("output_size", text)
        expected_order = [
            "kernel_size=(2, 3)",
            "dilation=(2, 2)",
            "padding=(1, 1)",
            "stride=(2, 2)",
        ]
        positions = [text.index(field) for field in expected_order]
        self.assertEqual(positions, sorted(positions))

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.Unfold((2, 3))
        self.assertEqual(
            repr(m),
            "Unfold(kernel_size=(2, 3), dilation=1, padding=0, stride=1)",
        )

    def test_extra_repr_reflects_mutated_attributes(self):
        """extra_repr reads the current attribute values at call time."""
        m = torch.nn.Unfold((2, 3))
        m.kernel_size = (3, 3)
        m.dilation = (2, 1)
        self.assertEqual(
            m.extra_repr(),
            "kernel_size=(3, 3), dilation=(2, 1), padding=0, stride=1",
        )

    def test_constructor_kernel_required(self):
        """Unfold requires kernel_size (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.Unfold()

    def test_extra_repr_device_consistency(self):
        """Device-constructed Unfold has an identical representation string."""
        cpu_m = torch.nn.Unfold((2, 3), dilation=(2, 2), padding=(1, 1),
                                stride=(2, 2))
        device_m = torch.nn.Unfold((2, 3), dilation=(2, 2), padding=(1, 1),
                                   stride=(2, 2)).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())


if __name__ == "__main__":
    run_tests()
