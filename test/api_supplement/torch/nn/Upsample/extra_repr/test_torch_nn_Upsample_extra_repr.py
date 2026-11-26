# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Upsample.extra_repr 接口功能正确性
API 名称：torch.nn.Upsample.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；size 路径与 scale_factor 路径两种非空状态精确断言 | 已覆盖（test_extra_repr_size_path/test_extra_repr_scale_path） |
| 枚举选项 | mode 取 nearest/bilinear/bicubic/trilinear/nearest-exact 全部合法枚举 | 已覆盖（test_extra_repr_mode_enum） |
| 参数类型 | 返回值必须为 str 类型；scale_factor 支持 float 与 tuple 两种类型 | 已覆盖（test_extra_repr_returns_str/test_extra_repr_scale_tuple） |
| 传参与不传参 | size/scale_factor 互斥二选一；mode 默认 nearest；均缺省时前向报错 | 已覆盖（scale_path/size_path/forward_without_target） |
| 等价类/边界值 | size 与 scale_factor 同给时 scale_factor 优先；align_corners 不输出 | 已覆盖（scale_preferred/omits_align_corners） |
| 正常传参场景 | repr(module) 以 "Upsample(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | size 与 scale_factor 均缺省时 forward 触发 ValueError | 已覆盖（test_forward_without_target_raises） |
| 混合设备类型 | 设备侧构造的 Upsample 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无自身异常路径；异常场景覆盖 forward 的互斥校验
  （构造函数对 size/scale_factor 均缺省不做校验，构造可成功）。
- 版本行为说明：2.15 的 Upsample.extra_repr 只输出 scale_factor（或 size）与 mode 两个字段，
  align_corners 与 recompute_scale_factor 不参与表示串，精确断言按此契约编写。
- 本文件同时看护别名路径 torch.nn.modules.upsampling.Upsample.extra_repr：
  torch.nn.Upsample 与 torch.nn.modules.upsampling.Upsample 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
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
class TestUpsampleExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Upsample.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Upsample and torch.nn.modules.upsampling.Upsample are the
        same class."""
        self.assertIs(torch.nn.Upsample, torch.nn.modules.upsampling.Upsample)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for the size path and the scale path."""
        self.assertIsInstance(torch.nn.Upsample(size=(4, 5)).extra_repr(), str)
        self.assertIsInstance(
            torch.nn.Upsample(scale_factor=2.0).extra_repr(), str
        )

    def test_extra_repr_size_path(self):
        """size construction is reported with size and mode fields."""
        m = torch.nn.Upsample(size=(4, 5))
        self.assertEqual(m.extra_repr(), "size=(4, 5), mode='nearest'")
        m_int = torch.nn.Upsample(size=4, mode="bilinear")
        self.assertEqual(m_int.extra_repr(), "size=4, mode='bilinear'")

    def test_extra_repr_scale_path(self):
        """scale_factor construction is reported with scale_factor and mode."""
        m = torch.nn.Upsample(scale_factor=2.0)
        self.assertEqual(m.extra_repr(), "scale_factor=2.0, mode='nearest'")
        m_ali = torch.nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )
        self.assertEqual(m_ali.extra_repr(), "scale_factor=2.0, mode='bilinear'")

    def test_extra_repr_scale_tuple(self):
        """Tuple scale_factor is reported via repr of the stored tuple."""
        m = torch.nn.Upsample(scale_factor=(2.0, 3.0), mode="bilinear")
        self.assertEqual(m.extra_repr(), "scale_factor=(2.0, 3.0), mode='bilinear'")

    def test_extra_repr_mode_enum(self):
        """All legal mode values are reported with their repr inside quotes."""
        expected_modes = {
            "nearest": "mode='nearest'",
            "nearest-exact": "mode='nearest-exact'",
            "bilinear": "mode='bilinear'",
            "bicubic": "mode='bicubic'",
            "trilinear": "mode='trilinear'",
        }
        for mode, expected_field in expected_modes.items():
            m = torch.nn.Upsample(size=(2, 2), mode=mode)
            self.assertEqual(m.extra_repr(), f"size=(2, 2), {expected_field}")

    def test_extra_repr_scale_preferred_over_size(self):
        """When both size and scale_factor are given, scale_factor is reported."""
        m = torch.nn.Upsample(size=(4, 5), scale_factor=2.0)
        self.assertEqual(m.extra_repr(), "scale_factor=2.0, mode='nearest'")

    def test_extra_repr_omits_align_corners(self):
        """align_corners and recompute_scale_factor are not part of the repr."""
        m = torch.nn.Upsample(
            scale_factor=2.0,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=True,
        )
        text = m.extra_repr()
        self.assertNotIn("align_corners", text)
        self.assertNotIn("recompute", text)
        self.assertEqual(m.extra_repr(), "scale_factor=2.0, mode='bilinear'")

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.Upsample(size=(4, 5))
        self.assertEqual(repr(m), "Upsample(size=(4, 5), mode='nearest')")

    def test_extra_repr_reflects_mutated_attributes(self):
        """extra_repr reads the current size/mode attributes at call time."""
        m = torch.nn.Upsample(size=(4, 5))
        m.size = (7, 8)
        m.mode = "bicubic"
        self.assertEqual(m.extra_repr(), "size=(7, 8), mode='bicubic'")

    def test_forward_without_target_raises(self):
        """Upsample without size or scale_factor raises ValueError on forward
        (construction itself accepts the missing target)."""
        m = torch.nn.Upsample()
        with self.assertRaises(ValueError):
            m(torch.randn(1, 1, 4, 4))

    def test_extra_repr_device_consistency(self):
        """Device-constructed Upsample has an identical representation string."""
        cpu_m = torch.nn.Upsample(scale_factor=(2.0, 3.0), mode="bilinear")
        device_m = torch.nn.Upsample(
            scale_factor=(2.0, 3.0), mode="bilinear"
        ).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(
            device_m.extra_repr(), "scale_factor=(2.0, 3.0), mode='bilinear'"
        )


if __name__ == "__main__":
    run_tests()
