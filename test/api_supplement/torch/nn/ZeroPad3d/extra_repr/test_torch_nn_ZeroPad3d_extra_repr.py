# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.ZeroPad3d.extra_repr 接口功能正确性
API 名称：torch.nn.ZeroPad3d.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；int 与 6-tuple 两种 padding 形态精确断言 | 已覆盖（test_extra_repr_int_padding/test_extra_repr_tuple_padding） |
| 枚举选项 | padding 取 int 与 6-tuple 两种合法形态 | 已覆盖（test_extra_repr_int_padding/test_extra_repr_tuple_padding） |
| 参数类型 | 返回值必须为 str 类型；padding 属性存储为归一化 6-tuple | 已覆盖（test_extra_repr_returns_str/test_padding_normalized_to_sixtuple） |
| 传参与不传参 | extra_repr 无参；构造参数 padding 必填，缺省构造断言 TypeError | 已覆盖（test_constructor_padding_required） |
| 等价类/边界值 | int 广播为 6 元组；非对称 tuple 按序输出；属性修改后同步反映 | 已覆盖（int_padding/tuple_padding/reflects_mutated_padding） |
| 正常传参场景 | repr(module) 以 "ZeroPad3d(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺参触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_padding_required） |
| 混合设备类型 | 设备侧构造的 ZeroPad3d 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；异常场景覆盖构造缺参。
- 零填充的前向数值语义不属于本 API（extra_repr）契约，由 ZeroPad 模块自身场景看护，未覆盖。
- 版本行为说明：2.15 的 ZeroPad3d 自定义 extra_repr，输出 f"{self.padding}"（仅元组 repr，
  不带 padding= 前缀，无 value 字段），padding 经 _ntuple(6) 归一化存储。
- 本文件同时看护别名路径 torch.nn.modules.padding.ZeroPad3d.extra_repr：torch.nn.ZeroPad3d
  与 torch.nn.modules.padding.ZeroPad3d 为同一类对象，test_dual_import_paths_same_object
  断言其同一性。
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
class TestZeroPad3dExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.ZeroPad3d.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.ZeroPad3d and torch.nn.modules.padding.ZeroPad3d are the
        same class."""
        self.assertIs(
            torch.nn.ZeroPad3d, torch.nn.modules.padding.ZeroPad3d
        )

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for int and tuple padding forms."""
        self.assertIsInstance(torch.nn.ZeroPad3d(2).extra_repr(), str)
        self.assertIsInstance(
            torch.nn.ZeroPad3d((1, 2, 3, 4, 5, 6)).extra_repr(), str
        )

    def test_extra_repr_int_padding(self):
        """Int padding is broadcast to a 6-tuple and reported as tuple repr."""
        m = torch.nn.ZeroPad3d(2)
        self.assertEqual(m.extra_repr(), "(2, 2, 2, 2, 2, 2)")

    def test_extra_repr_tuple_padding(self):
        """Asymmetric 6-tuple padding is reported in order as the tuple repr."""
        m = torch.nn.ZeroPad3d((1, 2, 3, 4, 5, 6))
        self.assertEqual(m.extra_repr(), "(1, 2, 3, 4, 5, 6)")

    def test_padding_normalized_to_sixtuple(self):
        """The stored padding attribute is a normalized 6-tuple."""
        m = torch.nn.ZeroPad3d(2)
        self.assertEqual(m.padding, (2, 2, 2, 2, 2, 2))

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.ZeroPad3d((1, 2, 3, 4, 5, 6))
        self.assertEqual(repr(m), "ZeroPad3d((1, 2, 3, 4, 5, 6))")

    def test_extra_repr_reflects_mutated_padding(self):
        """extra_repr reads the current padding attribute at call time."""
        m = torch.nn.ZeroPad3d(2)
        m.padding = (0, 1, 2, 3, 4, 5)
        self.assertEqual(m.extra_repr(), "(0, 1, 2, 3, 4, 5)")

    def test_constructor_padding_required(self):
        """ZeroPad3d requires padding (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.ZeroPad3d()

    def test_extra_repr_device_consistency(self):
        """Device-constructed ZeroPad3d has an identical representation string."""
        cpu_m = torch.nn.ZeroPad3d((1, 2, 3, 4, 5, 6))
        device_m = torch.nn.ZeroPad3d((1, 2, 3, 4, 5, 6)).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "(1, 2, 3, 4, 5, 6)")


if __name__ == "__main__":
    run_tests()
