# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.bias.CausalVariant.as_integer_ratio 接口功能正确性
API 名称：torch.nn.attention.bias.CausalVariant.as_integer_ratio
API 签名：CausalVariant.as_integer_ratio(self) -> tuple[int, int]
         （继承自 int 的绑定方法；CausalVariant 为 IntEnum，成员 UPPER_LEFT=1、LOWER_RIGHT=2）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                             |
|------------------|-----------------------------------------------------|--------------------------------------|
| 空/非空          | 绑定常量，无空值语义；成员集合非空（2 个成员）      | 已覆盖（member_values）              |
| 枚举选项         | UPPER_LEFT / LOWER_RIGHT 两成员逐一验证精确返回值   | 已覆盖（member_values）              |
| 参数类型         | 返回 tuple，两元素均为 int                          | 已覆盖（return_type）                |
| 传参与不传参     | 方法无参调用；多传位置参数触发 TypeError            | 已覆盖（member_values / extra_arg）  |
| 等价类/边界值    | 按 int 值 1（下界）与 2（上界）查找还原成员         | 已覆盖（member_values）              |
| 正常传参场景     | 与 int 协议返回值一致；类上为 method_descriptor     | 已覆盖（int_protocol / binding）     |
| 异常传参场景     | 多传位置参数触发 TypeError                          | 已覆盖（extra_arg_typeerror）        |
| 混合设备类型     | 不适用：无 Tensor 入参，行为与设备无关              | 不适用                               |

未覆盖项及原因：
- 内存：不适用。绑定方法不可变，无分配/释放语义可看护。
- 反向：不适用。无 Tensor 输出，无梯度语义。
- 小值域：不适用。无数值计算精度敏感区间。
- 序列化：未覆盖。pickle 行为属 enum/int 基类协议，非本 API 看护点。

注意：纯 Python 绑定方法，执行不依赖 NPU 计算；按返回值、类型与异常
     行为断言，与 int 协议逐项对照，无 CPU Golden 浮点对照路径。
"""
import os
import unittest

import torch  # noqa: F401

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Resolve device only when tests actually run; keep default CI collection cheap.
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


from torch.nn.attention.bias import CausalVariant  # noqa: E402

# These tests assume IntEnum members (CausalVariant inherits int); pin the
# premise so an upstream switch back to plain Enum fails loudly here instead
# of raising AttributeError across every test below.
assert issubclass(CausalVariant, int), (
    "CausalVariant must stay an IntEnum for the int-protocol assertions below")

# (member name, expected as_integer_ratio()) for every enum member.
EXPECTED = (
    ("UPPER_LEFT", (1, 1)),
    ("LOWER_RIGHT", (2, 1)),
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCausalVariantAsIntegerRatio(TestCase):
    """Test cases for torch.nn.attention.bias.CausalVariant.as_integer_ratio."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_as_integer_ratio_member_values(self):
        # Exact (numerator, denominator) pairs for both members, plus value lookup.
        for name, expected in EXPECTED:
            member = getattr(CausalVariant, name)
            self.assertEqual(member.as_integer_ratio(), expected, name)
            self.assertEqual(CausalVariant(member.value).as_integer_ratio(), expected, name)

    def test_as_integer_ratio_matches_int_protocol(self):
        # The binding inherits int.as_integer_ratio; results must agree exactly.
        for name, _ in EXPECTED:
            member = getattr(CausalVariant, name)
            self.assertEqual(member.as_integer_ratio(), int(member).as_integer_ratio(), name)

    def test_as_integer_ratio_return_type(self):
        # tuple of two positive ints whose division reproduces the member value.
        for name, _ in EXPECTED:
            member = getattr(CausalVariant, name)
            ratio = member.as_integer_ratio()
            self.assertIsInstance(ratio, tuple, name)
            self.assertEqual(len(ratio), 2, name)
            numerator, denominator = ratio
            self.assertIsInstance(numerator, int, name)
            self.assertIsInstance(denominator, int, name)
            self.assertGreater(denominator, 0, name)
            self.assertEqual(numerator, member.value, name)
            self.assertEqual(denominator, 1, name)

    def test_as_integer_ratio_class_binding(self):
        # The class attribute is the int method descriptor exposed on the enum.
        self.assertEqual(type(CausalVariant.as_integer_ratio).__name__, "method_descriptor")
        for name, _ in EXPECTED:
            self.assertTrue(callable(getattr(CausalVariant, name).as_integer_ratio), name)

    def test_as_integer_ratio_extra_arg_typeerror(self):
        # int.as_integer_ratio takes no extra positional argument.
        for name, _ in EXPECTED:
            member = getattr(CausalVariant, name)
            with self.assertRaises(TypeError, msg=name):
                member.as_integer_ratio(0)

    def test_as_integer_ratio_deterministic_repeated_reads(self):
        # Repeated calls with fixed members return identical pairs.
        first = [getattr(CausalVariant, name).as_integer_ratio() for name, _ in EXPECTED]
        for _ in range(3):
            again = [getattr(CausalVariant, name).as_integer_ratio() for name, _ in EXPECTED]
            self.assertEqual(again, first)


if __name__ == "__main__":
    run_tests()
