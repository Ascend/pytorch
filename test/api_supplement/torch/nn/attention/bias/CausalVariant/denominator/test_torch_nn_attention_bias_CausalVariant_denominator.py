# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.bias.CausalVariant.denominator 接口功能正确性
API 名称：torch.nn.attention.bias.CausalVariant.denominator
API 签名：CausalVariant.denominator -> int
         （继承自 int 的只读 getset 属性；CausalVariant 为 IntEnum，成员 UPPER_LEFT=1、LOWER_RIGHT=2）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                             |
|------------------|-----------------------------------------------------|--------------------------------------|
| 空/非空          | 绑定常量，无空值语义；成员集合非空（2 个成员）      | 已覆盖（member_values）              |
| 枚举选项         | UPPER_LEFT / LOWER_RIGHT 两成员逐一验证精确返回值   | 已覆盖（member_values）              |
| 参数类型         | 属性返回 int                                        | 已覆盖（result_type）                |
| 传参与不传参     | 属性无参读取；只读属性赋值触发 AttributeError       | 已覆盖（member_values / read_only）  |
| 等价类/边界值    | 按 int 值 1（下界）与 2（上界）查找还原成员         | 已覆盖（member_values）              |
| 正常传参场景     | 与 int 协议取值一致；类上为 getset_descriptor       | 已覆盖（int_protocol / binding）     |
| 异常传参场景     | 只读属性赋值触发 AttributeError                     | 已覆盖（read_only_attributeerror）   |
| 混合设备类型     | 不适用：无 Tensor 入参，行为与设备无关              | 不适用                               |

未覆盖项及原因：
- 内存：不适用。绑定属性不可变，无分配/释放语义可看护。
- 反向：不适用。无 Tensor 输出，无梯度语义。
- 小值域：不适用。无数值计算精度敏感区间。
- 序列化：未覆盖。pickle 行为属 enum/int 基类协议，非本 API 看护点。

注意：纯 Python 绑定属性，执行不依赖 NPU 计算；按取值、类型、只读
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

# (member name, expected denominator) for every enum member; int denominator
# is always 1.
EXPECTED = (
    ("UPPER_LEFT", 1),
    ("LOWER_RIGHT", 1),
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCausalVariantDenominator(TestCase):
    """Test cases for torch.nn.attention.bias.CausalVariant.denominator."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_denominator_member_values(self):
        # Exact denominator for both members, plus value lookup by int.
        for name, expected in EXPECTED:
            member = getattr(CausalVariant, name)
            self.assertEqual(member.denominator, expected, name)
            self.assertEqual(CausalVariant(member.value).denominator, expected, name)

    def test_denominator_matches_int_protocol(self):
        # The binding inherits int.denominator; values must agree exactly.
        for name, _ in EXPECTED:
            member = getattr(CausalVariant, name)
            self.assertEqual(member.denominator, int(member).denominator, name)

    def test_denominator_result_type(self):
        # The attribute is a plain positive int.
        for name, _ in EXPECTED:
            denominator = getattr(CausalVariant, name).denominator
            self.assertIsInstance(denominator, int, name)
            self.assertGreater(denominator, 0, name)

    def test_denominator_class_binding(self):
        # The class attribute is the int getset descriptor exposed on the enum.
        self.assertEqual(type(CausalVariant.denominator).__name__, "getset_descriptor")

    def test_denominator_read_only_attributeerror(self):
        # int.denominator is a read-only getset attribute.
        for name, _ in EXPECTED:
            member = getattr(CausalVariant, name)
            with self.assertRaises(AttributeError, msg=name):
                member.denominator = 5

    def test_denominator_deterministic_repeated_reads(self):
        # Repeated reads with fixed members return identical values.
        first = [getattr(CausalVariant, name).denominator for name, _ in EXPECTED]
        for _ in range(3):
            again = [getattr(CausalVariant, name).denominator for name, _ in EXPECTED]
            self.assertEqual(again, first)


if __name__ == "__main__":
    run_tests()
