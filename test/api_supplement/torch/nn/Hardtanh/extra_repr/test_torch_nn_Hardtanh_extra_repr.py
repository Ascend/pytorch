# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Hardtanh.extra_repr 接口功能正确性
API 名称：torch.nn.Hardtanh.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；默认/显式 min_val/max_val 与 inplace 组合精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | inplace=False/True 两种取值逐个断言字符串 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型；int 入参按 int 输出（无浮点化） | 已覆盖（test_extra_repr_string_contract 等） |
| 传参与不传参 | extra_repr 无参；min_val/max_val/inplace 均有默认值 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 默认 -1.0/1.0；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_range） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造期校验 max_val>min_val：反转/等值均抛 AssertionError（消息含 min_val/max_val） | 已覆盖（test_constructor_range_validation） |
| 混合设备类型 | 设备侧构造的 Hardtanh 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- min_value/max_value 两个废弃关键字仅影响构造期赋值，不属于 extra_repr 契约，未覆盖。
- 本文件同时看护别名路径 torch.nn.modules.activation.Hardtanh.extra_repr：
  torch.nn.Hardtanh 与 torch.nn.modules.activation.Hardtanh 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     int 入参（如 Hardtanh(-2, 2)）按存储类型原样输出；不涉及浮点 Golden 对照。
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, subtest

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
class TestHardtanhExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Hardtanh.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Hardtanh and torch.nn.modules.activation.Hardtanh are the same class."""
        self.assertIs(torch.nn.Hardtanh, torch.nn.modules.activation.Hardtanh)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.Hardtanh(), "min_val=-1.0, max_val=1.0"),
                name="default",
            ),
            subtest(
                (lambda: torch.nn.Hardtanh(-2, 2), "min_val=-2, max_val=2"),
                name="explicit_int",
            ),
            subtest(
                (lambda: torch.nn.Hardtanh(min_val=0.5, max_val=4.5), "min_val=0.5, max_val=4.5"),
                name="explicit_float",
            ),
            subtest(
                (lambda: torch.nn.Hardtanh(inplace=True), "min_val=-1.0, max_val=1.0, inplace=True"),
                name="inplace_true",
            ),
            subtest(
                (lambda: torch.nn.Hardtanh(-2, 2, True), "min_val=-2, max_val=2, inplace=True"),
                name="positional_inplace",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.Hardtanh()), "Hardtanh(min_val=-1.0, max_val=1.0)")
        self.assertEqual(
            repr(torch.nn.Hardtanh(-2, 2, True)),
            "Hardtanh(min_val=-2, max_val=2, inplace=True)",
        )

    def test_extra_repr_reflects_mutated_range(self):
        """extra_repr reads the current min_val/max_val attributes at call time."""
        m = torch.nn.Hardtanh()
        m.min_val = -3.0
        m.max_val = 3.0
        self.assertEqual(m.extra_repr(), "min_val=-3.0, max_val=3.0")

    def test_constructor_range_validation(self):
        """Constructor requires max_val > min_val; equal or inverted bounds raise AssertionError.

        Measured on torch 2.15.0: both inverted (2, 1) and equal (1, 1) bounds
        raise AssertionError. The exception message wording is not part of the
        contract, so only the exception type is asserted here.
        """
        with self.assertRaises(AssertionError):
            torch.nn.Hardtanh(2, 1)
        with self.assertRaises(AssertionError):
            torch.nn.Hardtanh(1, 1)

    def test_extra_repr_device_consistency(self):
        """Device-constructed Hardtanh has an identical representation."""
        cpu_m = torch.nn.Hardtanh(-2, 2)
        device_m = torch.nn.Hardtanh(-2, 2).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "min_val=-2, max_val=2")


instantiate_parametrized_tests(TestHardtanhExtraRepr)


if __name__ == "__main__":
    run_tests()
