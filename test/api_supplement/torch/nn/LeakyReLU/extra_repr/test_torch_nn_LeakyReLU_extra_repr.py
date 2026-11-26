# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LeakyReLU.extra_repr 接口功能正确性
API 名称：torch.nn.LeakyReLU.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认/显式 negative_slope 与 inplace 组合精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | inplace=False/True 两种取值逐个断言字符串 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型；默认 1e-2 输出为 0.01 | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；negative_slope/inplace 均有默认值 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 默认 0.01；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_slope） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，构造参数均有默认值，无稳定异常 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 LeakyReLU 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；构造参数均有默认值，无稳定异常场景。
- 本文件同时看护别名路径 torch.nn.modules.activation.LeakyReLU.extra_repr：
  torch.nn.LeakyReLU 与 torch.nn.modules.activation.LeakyReLU 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     默认 negative_slope=1e-2 的 str 输出为 0.01；不涉及浮点 Golden 对照。
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
class TestLeakyReLUExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.LeakyReLU.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.LeakyReLU and torch.nn.modules.activation.LeakyReLU are the same class."""
        self.assertIs(torch.nn.LeakyReLU, torch.nn.modules.activation.LeakyReLU)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.LeakyReLU(), "negative_slope=0.01"),
                name="default",
            ),
            subtest(
                (lambda: torch.nn.LeakyReLU(0.1), "negative_slope=0.1"),
                name="explicit_positional",
            ),
            subtest(
                (lambda: torch.nn.LeakyReLU(negative_slope=0.2), "negative_slope=0.2"),
                name="explicit_keyword",
            ),
            subtest(
                (lambda: torch.nn.LeakyReLU(inplace=True), "negative_slope=0.01, inplace=True"),
                name="inplace_true",
            ),
            subtest(
                (lambda: torch.nn.LeakyReLU(0.05, True), "negative_slope=0.05, inplace=True"),
                name="positional_inplace",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.LeakyReLU()), "LeakyReLU(negative_slope=0.01)")
        self.assertEqual(
            repr(torch.nn.LeakyReLU(0.05, True)), "LeakyReLU(negative_slope=0.05, inplace=True)"
        )

    def test_extra_repr_reflects_mutated_slope(self):
        """extra_repr reads the current negative_slope attribute at call time."""
        m = torch.nn.LeakyReLU()
        m.negative_slope = 0.3
        self.assertEqual(m.extra_repr(), "negative_slope=0.3")

    def test_extra_repr_device_consistency(self):
        """Device-constructed LeakyReLU has an identical representation."""
        cpu_m = torch.nn.LeakyReLU(0.1)
        device_m = torch.nn.LeakyReLU(0.1).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "negative_slope=0.1")


instantiate_parametrized_tests(TestLeakyReLUExtraRepr)


if __name__ == "__main__":
    run_tests()
