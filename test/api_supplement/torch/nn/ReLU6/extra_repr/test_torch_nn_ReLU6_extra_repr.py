# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.ReLU6.extra_repr 接口功能正确性
API 名称：torch.nn.ReLU6.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认构造输出空字符串（空/非空两种形态均精确断言） | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | inplace=False/True 两种取值逐个断言字符串 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型（含空字符串） | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；inplace 有默认值 False，省略与显式传参分别覆盖 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 空字符串为默认形态；继承的 min_val/max_val 不参与输出 | 已覆盖（test_extra_repr_override_ignores_inherited_range） |
| 正常传参场景 | repr(module) 在默认形态下仅含类名，inplace 形态内嵌标记 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，inplace 有默认值，无稳定异常场景 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 ReLU6 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；inplace 有默认值，无稳定异常场景。
- ReLU6 继承 Hardtanh 的 min_val=0.0/max_val=6.0 属性，但其覆写的 extra_repr 不输出该范围，
  专例断言该覆写行为。
- 本文件同时看护别名路径 torch.nn.modules.activation.ReLU6.extra_repr：
  torch.nn.ReLU6 与 torch.nn.modules.activation.ReLU6 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     默认（inplace=False）时 extra_repr 返回空字符串；不涉及浮点 Golden 对照。
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
class TestReLU6ExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.ReLU6.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.ReLU6 and torch.nn.modules.activation.ReLU6 are the same class."""
        self.assertIs(torch.nn.ReLU6, torch.nn.modules.activation.ReLU6)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.ReLU6(), ""),
                name="default_empty",
            ),
            subtest(
                (lambda: torch.nn.ReLU6(inplace=True), "inplace=True"),
                name="inplace_true",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_override_ignores_inherited_range(self):
        """ReLU6 carries inherited min_val=0.0/max_val=6.0 but its own extra_repr
        override does not output the range."""
        m = torch.nn.ReLU6()
        self.assertEqual(m.min_val, 0.0)
        self.assertEqual(m.max_val, 6.0)
        self.assertEqual(m.extra_repr(), "")

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output (empty by default)."""
        self.assertEqual(repr(torch.nn.ReLU6()), "ReLU6()")
        self.assertEqual(repr(torch.nn.ReLU6(inplace=True)), "ReLU6(inplace=True)")

    def test_extra_repr_reflects_mutated_inplace(self):
        """extra_repr reads the current inplace attribute at call time."""
        m = torch.nn.ReLU6()
        m.inplace = True
        self.assertEqual(m.extra_repr(), "inplace=True")

    def test_extra_repr_device_consistency(self):
        """Device-constructed ReLU6 has an identical representation."""
        cpu_m = torch.nn.ReLU6(inplace=True)
        device_m = torch.nn.ReLU6(inplace=True).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "inplace=True")


instantiate_parametrized_tests(TestReLU6ExtraRepr)


if __name__ == "__main__":
    run_tests()
