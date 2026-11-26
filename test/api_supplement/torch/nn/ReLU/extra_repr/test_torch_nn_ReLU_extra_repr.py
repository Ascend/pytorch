# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.ReLU.extra_repr 接口功能正确性
API 名称：torch.nn.ReLU.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认构造输出空字符串（空/非空两种形态均精确断言） | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | inplace=False/True 两种取值逐个断言字符串 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型（含空字符串） | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；inplace 有默认值 False，省略与显式传参分别覆盖 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 空字符串为默认形态；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_inplace） |
| 正常传参场景 | repr(module) 在默认形态下仅含类名，inplace 形态内嵌标记 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，inplace 有默认值，无稳定异常场景 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 ReLU 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；inplace 有默认值，无稳定异常场景。
- 本文件同时看护别名路径 torch.nn.modules.activation.ReLU.extra_repr：
  torch.nn.ReLU 与 torch.nn.modules.activation.ReLU 为同一类对象，
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
class TestReLUExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.ReLU.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.ReLU and torch.nn.modules.activation.ReLU are the same class."""
        self.assertIs(torch.nn.ReLU, torch.nn.modules.activation.ReLU)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.ReLU(), ""),
                name="default_empty",
            ),
            subtest(
                (lambda: torch.nn.ReLU(inplace=True), "inplace=True"),
                name="inplace_true",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output (empty by default)."""
        self.assertEqual(repr(torch.nn.ReLU()), "ReLU()")
        self.assertEqual(repr(torch.nn.ReLU(inplace=True)), "ReLU(inplace=True)")

    def test_extra_repr_reflects_mutated_inplace(self):
        """extra_repr reads the current inplace attribute at call time."""
        m = torch.nn.ReLU()
        m.inplace = True
        self.assertEqual(m.extra_repr(), "inplace=True")

    def test_extra_repr_device_consistency(self):
        """Device-constructed ReLU has an identical representation."""
        cpu_m = torch.nn.ReLU(inplace=True)
        device_m = torch.nn.ReLU(inplace=True).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "inplace=True")


instantiate_parametrized_tests(TestReLUExtraRepr)


if __name__ == "__main__":
    run_tests()
