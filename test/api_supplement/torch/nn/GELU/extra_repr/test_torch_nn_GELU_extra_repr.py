# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.GELU.extra_repr 接口功能正确性
API 名称：torch.nn.GELU.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；默认 approximate 与显式取值分别精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | approximate='none'/'tanh' 两种合法取值逐个断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 参数类型 | 返回值必须为 str 类型；字符串值经 repr() 输出带引号 | 已覆盖（test_extra_repr_string_contract 等） |
| 传参与不传参 | extra_repr 无参；approximate 有默认值，省略与显式传参分别覆盖 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 默认 'none' 边界；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_approximate） |
| 正常传参场景 | repr(module) 以类名开头且内嵌带引号的 approximate 值 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，approximate 有默认值，无稳定异常场景 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 GELU 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；构造参数有默认值，无稳定异常场景。
- 本文件同时看护别名路径 torch.nn.modules.activation.GELU.extra_repr：
  torch.nn.GELU 与 torch.nn.modules.activation.GELU 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     approximate 输出经 repr() 序列化（带单引号）；不涉及浮点 Golden 对照。
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
class TestGELUExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.GELU.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.GELU and torch.nn.modules.activation.GELU are the same class."""
        self.assertIs(torch.nn.GELU, torch.nn.modules.activation.GELU)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.GELU(), "approximate='none'"),
                name="default",
            ),
            subtest(
                (lambda: torch.nn.GELU(approximate='none'), "approximate='none'"),
                name="approximate_none",
            ),
            subtest(
                (lambda: torch.nn.GELU(approximate='tanh'), "approximate='tanh'"),
                name="approximate_tanh",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output with the quoted value."""
        self.assertEqual(repr(torch.nn.GELU()), "GELU(approximate='none')")
        self.assertEqual(repr(torch.nn.GELU(approximate="tanh")), "GELU(approximate='tanh')")

    def test_extra_repr_reflects_mutated_approximate(self):
        """extra_repr reads the current approximate attribute at call time."""
        m = torch.nn.GELU()
        m.approximate = "tanh"
        self.assertEqual(m.extra_repr(), "approximate='tanh'")

    def test_extra_repr_device_consistency(self):
        """Device-constructed GELU has an identical representation."""
        cpu_m = torch.nn.GELU(approximate="tanh")
        device_m = torch.nn.GELU(approximate="tanh").to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "approximate='tanh'")


instantiate_parametrized_tests(TestGELUExtraRepr)


if __name__ == "__main__":
    run_tests()
