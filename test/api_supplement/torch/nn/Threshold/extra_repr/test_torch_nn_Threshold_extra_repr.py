# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Threshold.extra_repr 接口功能正确性
API 名称：torch.nn.Threshold.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认/显式 threshold/value 与 inplace 组合精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | inplace=False/True 两种取值逐个断言字符串 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型；float 入参按 str 输出 | 已覆盖（test_extra_repr_string_contract 等） |
| 传参与不传参 | extra_repr 无参；threshold/value 为必填构造参数，inplace 有默认值 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 负值 value=-1.0 输出；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_params） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺 threshold/value 触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_params_required） |
| 混合设备类型 | 设备侧构造的 Threshold 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，除构造缺参 TypeError 外无其他异常路径。
- 本文件同时看护别名路径 torch.nn.modules.activation.Threshold.extra_repr：
  torch.nn.Threshold 与 torch.nn.modules.activation.Threshold 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     输出格式为 "threshold=..., value=..."，inplace=True 以逗号后缀追加；
     不涉及浮点 Golden 对照。
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
class TestThresholdExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Threshold.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Threshold and torch.nn.modules.activation.Threshold are the same class."""
        self.assertIs(torch.nn.Threshold, torch.nn.modules.activation.Threshold)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.Threshold(0.1, 2.0), "threshold=0.1, value=2.0"),
                name="explicit_params",
            ),
            subtest(
                (lambda: torch.nn.Threshold(0.25, 5.0), "threshold=0.25, value=5.0"),
                name="explicit_params_2",
            ),
            subtest(
                (lambda: torch.nn.Threshold(0.5, -1.0, inplace=True), "threshold=0.5, value=-1.0, inplace=True"),
                name="inplace_suffix",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.Threshold(0.1, 2.0)), "Threshold(threshold=0.1, value=2.0)")
        self.assertEqual(
            repr(torch.nn.Threshold(0.5, -1.0, inplace=True)),
            "Threshold(threshold=0.5, value=-1.0, inplace=True)",
        )

    def test_extra_repr_reflects_mutated_params(self):
        """extra_repr reads the current threshold/value attributes at call time."""
        m = torch.nn.Threshold(0.1, 2.0)
        m.threshold = 0.5
        m.value = 3.0
        self.assertEqual(m.extra_repr(), "threshold=0.5, value=3.0")

    def test_constructor_params_required(self):
        """Threshold requires both threshold and value (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.Threshold()
        with self.assertRaises(TypeError):
            torch.nn.Threshold(0.1)

    def test_extra_repr_device_consistency(self):
        """Device-constructed Threshold has an identical representation."""
        cpu_m = torch.nn.Threshold(0.1, 2.0)
        device_m = torch.nn.Threshold(0.1, 2.0).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "threshold=0.1, value=2.0")


instantiate_parametrized_tests(TestThresholdExtraRepr)


if __name__ == "__main__":
    run_tests()
