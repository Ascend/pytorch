# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Softplus.extra_repr 接口功能正确性
API 名称：torch.nn.Softplus.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认/显式 beta/threshold 组合分别精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | beta/threshold 默认值与显式值逐项断言 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；beta/threshold 均有默认值，支持部分显式传参 | 已覆盖（test_extra_repr_string_contract） |
| 等价类/边界值 | 默认 1.0/20.0；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_threshold） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，构造参数均有默认值，无稳定异常 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧构造的 Softplus 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；构造参数均有默认值，无稳定异常场景。
- 本文件同时看护别名路径 torch.nn.modules.activation.Softplus.extra_repr：
  torch.nn.Softplus 与 torch.nn.modules.activation.Softplus 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
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
class TestSoftplusExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Softplus.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Softplus and torch.nn.modules.activation.Softplus are the same class."""
        self.assertIs(torch.nn.Softplus, torch.nn.modules.activation.Softplus)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.Softplus(), "beta=1.0, threshold=20.0"),
                name="default_params",
            ),
            subtest(
                (lambda: torch.nn.Softplus(beta=2.0, threshold=15.0), "beta=2.0, threshold=15.0"),
                name="explicit_params",
            ),
            subtest(
                (lambda: torch.nn.Softplus(threshold=10.0), "beta=1.0, threshold=10.0"),
                name="partial_threshold",
            ),
            subtest(
                (lambda: torch.nn.Softplus(beta=1.5), "beta=1.5, threshold=20.0"),
                name="partial_beta",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.Softplus()), "Softplus(beta=1.0, threshold=20.0)")
        self.assertEqual(
            repr(torch.nn.Softplus(beta=2.0, threshold=15.0)),
            "Softplus(beta=2.0, threshold=15.0)",
        )

    def test_extra_repr_reflects_mutated_threshold(self):
        """extra_repr reads the current threshold attribute at call time."""
        m = torch.nn.Softplus()
        m.threshold = 12.0
        self.assertEqual(m.extra_repr(), "beta=1.0, threshold=12.0")

    def test_extra_repr_device_consistency(self):
        """Device-constructed Softplus has an identical representation."""
        cpu_m = torch.nn.Softplus(beta=2.0, threshold=15.0)
        device_m = torch.nn.Softplus(beta=2.0, threshold=15.0).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "beta=2.0, threshold=15.0")


instantiate_parametrized_tests(TestSoftplusExtraRepr)


if __name__ == "__main__":
    run_tests()
