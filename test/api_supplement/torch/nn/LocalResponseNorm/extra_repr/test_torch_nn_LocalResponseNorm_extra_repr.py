# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LocalResponseNorm.extra_repr 接口功能正确性
API 名称：torch.nn.LocalResponseNorm.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认/显式 size/alpha/beta/k 组合精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | size/alpha/beta/k 默认值与显式值逐项断言 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型；size 以裸值输出（无 size= 前缀） | 已覆盖（test_extra_repr_string_contract 等） |
| 传参与不传参 | extra_repr 无参；alpha/beta/k 有默认值，size 必填 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 默认 alpha=0.0001 格式化；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_attrs） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺 size 触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_size_required） |
| 混合设备类型 | 设备侧构造的 LocalResponseNorm 与 CPU 构造实例输出一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无其他稳定异常路径。
- 本文件同时看护别名路径 torch.nn.modules.normalization.LocalResponseNorm.extra_repr：
  torch.nn.LocalResponseNorm 与 torch.nn.modules.normalization.LocalResponseNorm 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性；torch.nn.modules.activation 命名空间
  不再导出 LocalResponseNorm（test_activation_namespace_has_no_lrn 记录该事实）。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     输出格式为 "{size}, alpha=..., beta=..., k=..."（size 裸值在前）；不涉及浮点 Golden 对照。
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, subtest
import torch.nn.modules.activation as activation_ns

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
class TestLocalResponseNormExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.LocalResponseNorm.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.LocalResponseNorm and torch.nn.modules.normalization.LocalResponseNorm
        are the same class."""
        self.assertIs(torch.nn.LocalResponseNorm, torch.nn.modules.normalization.LocalResponseNorm)

    def test_activation_namespace_has_no_lrn(self):
        """torch.nn.modules.activation does not re-export LocalResponseNorm in torch 2.15;
        the canonical module namespace is torch.nn.modules.normalization."""
        self.assertFalse(hasattr(activation_ns, "LocalResponseNorm"))

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.LocalResponseNorm(5), "5, alpha=0.0001, beta=0.75, k=1.0"),
                name="default_params",
            ),
            subtest(
                (lambda: torch.nn.LocalResponseNorm(3, alpha=0.5, beta=0.9, k=2.0), "3, alpha=0.5, beta=0.9, k=2.0"),
                name="explicit_params",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(
            repr(torch.nn.LocalResponseNorm(5)),
            "LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)",
        )

    def test_extra_repr_reflects_mutated_attrs(self):
        """extra_repr reads the current attributes at call time."""
        m = torch.nn.LocalResponseNorm(5)
        m.size = 7
        m.alpha = 0.25
        self.assertEqual(m.extra_repr(), "7, alpha=0.25, beta=0.75, k=1.0")

    def test_constructor_size_required(self):
        """LocalResponseNorm requires the size argument (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.LocalResponseNorm()

    def test_extra_repr_device_consistency(self):
        """Device-constructed LocalResponseNorm has an identical representation."""
        cpu_m = torch.nn.LocalResponseNorm(5)
        device_m = torch.nn.LocalResponseNorm(5).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "5, alpha=0.0001, beta=0.75, k=1.0")


instantiate_parametrized_tests(TestLocalResponseNormExtraRepr)


if __name__ == "__main__":
    run_tests()
