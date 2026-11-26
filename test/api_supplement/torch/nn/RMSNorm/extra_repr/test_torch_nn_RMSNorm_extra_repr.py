# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.RMSNorm.extra_repr 接口功能正确性
API 名称：torch.nn.RMSNorm.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认与显式 normalized_shape/eps/affine 组合精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | normalized_shape 接受 int/list/tuple 三种形态逐个断言 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型；normalized_shape 输出为元组 repr | 已覆盖（test_extra_repr_string_contract 等） |
| 传参与不传参 | extra_repr 无参；eps/elementwise_affine 有默认值，normalized_shape 必填 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | eps=None 默认与显式 1e-6 的格式化；属性修改后同步反映 | 已覆盖（test_extra_repr_reflects_mutated_eps） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造缺 normalized_shape 触发 TypeError（extra_repr 自身无校验路径） | 已覆盖（test_constructor_normalized_shape_required） |
| 混合设备类型 | 设备侧以 factory device 参数构造 RMSNorm，字符串与 CPU 实例一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- eps 为 None 时 extra_repr 输出字面 "None"；具体 eps 数值仅影响前向计算，不属于 repr 契约。
- torch.nn.RMSNorm 在 2.15 无第二条公开导入路径别名条目；本文件断言其与
  torch.nn.modules.normalization.RMSNorm 为同一类对象。
- 设备构造测试在 NPU/GPU 上真实分配 weight 参数张量；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     输出格式为 "{normalized_shape}, eps=..., elementwise_affine=..."；不涉及浮点 Golden 对照。
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
class TestRMSNormExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.RMSNorm.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_same_class_as_normalization_module(self):
        """torch.nn.RMSNorm is torch.nn.modules.normalization.RMSNorm."""
        self.assertIs(torch.nn.RMSNorm, torch.nn.modules.normalization.RMSNorm)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.RMSNorm(4), "(4,), eps=None, elementwise_affine=True"),
                name="default_shape",
            ),
            subtest(
                (lambda: torch.nn.RMSNorm([2, 3]), "(2, 3), eps=None, elementwise_affine=True"),
                name="shape_list",
            ),
            subtest(
                (lambda: torch.nn.RMSNorm((4, 5)), "(4, 5), eps=None, elementwise_affine=True"),
                name="shape_tuple",
            ),
            subtest(
                (lambda: torch.nn.RMSNorm(4, eps=1e-6), "(4,), eps=1e-06, elementwise_affine=True"),
                name="explicit_eps",
            ),
            subtest(
                (lambda: torch.nn.RMSNorm(4, elementwise_affine=False), "(4,), eps=None, elementwise_affine=False"),
                name="no_affine",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(
            repr(torch.nn.RMSNorm(4)), "RMSNorm((4,), eps=None, elementwise_affine=True)"
        )

    def test_extra_repr_reflects_mutated_eps(self):
        """extra_repr reads the current eps attribute at call time."""
        m = torch.nn.RMSNorm(4)
        m.eps = 1e-5
        self.assertEqual(m.extra_repr(), "(4,), eps=1e-05, elementwise_affine=True")

    def test_constructor_normalized_shape_required(self):
        """RMSNorm requires the normalized_shape argument (TypeError when missing)."""
        with self.assertRaises(TypeError):
            torch.nn.RMSNorm()

    def test_extra_repr_device_consistency(self):
        """Device-constructed RMSNorm allocates weight on device with the same repr."""
        cpu_m = torch.nn.RMSNorm(4)
        device_m = torch.nn.RMSNorm(4, device=self.device_name)
        self.assertEqual(device_m.weight.device.type, self.device_name)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "(4,), eps=None, elementwise_affine=True")


instantiate_parametrized_tests(TestRMSNormExtraRepr)


if __name__ == "__main__":
    run_tests()
