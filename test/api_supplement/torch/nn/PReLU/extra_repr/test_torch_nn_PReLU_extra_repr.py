# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.PReLU.extra_repr 接口功能正确性
API 名称：torch.nn.PReLU.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对默认与显式 num_parameters 值分别精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | num_parameters=1（默认）/4/8 多个取值逐个断言 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；num_parameters 有默认值 1，省略与显式传参分别覆盖 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | 默认 1；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_num_parameters） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | extra_repr 无校验路径，num_parameters 有默认值，无稳定异常 | 未覆盖（无稳定异常） |
| 混合设备类型 | 设备侧以 factory device 参数构造 PReLU，字符串与 CPU 实例一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，无异常路径；num_parameters 有默认值，无稳定异常场景。
- init 参数只影响 weight 初始化，不属于 extra_repr 契约，未覆盖（reset_parameters 另有文件看护）。
- 本文件同时看护别名路径 torch.nn.modules.activation.PReLU.extra_repr：
  torch.nn.PReLU 与 torch.nn.modules.activation.PReLU 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实分配 weight 参数张量；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     设备侧验证真实参数张量分配与表示行为，不涉及浮点数值 Golden 对照。
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
class TestPReLUExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.PReLU.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.PReLU and torch.nn.modules.activation.PReLU are the same class."""
        self.assertIs(torch.nn.PReLU, torch.nn.modules.activation.PReLU)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.PReLU(), "num_parameters=1"),
                name="default",
            ),
            subtest(
                (lambda: torch.nn.PReLU(4), "num_parameters=4"),
                name="explicit_positional",
            ),
            subtest(
                (lambda: torch.nn.PReLU(num_parameters=8), "num_parameters=8"),
                name="explicit_keyword",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.PReLU()), "PReLU(num_parameters=1)")
        self.assertEqual(repr(torch.nn.PReLU(4)), "PReLU(num_parameters=4)")

    def test_extra_repr_reflects_mutated_num_parameters(self):
        """extra_repr reads the current num_parameters attribute at call time."""
        m = torch.nn.PReLU()
        m.num_parameters = 16
        self.assertEqual(m.extra_repr(), "num_parameters=16")

    def test_extra_repr_independent_of_weight_value(self):
        """extra_repr reports the configured count, not the weight content."""
        m = torch.nn.PReLU(3)
        with torch.no_grad():
            m.weight.fill_(7.5)
        self.assertEqual(m.extra_repr(), "num_parameters=3")

    def test_extra_repr_device_consistency(self):
        """Device-constructed PReLU allocates weight on device with the same repr."""
        cpu_m = torch.nn.PReLU(4)
        device_m = torch.nn.PReLU(4, device=self.device_name)
        self.assertEqual(device_m.weight.device.type, self.device_name)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "num_parameters=4")


instantiate_parametrized_tests(TestPReLUExtraRepr)


if __name__ == "__main__":
    run_tests()
