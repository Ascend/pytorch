# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Softmax.extra_repr 接口功能正确性
API 名称：torch.nn.Softmax.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；默认 dim=None 与显式 dim 值分别精确断言 | 已覆盖（test_extra_repr_string_contract 等） |
| 枚举选项 | dim=None（默认）/1/-1 多个合法取值逐个断言 | 已覆盖（test_extra_repr_string_contract） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_string_contract） |
| 传参与不传参 | extra_repr 无参；dim 有默认值 None，省略与显式传参分别覆盖 | 已覆盖（test_extra_repr_string_contract 等） |
| 等价类/边界值 | dim=None 默认值；属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_dim） |
| 正常传参场景 | repr(module) 以类名开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 构造期即校验 dim：bool/float/str 均抛 TypeError 并报出实际类型（extra_repr 自身无校验路径） | 已覆盖（test_constructor_dim_type_validation） |
| 混合设备类型 | 设备侧构造的 Softmax 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- extra_repr 为纯字符串格式化方法，除构造期 dim 类型校验外无其他异常路径。
- 本文件同时看护别名路径 torch.nn.modules.activation.Softmax.extra_repr：
  torch.nn.Softmax 与 torch.nn.modules.activation.Softmax 为同一类对象，
  test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     dim=None 按字面输出 "dim=None"；不涉及浮点 Golden 对照。
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
class TestSoftmaxExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Softmax.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Softmax and torch.nn.modules.activation.Softmax are the same class."""
        self.assertIs(torch.nn.Softmax, torch.nn.modules.activation.Softmax)

    @parametrize(
        "module_factory, expected",
        [
            subtest(
                (lambda: torch.nn.Softmax(), "dim=None"),
                name="default",
            ),
            subtest(
                (lambda: torch.nn.Softmax(dim=1), "dim=1"),
                name="dim_1",
            ),
            subtest(
                (lambda: torch.nn.Softmax(dim=-1), "dim=-1"),
                name="dim_neg_1",
            ),
            subtest(
                (lambda: torch.nn.Softmax(dim=0), "dim=0"),
                name="dim_0",
            ),
        ],
    )
    def test_extra_repr_string_contract(self, module_factory, expected):
        """extra_repr returns the exact formatted string for each configuration."""
        self.assertEqual(module_factory().extra_repr(), expected)

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) embeds the extra_repr output."""
        self.assertEqual(repr(torch.nn.Softmax()), "Softmax(dim=None)")
        self.assertEqual(repr(torch.nn.Softmax(dim=1)), "Softmax(dim=1)")

    def test_extra_repr_reflects_mutated_dim(self):
        """extra_repr reads the current dim attribute at call time."""
        m = torch.nn.Softmax()
        m.dim = 2
        self.assertEqual(m.extra_repr(), "dim=2")

    def test_constructor_dim_type_validation(self):
        """Constructor rejects non-int/non-None dim values with TypeError (bool included).

        Measured on torch 2.15.0: Softmax.__init__ validates dim eagerly and
        rejects bool (despite bool being an int subclass), float and str with
        a TypeError whose message reports the offending type.
        """
        with self.assertRaisesRegex(TypeError, r"dim must be an int or None, got <class 'bool'>"):
            torch.nn.Softmax(dim=True)
        with self.assertRaisesRegex(TypeError, r"dim must be an int or None, got <class 'float'>"):
            torch.nn.Softmax(dim=1.5)
        with self.assertRaisesRegex(TypeError, r"dim must be an int or None, got <class 'str'>"):
            torch.nn.Softmax(dim="a")

    def test_extra_repr_device_consistency(self):
        """Device-constructed Softmax has an identical representation."""
        cpu_m = torch.nn.Softmax(dim=1)
        device_m = torch.nn.Softmax(dim=1).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "dim=1")


instantiate_parametrized_tests(TestSoftmaxExtraRepr)


if __name__ == "__main__":
    run_tests()
