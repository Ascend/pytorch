# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantization_mappings.get_dynamic_quant_module_class 接口功能正确性
API 名称：torch.ao.quantization.quantization_mappings.get_dynamic_quant_module_class
API 签名：get_dynamic_quant_module_class(float_module_class, additional_dynamic_quant_mapping=None) -> Any

覆盖维度表：
| 覆盖维度      | 说明                                  | 覆盖情况                                                 |
|---------------|---------------------------------------|----------------------------------------------------------|
| 空/非空       | additional_dynamic_quant_mapping 传   | 已覆盖：test_default_mapping_arg_none /                  |
|               | None(空) 与非空 dict                  | test_additional_mapping_adds_new_key                     |
| 枚举选项      | 该接口无枚举参数                      | 不适用                                                   |
| 参数类型      | float_module_class 为类，additional   | 已覆盖：test_additional_mapping_adds_new_key             |
|               | 为 dict                               |                                                          |
| 传参与不传参  | 省略 additional（默认                 | 已覆盖：test_default_mapping_arg_none /                  |
|               | None）与显式传入                      | test_explicit_empty_mapping_matches_default              |
| 等价类/边界值 | 已知类命中 / 未知类抛 AssertionError  | 已覆盖：test_known_class_returns_dynamic_class /         |
|               |                                       | test_unknown_class_raises                                |
| 正常传参场景  | 默认映射与附加映射覆盖既有键/新增键   | 已覆盖：test_additional_mapping_overrides_existing_key   |
| 异常传参场景  | 未知浮点模块类抛 AssertionError       | 已覆盖：test_unknown_class_raises /                      |
|               |                                       | test_unknown_class_message                               |
| 混合设备类型  | 纯 Python 映射查询，无设备执行路径    | 不适用                                                   |

未覆盖项及原因：
- 混合设备类型：该接口仅查询模块类映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 映射工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回类对象与异常契约做断言。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn
from torch.ao.quantization.quantization_mappings import get_dynamic_quant_module_class

_ERROR_RE = "does not have a corresponding quantized module class"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetDynamicQuantModuleClassUt(TestCase):
    """Functional tests for quantization_mappings.get_dynamic_quant_module_class."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_known_class_returns_dynamic_class(self):
        """A known float module class maps to its dynamic quantized class."""
        self.assertIs(get_dynamic_quant_module_class(torch.nn.Linear), nnqd.Linear)
        self.assertIs(get_dynamic_quant_module_class(torch.nn.LSTM), nnqd.LSTM)
        self.assertIs(get_dynamic_quant_module_class(torch.nn.GRU), nnqd.GRU)
        self.assertIs(get_dynamic_quant_module_class(torch.nn.GRUCell), nnqd.GRUCell)

    def test_returned_object_is_a_class(self):
        """The returned object is a class, not an instance."""
        result = get_dynamic_quant_module_class(torch.nn.Linear)
        self.assertIsInstance(result, type)

    def test_default_mapping_arg_none(self):
        """Omitting the additional mapping argument resolves the default mapping."""
        self.assertIs(get_dynamic_quant_module_class(torch.nn.Linear), nnqd.Linear)

    def test_explicit_empty_mapping_matches_default(self):
        """An explicit empty additional mapping behaves like the default None."""
        explicit = get_dynamic_quant_module_class(torch.nn.Linear, {})
        omitted = get_dynamic_quant_module_class(torch.nn.Linear)
        self.assertIs(explicit, nnqd.Linear)
        self.assertIs(explicit, omitted)

    def test_additional_mapping_adds_new_key(self):
        """An additional mapping for a previously unknown key is honored."""
        result = get_dynamic_quant_module_class(torch.nn.Identity, {torch.nn.Identity: nnqd.Linear})
        self.assertIs(result, nnqd.Linear)

    def test_additional_mapping_overrides_existing_key(self):
        """An additional mapping value overrides the default for the same key."""
        result = get_dynamic_quant_module_class(torch.nn.Linear, {torch.nn.Linear: nnq.Conv2d})
        self.assertIs(result, nnq.Conv2d)

    def test_unknown_class_raises(self):
        """An unknown float module class raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_dynamic_quant_module_class(torch.nn.Conv2d)

    def test_unknown_class_message(self):
        """The AssertionError message names the missing quantized module class."""
        with self.assertRaisesRegex(AssertionError, _ERROR_RE):
            get_dynamic_quant_module_class(torch.nn.Identity)

    def test_additional_mapping_not_mutated(self):
        """The caller-supplied additional mapping is not modified in place."""
        additional = {torch.nn.Identity: nnqd.Linear}
        get_dynamic_quant_module_class(torch.nn.Identity, additional)
        self.assertEqual(set(additional.keys()), {torch.nn.Identity})
        self.assertIs(additional[torch.nn.Identity], nnqd.Linear)


if __name__ == "__main__":
    run_tests()
