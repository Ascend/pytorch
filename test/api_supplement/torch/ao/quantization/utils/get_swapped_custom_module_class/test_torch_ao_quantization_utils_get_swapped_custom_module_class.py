# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.get_swapped_custom_module_class 接口功能正确性
API 名称：torch.ao.quantization.utils.get_swapped_custom_module_class
API 签名：get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig)

覆盖维度表：
| 覆盖维度      | 说明                                                 | 覆盖情况                                  |
|---------------|------------------------------------------------------|-------------------------------------------|
| 空/非空       | 映射中存在/不存在对应 quant_type 键                  | 已覆盖：test_returns_mapped_class、       |
|               |                                                      | test_mapping_without_quant_type_raises    |
| 枚举选项      | quant_type：STATIC/DYNAMIC 映射键                    | 已覆盖：test_returns_mapped_class、       |
|               |                                                      | test_mapping_without_quant_type_raises    |
| 参数类型      | custom_module 为 nn.Module，mapping 为嵌套 dict      | 已覆盖：test_returns_mapped_class         |
| 传参与不传参  | 该接口无可选参数                                     | 不适用                                    |
| 等价类/边界值 | 映射命中 / 映射缺失（类型或 quant_type 缺失）        | 已覆盖：test_missing_module_type_raises   |
| 正常传参场景  | 返回映射的目标类                                     | 已覆盖：test_returns_mapped_class         |
| 异常传参场景  | 映射缺失或 qconfig 为 None 触发 AssertionError       | 已覆盖：test_missing_module_type_raises、 |
|               |                                                      | test_none_qconfig_raises                  |
| 混合设备类型  | 多 Tensor 异构设备输入                               | 不适用，该接口只做类映射查询，不涉及设备  |

未覆盖项及原因：
- 传参与不传参：接口参数均为必填，无可选参数可覆盖。
- 混合设备类型：纯类映射查询，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 结构工具（框架类 A），无 CPU/NPU 数值输出，按返回类身份与异常断言。
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

from torch.ao.quantization.quant_type import QuantType
from torch.ao.quantization.utils import get_swapped_custom_module_class

_QCONFIG = torch.ao.quantization.default_qconfig


class _FloatCustom(torch.nn.Module):
    def forward(self, x):
        return x


class _ObservedCustom(torch.nn.Module):
    def forward(self, x):
        return x


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetSwappedCustomModuleClassUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.get_swapped_custom_module_class."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_mapped_class(self):
        mapping = {QuantType.STATIC: {_FloatCustom: _ObservedCustom}}
        result = get_swapped_custom_module_class(_FloatCustom(), mapping, _QCONFIG)
        self.assertIs(result, _ObservedCustom)

    def test_missing_module_type_raises(self):
        mapping = {QuantType.STATIC: {_FloatCustom: _ObservedCustom}}
        with self.assertRaises(AssertionError):
            get_swapped_custom_module_class(_ObservedCustom(), mapping, _QCONFIG)

    def test_mapping_without_quant_type_raises(self):
        # No STATIC key in the mapping, so the lookup falls back to an empty dict.
        mapping = {QuantType.DYNAMIC: {_FloatCustom: _ObservedCustom}}
        with self.assertRaises(AssertionError):
            get_swapped_custom_module_class(_FloatCustom(), mapping, _QCONFIG)

    def test_none_qconfig_raises(self):
        mapping = {QuantType.STATIC: {_FloatCustom: _ObservedCustom}}
        with self.assertRaises(AssertionError):
            get_swapped_custom_module_class(_FloatCustom(), mapping, None)


if __name__ == "__main__":
    run_tests()
