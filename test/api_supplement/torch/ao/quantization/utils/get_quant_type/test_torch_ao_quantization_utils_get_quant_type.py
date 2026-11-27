# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.get_quant_type 接口功能正确性
API 名称：torch.ao.quantization.utils.get_quant_type
API 签名：get_quant_type(qconfig) -> QuantType

覆盖维度表：
| 覆盖维度      | 说明                                      | 覆盖情况                                             |
|---------------|-------------------------------------------|------------------------------------------------------|
| 空/非空       | qconfig 为非空对象 / None                 | 已覆盖：test_static_qconfig、                        |
|               |                                           | test_none_qconfig_raises                             |
| 枚举选项      | 返回 QuantType.STATIC/DYNAMIC/WEIGHT_ONLY | 已覆盖：test_static_qconfig、test_dynamic_qconfig、  |
|               |                                           | test_weight_only_qconfig                             |
| 参数类型      | qconfig 为 QConfig 实例                   | 已覆盖：test_static_qconfig                          |
| 传参与不传参  | 该接口无可选参数                          | 不适用                                               |
| 等价类/边界值 | float16 静态/动态两种等价边界             | 已覆盖：test_float16_static、test_float16_dynamic    |
| 正常传参场景  | 对已知 dtype 组合返回对应 QuantType       | 已覆盖：test_static_qconfig、test_dynamic_qconfig    |
| 异常传参场景  | 非法组合抛 Exception；qconfig 为 None     | 已覆盖：test_unrecognized_dtype_combination_raises、 |
|               | 抛断言                                    |                                                      |
|               |                                           | test_none_qconfig_raises                             |
| 混合设备类型  | 多 Tensor 异构设备输入                    | 不适用，该接口仅依据 qconfig 元信息分类，不涉及设备  |

未覆盖项及原因：
- 传参与不传参：接口仅一个必填参数，无可选参数可覆盖。
- 混合设备类型：分类只读取 qconfig 精度信息，不消费 Tensor，无设备执行路径，故不覆盖。

注：该 API 为纯 Python 分类工具（框架类 A），无 CPU/NPU 数值输出，按 QuantType 枚举断言。
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

from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MinMaxObserver, PlaceholderObserver
from torch.ao.quantization.quant_type import QuantType
from torch.ao.quantization.utils import get_quant_type


def _qconfig(activation_dtype, weight_dtype=torch.qint8, dynamic=False):
    if dynamic:
        activation = PlaceholderObserver.with_args(dtype=activation_dtype, is_dynamic=True)
    elif activation_dtype in (torch.float16, torch.float32):
        activation = PlaceholderObserver.with_args(dtype=activation_dtype)
    else:
        activation = MinMaxObserver.with_args(dtype=activation_dtype)
    if weight_dtype in (torch.float16, torch.float32):
        weight = PlaceholderObserver.with_args(dtype=weight_dtype)
    else:
        weight = MinMaxObserver.with_args(dtype=weight_dtype)
    return QConfig(activation=activation, weight=weight)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetQuantTypeUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.get_quant_type."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_static_qconfig(self):
        self.assertEqual(get_quant_type(_qconfig(torch.quint8, torch.qint8)), QuantType.STATIC)

    def test_dynamic_qconfig(self):
        self.assertEqual(
            get_quant_type(_qconfig(torch.quint8, torch.qint8, dynamic=True)), QuantType.DYNAMIC
        )

    def test_weight_only_qconfig(self):
        self.assertEqual(get_quant_type(_qconfig(torch.float32, torch.qint8)), QuantType.WEIGHT_ONLY)

    def test_float16_static(self):
        self.assertEqual(get_quant_type(_qconfig(torch.float16, torch.float16)), QuantType.STATIC)

    def test_float16_dynamic(self):
        self.assertEqual(
            get_quant_type(_qconfig(torch.float16, torch.float16, dynamic=True)), QuantType.DYNAMIC
        )

    def test_unrecognized_dtype_combination_raises(self):
        qconfig = _qconfig(torch.quint8, torch.float32)
        with self.assertRaises(Exception) as ctx:
            get_quant_type(qconfig)
        # The implementation raises a plain Exception, not an AssertionError.
        self.assertNotIsInstance(ctx.exception, AssertionError)

    def test_none_qconfig_raises(self):
        with self.assertRaises(AssertionError):
            get_quant_type(None)


if __name__ == "__main__":
    run_tests()
