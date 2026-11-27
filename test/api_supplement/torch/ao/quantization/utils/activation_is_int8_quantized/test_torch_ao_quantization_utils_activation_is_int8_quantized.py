# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.activation_is_int8_quantized 接口功能正确性
API 名称：torch.ao.quantization.utils.activation_is_int8_quantized
API 签名：activation_is_int8_quantized(qconfig)

覆盖维度表：
| 覆盖维度      | 说明                                       | 覆盖情况                                            |
|---------------|--------------------------------------------|-----------------------------------------------------|
| 空/非空       | qconfig 为非空对象 / None                  | 已覆盖：test_quint8_true、test_none_qconfig_raises  |
| 枚举选项      | 激活 dtype：quint8/qint8/uint8/int8/       | 已覆盖：test_quint8_true、test_qint8_true、         |
|               | qint32/float16                             | test_uint8_true、                                   |
|               |                                            | test_int8_true、test_qint32_false、                 |
|               |                                            | test_float16_false                                  |
| 参数类型      | qconfig 为 QConfig 实例                    | 已覆盖：test_quint8_true                            |
| 传参与不传参  | 该接口无可选参数                           | 不适用                                              |
| 等价类/边界值 | 带/不带 zero-point 的 int8 表达（q/u       | 已覆盖：test_quint8_true、test_qint8_true、         |
|               | 前缀）                                     | test_uint8_true                                     |
| 正常传参场景  | 对 int8 激活 qconfig 返回 True             | 已覆盖：test_quint8_true、test_qint8_true           |
| 异常传参场景  | qconfig 为 None 触发 AssertionError        | 已覆盖：test_none_qconfig_raises                    |
| 混合设备类型  | 多 Tensor 异构设备输入                     | 不适用，该接口仅读取 qconfig 精度元信息，不涉及设备 |

未覆盖项及原因：
- 传参与不传参：接口仅一个必填参数，无可选参数可覆盖。
- 混合设备类型：谓词只检查激活 dtype，不消费 Tensor，无设备执行路径，故不覆盖。

注：该 API 为纯 Python 谓词（框架类 A），无 CPU/NPU 数值输出，不做 CPU Golden 对照。
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
from torch.ao.quantization.utils import activation_is_int8_quantized


def _qconfig(activation_dtype, weight_dtype=torch.qint8):
    if activation_dtype in (torch.float16, torch.float32):
        activation = PlaceholderObserver.with_args(dtype=activation_dtype)
    else:
        activation = MinMaxObserver.with_args(dtype=activation_dtype)
    if weight_dtype in (torch.float16, torch.float32):
        weight = PlaceholderObserver.with_args(dtype=weight_dtype)
    else:
        weight = MinMaxObserver.with_args(dtype=weight_dtype)
    return QConfig(activation=activation, weight=weight)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestActivationIsInt8QuantizedUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.activation_is_int8_quantized."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_quint8_true(self):
        self.assertTrue(activation_is_int8_quantized(_qconfig(torch.quint8)))

    def test_qint8_true(self):
        self.assertTrue(activation_is_int8_quantized(_qconfig(torch.qint8)))

    def test_uint8_true(self):
        self.assertTrue(activation_is_int8_quantized(_qconfig(torch.uint8)))

    def test_int8_true(self):
        self.assertTrue(activation_is_int8_quantized(_qconfig(torch.int8)))

    def test_qint32_false(self):
        self.assertFalse(activation_is_int8_quantized(_qconfig(torch.qint32, torch.qint32)))

    def test_float16_false(self):
        self.assertFalse(activation_is_int8_quantized(_qconfig(torch.float16, torch.float16)))

    def test_none_qconfig_raises(self):
        with self.assertRaises(AssertionError):
            activation_is_int8_quantized(None)


if __name__ == "__main__":
    run_tests()
