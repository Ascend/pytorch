# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.calculate_qmin_qmax 接口功能正确性
API 名称：torch.ao.quantization.utils.calculate_qmin_qmax
API 签名：calculate_qmin_qmax(quant_min, quant_max, has_customized_qrange, dtype, reduce_range)

覆盖维度表：
| 覆盖维度      | 说明                                     | 覆盖情况                                              |
|---------------|------------------------------------------|-------------------------------------------------------|
| 空/非空       | 自定义区间长度为正 / 非正                | 已覆盖：test_custom_qrange_used、                     |
|               |                                          | test_custom_qrange_non_positive_len_raises            |
| 枚举选项      | dtype：qint8/int8/quint8/uint8/qint32/   | 已覆盖：test_qint8_default、test_quint8_default、     |
|               | uint16/int16                             |                                                       |
|               | / 其他（float32）                        | test_qint32_default、test_uint16_default、            |
|               |                                          | test_int16_default、test_unsupported_dtype_default    |
| 参数类型      | quant_min/quant_max 为 int，标志为 bool  | 已覆盖：test_qint8_default                            |
| 传参与不传参  | has_customized_qrange / reduce_range     | 已覆盖：test_qint8_reduce、test_custom_qrange_reduce  |
|               | 布尔开关                                 |                                                       |
| 等价类/边界值 | 自定义区间长度恰为 256 的上界            | 已覆盖：test_custom_qrange_length_256_ok              |
| 正常传参场景  | 返回 (quant_min, quant_max) 整数二元组   | 已覆盖：test_qint8_default、test_quint8_reduce        |
| 异常传参场景  | 自定义区间长度非正 / 超过位宽触发        | 已覆盖：test_custom_qrange_exceeds_bit_range_raises、 |
|               | AssertionError                           |                                                       |
|               |                                          | test_custom_qrange_non_positive_len_raises            |
| 混合设备类型  | 多 Tensor 异构设备输入                   | 不适用，该接口仅做整数区间运算，不涉及设备            |

未覆盖项及原因：
- 传参与不传参：接口参数均为必填，无可选参数可覆盖。
- 混合设备类型：接口只处理整数标量，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 整数工具（框架类 A），无 CPU/NPU 数值输出，整数结果精确断言。
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

from torch.ao.quantization.utils import calculate_qmin_qmax


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCalculateQminQmaxUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.calculate_qmin_qmax."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_qint8_default(self):
        qmin, qmax = calculate_qmin_qmax(0, 255, False, torch.qint8, False)
        self.assertEqual((qmin, qmax), (-128, 127))

    def test_qint8_reduce(self):
        qmin, qmax = calculate_qmin_qmax(0, 255, False, torch.qint8, True)
        self.assertEqual((qmin, qmax), (-64, 63))

    def test_quint8_default(self):
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.quint8, False), (0, 255))

    def test_quint8_reduce(self):
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.quint8, True), (0, 127))

    def test_int8_same_as_qint8(self):
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.int8, False), (-128, 127))

    def test_qint32_default(self):
        self.assertEqual(
            calculate_qmin_qmax(0, 255, False, torch.qint32, False),
            (-(2 ** 31), 2 ** 31 - 1),
        )

    def test_uint16_default(self):
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.uint16, False), (0, 2 ** 16 - 1))

    def test_int16_default(self):
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.int16, False), (-(2 ** 15), 2 ** 15 - 1))

    def test_unsupported_dtype_default(self):
        # Unknown dtypes fall back to a 4-bit range.
        self.assertEqual(calculate_qmin_qmax(0, 255, False, torch.float32, False), (0, 15))

    def test_custom_qrange_used(self):
        self.assertEqual(calculate_qmin_qmax(2, 10, True, torch.qint8, False), (2, 10))

    def test_custom_qrange_reduce(self):
        self.assertEqual(calculate_qmin_qmax(2, 10, True, torch.qint8, True), (1, 5))

    def test_custom_qrange_length_256_ok(self):
        # Length == 256 is the inclusive upper bound and must be accepted.
        self.assertEqual(calculate_qmin_qmax(0, 255, True, torch.qint8, False), (0, 255))

    def test_custom_qrange_exceeds_bit_range_raises(self):
        with self.assertRaises(AssertionError):
            calculate_qmin_qmax(0, 256, True, torch.qint8, False)

    def test_custom_qrange_non_positive_len_raises(self):
        with self.assertRaises(AssertionError):
            calculate_qmin_qmax(10, 5, True, torch.qint8, False)

    def test_return_type_is_int_tuple(self):
        qmin, qmax = calculate_qmin_qmax(0, 255, False, torch.quint8, False)
        self.assertIsInstance(qmin, int)
        self.assertIsInstance(qmax, int)


if __name__ == "__main__":
    run_tests()
