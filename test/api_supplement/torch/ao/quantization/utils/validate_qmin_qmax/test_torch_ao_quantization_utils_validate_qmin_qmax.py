# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.validate_qmin_qmax 接口功能正确性
API 名称：torch.ao.quantization.utils.validate_qmin_qmax
API 签名：validate_qmin_qmax(quant_min, quant_max) -> None

覆盖维度表：
| 覆盖维度      | 说明                                         | 覆盖情况                                          |
|---------------|----------------------------------------------|---------------------------------------------------|
| 空/非空       | 区间包含 0 / 不包含 0                        | 已覆盖：test_valid_signed_range、                 |
|               |                                              | test_range_not_including_zero_raises              |
| 枚举选项      | 有符号 qint8 区间 / 无符号 quint8 区间       | 已覆盖：test_valid_signed_range、                 |
|               |                                              | test_valid_unsigned_range                         |
| 参数类型      | quant_min/quant_max 为 int                   | 已覆盖：test_valid_signed_range                   |
| 传参与不传参  | 该接口无可选参数                             | 不适用                                            |
| 等价类/边界值 | 包含 0 的合法区间 / 不含 0 或 qmin>=qmax     | 已覆盖：test_qmin_equal_qmax_raises               |
|               | 的非法区间                                   |                                                   |
| 正常传参场景  | 合法区间返回 None                            | 已覆盖：test_returns_none                         |
| 异常传参场景  | 区间不含 0 或 qmin>=qmax 触发 AssertionError | 已覆盖：test_range_not_including_zero_raises、    |
|               |                                              | test_qmin_equal_qmax_raises                       |
| 混合设备类型  | 多 Tensor 异构设备输入                       | 不适用，该接口仅校验整数区间，不涉及设备          |

未覆盖项及原因：
- 传参与不传参：接口仅两个必填参数，无可选参数可覆盖。
- 混合设备类型：纯整数区间校验，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 校验工具（框架类 A），无 CPU/NPU 数值输出，按返回值与异常断言。
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

from torch.ao.quantization.utils import validate_qmin_qmax


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestValidateQminQmaxUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.validate_qmin_qmax."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_valid_signed_range(self):
        self.assertIsNone(validate_qmin_qmax(-128, 127))

    def test_valid_unsigned_range(self):
        self.assertIsNone(validate_qmin_qmax(0, 255))

    def test_returns_none(self):
        self.assertIsNone(validate_qmin_qmax(-4, 3))

    def test_range_not_including_zero_raises(self):
        with self.assertRaises(AssertionError):
            validate_qmin_qmax(1, 10)

    def test_qmin_equal_qmax_raises(self):
        # Includes 0 but violates qmin < qmax.
        with self.assertRaises(AssertionError):
            validate_qmin_qmax(0, 0)


if __name__ == "__main__":
    run_tests()
