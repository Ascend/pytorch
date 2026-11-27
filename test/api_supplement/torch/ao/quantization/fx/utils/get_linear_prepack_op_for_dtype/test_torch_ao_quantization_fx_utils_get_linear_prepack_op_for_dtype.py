# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_linear_prepack_op_for_dtype 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_linear_prepack_op_for_dtype
API 签名：get_linear_prepack_op_for_dtype(dtype)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | dtype 为非空 torch.dtype | 已覆盖：test_float16_maps_to_prepack_fp16 |
| 枚举选项 | float16 / qint8 两个合法分支 | 已覆盖：test_float16_maps_to_prepack_fp16 / test_qint8_maps_to_linear_prepack |
| 参数类型 | 入参为 torch.dtype | 已覆盖：test_returns_callable |
| 传参与不传参 | dtype 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 合法 dtype 与非法 dtype 两类等价类 | 已覆盖：test_invalid_dtype_raises |
| 正常传参场景 | 合法 dtype 返回对应 prepack 算子 | 已覆盖：test_float16_maps_to_prepack_fp16 |
| 异常传参场景 | 非法 dtype 抛 Exception | 已覆盖：test_invalid_dtype_raises |
| 混合设备类型 | 仅按 dtype 分派，无设备输入 | 不适用 |

未覆盖项及原因：
- 空/非空边界：torch.dtype 无「空」取值，None 会落入非法分支（已在异常用例覆盖）。
- 混合设备类型：按 dtype 分派的纯查找工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 分派工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回算子对象同一性与异常断言。
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

from torch.ao.quantization.fx.utils import get_linear_prepack_op_for_dtype


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetLinearPrepackOpForDtypeUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.get_linear_prepack_op_for_dtype."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_float16_maps_to_prepack_fp16(self):
        op = get_linear_prepack_op_for_dtype(torch.float16)
        self.assertIs(op, torch.ops.quantized.linear_prepack_fp16)

    def test_qint8_maps_to_linear_prepack(self):
        op = get_linear_prepack_op_for_dtype(torch.qint8)
        self.assertIs(op, torch.ops.quantized.linear_prepack)

    def test_returns_callable(self):
        self.assertTrue(callable(get_linear_prepack_op_for_dtype(torch.float16)))
        self.assertTrue(callable(get_linear_prepack_op_for_dtype(torch.qint8)))

    def test_distinct_ops_for_distinct_dtypes(self):
        fp16_op = get_linear_prepack_op_for_dtype(torch.float16)
        qint8_op = get_linear_prepack_op_for_dtype(torch.qint8)
        self.assertIsNot(fp16_op, qint8_op)

    def test_invalid_dtype_raises(self):
        with self.assertRaises(Exception):
            get_linear_prepack_op_for_dtype(torch.quint8)

    def test_invalid_dtype_message(self):
        with self.assertRaisesRegex(Exception, "can't get linear prepack op"):
            get_linear_prepack_op_for_dtype(torch.int32)


if __name__ == "__main__":
    run_tests()
