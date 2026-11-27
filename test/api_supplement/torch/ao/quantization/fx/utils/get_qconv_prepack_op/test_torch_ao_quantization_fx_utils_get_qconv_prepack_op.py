# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_qconv_prepack_op 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_qconv_prepack_op
API 签名：get_qconv_prepack_op(conv_op) -> Callable

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | conv_op 为受支持函数与不受支持函数 | 已覆盖：test_conv2d_maps_to_prepack / test_unknown_op_raises |
| 枚举选项 | conv1d/2d/3d 及 conv_transpose1d/2d/3d | 已覆盖：test_all_supported_conv_ops_mapped |
| 参数类型 | 入参为 Callable | 已覆盖：test_returns_callable |
| 传参与不传参 | conv_op 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 受支持算子映射与未命中报错两类 | 已覆盖：test_conv2d_maps_to_prepack / test_unknown_op_raises |
| 正常传参场景 | 各卷积函数映射到对应 prepack 算子 | 已覆盖：test_all_supported_conv_ops_mapped |
| 异常传参场景 | 未命中时抛 AssertionError | 已覆盖：test_unknown_op_raises |
| 混合设备类型 | 按函数查找，无设备输入 | 不适用 |

未覆盖项及原因：
- 空/非空边界：conv_op 为可调用对象，无「空」取值；None 落入未命中分支（异常用例覆盖）。
- 混合设备类型：按函数名查找的纯分派工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

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

import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_qconv_prepack_op


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetQconvPrepackOpUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.get_qconv_prepack_op."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_conv2d_maps_to_prepack(self):
        op = get_qconv_prepack_op(F.conv2d)
        self.assertIs(op, torch.ops.quantized.conv2d_prepack)

    def test_conv1d_maps_to_prepack(self):
        self.assertIs(get_qconv_prepack_op(F.conv1d), torch.ops.quantized.conv1d_prepack)

    def test_conv_transpose2d_maps_to_prepack(self):
        op = get_qconv_prepack_op(F.conv_transpose2d)
        self.assertIs(op, torch.ops.quantized.conv_transpose2d_prepack)

    def test_all_supported_conv_ops_mapped(self):
        expected = {
            F.conv1d: torch.ops.quantized.conv1d_prepack,
            F.conv2d: torch.ops.quantized.conv2d_prepack,
            F.conv3d: torch.ops.quantized.conv3d_prepack,
            F.conv_transpose1d: torch.ops.quantized.conv_transpose1d_prepack,
            F.conv_transpose2d: torch.ops.quantized.conv_transpose2d_prepack,
            F.conv_transpose3d: torch.ops.quantized.conv_transpose3d_prepack,
        }
        for conv_op, prepack_op in expected.items():
            self.assertIs(get_qconv_prepack_op(conv_op), prepack_op)

    def test_returns_callable(self):
        self.assertTrue(callable(get_qconv_prepack_op(F.conv2d)))

    def test_unknown_op_raises(self):
        with self.assertRaises(AssertionError):
            get_qconv_prepack_op(F.max_pool2d)

    def test_unknown_op_message(self):
        with self.assertRaisesRegex(AssertionError, "Didn't find prepack op"):
            get_qconv_prepack_op(torch.relu)


if __name__ == "__main__":
    run_tests()
