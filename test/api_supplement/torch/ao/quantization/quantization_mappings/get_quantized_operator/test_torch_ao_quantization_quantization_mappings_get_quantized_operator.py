# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantization_mappings.get_quantized_operator 接口功能正确性
API 名称：torch.ao.quantization.quantization_mappings.get_quantized_operator
API 签名：get_quantized_operator(float_op: Callable | str) -> Callable

覆盖维度表：
| 覆盖维度      | 说明                                     | 覆盖情况                                              |
|---------------|------------------------------------------|-------------------------------------------------------|
| 空/非空       | 传入已注册浮点算子(非空命中)与未注册算子 | 已覆盖：test_known_operator_returns_quantized_op /    |
|               |                                          | test_unknown_operator_raises                          |
| 枚举选项      | 该接口无枚举参数                         | 不适用                                                |
| 参数类型      | float_op 支持 Callable 与 str；Callable  | 已覆盖：test_known_operator_returns_quantized_op /    |
|               | 命中，未注册 str 抛错                    | test_unregistered_string_raises                       |
| 传参与不传参  | 该接口无默认参数，float_op 必传          | 不适用                                                |
| 等价类/边界值 | 全部默认映射键逐项命中 / 未命中键        | 已覆盖：test_all_default_keys_map_consistently /      |
|               |                                          | test_unknown_operator_raises                          |
| 正常传参场景  | 返回映射中登记的量化算子且与映射值同一   | 已覆盖：test_returns_exact_mapped_operator            |
| 异常传参场景  | 未注册算子抛 AssertionError              | 已覆盖：test_unknown_operator_raises /                |
|               |                                          | test_unknown_operator_message                         |
| 混合设备类型  | 纯 Python 映射查询，无设备执行路径       | 不适用                                                |

未覆盖项及原因：
- 混合设备类型：该接口仅查询算子映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 映射工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回算子对象与异常契约做断言。
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
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
    get_quantized_operator,
)

_ERROR_RE = "does not have corresponding quantized op"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetQuantizedOperatorUt(TestCase):
    """Functional tests for quantization_mappings.get_quantized_operator."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_known_operator_returns_quantized_op(self):
        """Registered float operators map to their quantized counterparts."""
        self.assertIs(get_quantized_operator(F.elu), torch.ops.quantized.elu)
        self.assertIs(get_quantized_operator(F.hardswish), torch.ops.quantized.hardswish)
        self.assertIs(get_quantized_operator(F.leaky_relu), torch.ops.quantized.leaky_relu)
        self.assertIs(get_quantized_operator(F.layer_norm), torch.ops.quantized.layer_norm)

    def test_returned_object_is_callable(self):
        """The returned quantized operator is callable."""
        result = get_quantized_operator(F.elu)
        self.assertTrue(callable(result))

    def test_returns_exact_mapped_operator(self):
        """The return value is exactly the object stored in the default mapping."""
        for key, value in DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.items():
            self.assertIs(get_quantized_operator(key), value)

    def test_all_default_keys_map_consistently(self):
        """Every default mapping key resolves to a callable."""
        for key in DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS:
            self.assertTrue(callable(get_quantized_operator(key)))

    def test_unknown_operator_raises(self):
        """An unregistered float operator raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_quantized_operator(F.relu)

    def test_unregistered_string_raises(self):
        """An unregistered string operator name raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_quantized_operator("torch.quantized.not_registered")

    def test_unknown_operator_message(self):
        """The AssertionError message names the missing quantized op."""
        with self.assertRaisesRegex(AssertionError, _ERROR_RE):
            get_quantized_operator(F.relu)


if __name__ == "__main__":
    run_tests()
