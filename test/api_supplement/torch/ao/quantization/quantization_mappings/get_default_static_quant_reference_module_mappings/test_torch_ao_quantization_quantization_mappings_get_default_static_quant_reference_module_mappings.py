# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantization_mappings.get_default_static_quant_reference_module_mappings 接口功能正确性
API 名称：torch.ao.quantization.quantization_mappings.get_default_static_quant_reference_module_mappings
API 签名：get_default_static_quant_reference_module_mappings() -> dict[Callable, Any]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无入参，返回非空映射 | 已覆盖：test_returns_non_empty_dict |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | 无入参；返回键为模块/存根类、值为参考量化模块类 | 已覆盖：test_all_keys_and_values_are_classes |
| 传参与不传参 | 该接口无参数 | 不适用 |
| 等价类/边界值 | 普通模块键、存根类键、RNN 家族键 | 已覆盖：test_maps_conv_linear_to_reference / test_maps_stubs_to_quantize |
| 正常传参场景 | 返回浮点模块类到参考量化模块类的完整默认映射 | 已覆盖：test_key_set_matches_default_mapping |
| 异常传参场景 | 该接口无入参，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_default_static_quant_reference_module_mappings 不接收参数，内部仅做深拷贝，
  无调用方可构造的异常路径，故不单列。
- 混合设备类型：该接口仅构造模块类映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 映射工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按键集合、值类型与深拷贝独立性做断言。
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
import torch.ao.nn.quantized.reference as nnqr
import torch.nn
from torch.ao.quantization.quantization_mappings import (
    get_default_static_quant_reference_module_mappings,
)
from torch.ao.quantization.stubs import DeQuantStub, QuantStub


def _expected_key_set():
    return {
        QuantStub,
        DeQuantStub,
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Embedding,
        torch.nn.EmbeddingBag,
        torch.nn.GRUCell,
        torch.nn.LSTMCell,
        torch.nn.RNNCell,
        torch.nn.LSTM,
    }


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetDefaultStaticQuantReferenceModuleMappingsUt(TestCase):
    """Functional tests for quantization_mappings.get_default_static_quant_reference_module_mappings."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_non_empty_dict(self):
        """The interface returns a non-empty dict."""
        result = get_default_static_quant_reference_module_mappings()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_all_keys_and_values_are_classes(self):
        """Every key and value is a class object."""
        for key, value in get_default_static_quant_reference_module_mappings().items():
            self.assertIsInstance(key, type)
            self.assertIsInstance(value, type)

    def test_maps_conv_linear_to_reference(self):
        """Plain conv/linear modules map to their reference quantized modules."""
        result = get_default_static_quant_reference_module_mappings()
        self.assertIs(result[torch.nn.Linear], nnqr.Linear)
        self.assertIs(result[torch.nn.Conv2d], nnqr.Conv2d)
        self.assertIs(result[torch.nn.LSTM], nnqr.LSTM)

    def test_maps_stubs_to_quantize(self):
        """QuantStub/DeQuantStub map to nnq.Quantize/nnq.DeQuantize."""
        result = get_default_static_quant_reference_module_mappings()
        self.assertIs(result[QuantStub], nnq.Quantize)
        self.assertIs(result[DeQuantStub], nnq.DeQuantize)

    def test_key_set_matches_default_mapping(self):
        """The key set matches the documented default reference mapping exactly."""
        result = get_default_static_quant_reference_module_mappings()
        self.assertEqual(set(result.keys()), _expected_key_set())

    def test_returns_deepcopy_not_shared(self):
        """Each call returns a fresh dict; mutation does not leak to the module constant."""
        qm = torch.ao.quantization.quantization_mappings
        first = get_default_static_quant_reference_module_mappings()
        second = get_default_static_quant_reference_module_mappings()
        self.assertIsNot(first, second)
        first[torch.nn.Identity] = None
        self.assertNotIn(torch.nn.Identity, second)
        self.assertNotIn(torch.nn.Identity, qm.DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)


if __name__ == "__main__":
    run_tests()
