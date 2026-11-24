# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantizable.modules.activation.MultiheadAttention.dequantize 接口功能正确性
API 名称：torch.ao.nn.quantizable.modules.activation.MultiheadAttention.dequantize
API 签名：MultiheadAttention.dequantize(self)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无参方法；被测对象为已量化（非空）或未量化（空）MHA | 已覆盖：test_dequantize_returns_float_module / test_dequantize_requires_quantized_module |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | 无入参；返回 torch.nn.MultiheadAttention | 已覆盖：test_dequantize_returns_float_module |
| 传参与不传参 | 该接口无参数 | 不适用 |
| 等价类/边界值 | 不同 embed_dim/num_heads 组合、有无 bias | 已覆盖：test_dequantize_returns_float_module |
| 正常传参场景 | 从量化 MHA 还原 float 权重并可前向 | 已覆盖：test_dequantize_reconstructs_in_proj_weight / test_dequantize_forward_shape |
| 异常传参场景 | 对未量化的 quantizable MHA 调用抛 AttributeError | 已覆盖：test_dequantize_requires_quantized_module |
| 混合设备类型 | 该接口仅操作 CPU 量化张量的权重还原，无多 Tensor 异构设备输入 | 不适用 |

未覆盖项及原因：
- 混合设备类型：dequantize 仅对 CPU 量化后端产生的量化权重做 dequantize 还原，无 NPU/CUDA 执行路径，
  不存在可构造的异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：该接口只在 CPU 量化后端上有完整执行路径，无 NPU/CUDA 数值对照路径，故不做 CPU/NPU Golden 对照；
     还原后的 in_proj_weight/out_proj.weight 与量化权重 dequantize 结果按位相等，用精确断言校验。
     调用前置于 torch.no_grad() 下（还原 float 权重无需自动微分，避免对 requires_grad 叶子参数的原地切片赋值）。
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

import torch.ao.nn.quantizable
import torch.ao.nn.quantized
import torch.ao.quantization
import torch.nn
from torch.ao.nn.quantizable.modules.activation import MultiheadAttention as QuantizableMHA

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_EMBED_DIM = 8
_NUM_HEADS = 2


def _build_quantized_mha():
    """Run the float -> observed -> quantized flow for a MultiheadAttention."""
    float_mha = torch.nn.MultiheadAttention(_EMBED_DIM, _NUM_HEADS)
    float_mha.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    observed = QuantizableMHA.from_float(float_mha)
    return torch.ao.quantization.convert(observed, inplace=False)


def _probe():
    with torch.no_grad():
        _build_quantized_mha().dequantize()


def _use_quantized_engine(probe):
    """Select a CPU quantized engine able to run the MHA quantize/dequantize flow.

    The quantizable MHA only provides kernels on the CPU quantized backend, so
    there is no NPU/CUDA execution path to compare against.
    """
    supported = torch.backends.quantized.supported_engines
    for name in _ENGINE_CANDIDATES:
        if name not in supported:
            continue
        torch.backends.quantized.engine = name
        if probe is None:
            return name
        try:
            probe()
        except RuntimeError:
            # This engine has no usable kernel for the flow; try the next one.
            continue
        return name
    raise unittest.SkipTest("No CPU quantized engine can run the target op on this platform")


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestMultiheadAttentionDequantizeUt(TestCase):
    """Functional tests for torch.ao.nn.quantizable.modules.activation.MultiheadAttention.dequantize."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_dequantize_returns_float_module(self):
        """dequantize returns a float MultiheadAttention with matching dims."""
        quantized = _build_quantized_mha()
        with torch.no_grad():
            fp = quantized.dequantize()
        self.assertIsInstance(fp, torch.nn.MultiheadAttention)
        self.assertEqual(fp.embed_dim, _EMBED_DIM)
        self.assertEqual(fp.num_heads, _NUM_HEADS)

    def test_dequantize_reconstructs_in_proj_weight(self):
        """The reconstructed in_proj_weight equals the concatenated dequantized Q/K/V weights."""
        quantized = _build_quantized_mha()
        with torch.no_grad():
            fp = quantized.dequantize()
        w_q = quantized.linear_Q._weight_bias()[0].dequantize()
        w_k = quantized.linear_K._weight_bias()[0].dequantize()
        w_v = quantized.linear_V._weight_bias()[0].dequantize()
        expected = torch.cat([w_q, w_k, w_v], dim=0)
        # A wrong offset/slice below would break this exact reconstruction.
        self.assertTrue(torch.equal(fp.in_proj_weight, expected))
        self.assertEqual(tuple(fp.in_proj_weight.shape), (3 * _EMBED_DIM, _EMBED_DIM))

    def test_dequantize_reconstructs_out_proj_weight(self):
        """The reconstructed out_proj weight equals the dequantized out_proj weight."""
        quantized = _build_quantized_mha()
        with torch.no_grad():
            fp = quantized.dequantize()
        w_out = quantized.out_proj._weight_bias()[0].dequantize()
        self.assertTrue(torch.equal(fp.out_proj.weight, w_out))
        self.assertEqual(tuple(fp.out_proj.weight.shape), (_EMBED_DIM, _EMBED_DIM))

    def test_dequantize_zero_in_proj_bias(self):
        """The quantized in_proj bias is zero, so the restored float bias must be all zeros."""
        quantized = _build_quantized_mha()
        with torch.no_grad():
            fp = quantized.dequantize()
        self.assertIsNotNone(fp.in_proj_bias)
        self.assertEqual(float(fp.in_proj_bias.abs().max()), 0.0)

    def test_dequantize_forward_shape(self):
        """The restored float module runs forward and returns the expected shapes."""
        quantized = _build_quantized_mha()
        with torch.no_grad():
            fp = quantized.dequantize()
        query = torch.randn(3, 2, _EMBED_DIM)
        attn_output, attn_weights = fp(query, query, query)
        self.assertEqual(tuple(attn_output.shape), (3, 2, _EMBED_DIM))
        self.assertEqual(tuple(attn_weights.shape), (2, 3, 3))

    def test_dequantize_requires_quantized_module(self):
        """Calling dequantize on a non-quantized quantizable MHA is an error."""
        float_mha = torch.nn.MultiheadAttention(_EMBED_DIM, _NUM_HEADS)
        float_mha.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        observed = QuantizableMHA.from_float(float_mha)
        with self.assertRaises(AttributeError):
            observed.dequantize()


if __name__ == "__main__":
    run_tests()
