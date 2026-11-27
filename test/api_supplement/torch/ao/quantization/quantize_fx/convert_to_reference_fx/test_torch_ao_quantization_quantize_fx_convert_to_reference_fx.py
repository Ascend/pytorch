# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantize_fx.convert_to_reference_fx 接口功能正确性
API 名称：torch.ao.quantization.quantize_fx.convert_to_reference_fx
API 签名：convert_to_reference_fx(graph_module, convert_custom_config=None,
          _remove_qconfig=True, qconfig_mapping=None, backend_config=None)

覆盖维度表：
| 覆盖维度      | 说明                                       | 覆盖情况                                            |
|---------------|--------------------------------------------|-----------------------------------------------------|
| 空/非空       | convert_custom_config 为 None / 非空       | 已覆盖：test_convert_returns_graph_module、         |
|               |                                            | test_convert_preserves_attributes                   |
| 枚举选项      | _remove_qconfig 布尔开关                   | 已覆盖：test_convert_removes_qconfig、              |
|               |                                            | test_convert_keeps_qconfig_when_requested           |
| 参数类型      | graph_module 为 GraphModule                | 已覆盖：test_convert_rejects_non_graph_module       |
| 传参与不传参  | 省略可选参数 / 显式传入                    | 已覆盖：test_convert_returns_graph_module（省略）、 |
|               |                                            | test_convert_preserves_attributes（显式）           |
| 等价类/边界值 | 融合子模块 + 参考算子图结构                | 已覆盖：test_convert_node_set_has_reference_ops     |
| 正常传参场景  | 输出参考量化 GraphModule 且前向为浮点      | 已覆盖：test_convert_produces_float_output          |
| 异常传参场景  | 传入非 GraphModule 触发 ValueError         | 已覆盖：test_convert_rejects_non_graph_module       |
| 混合设备类型  | 多 Tensor 异构设备输入                     | 未覆盖，见下                                        |

未覆盖项及原因：
- 混合设备类型：参考量化仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供，无 NPU/CUDA 执行路径，
  无法构造异构设备输入，故不覆盖。
- 图模式：本用例针对 eager FX 图变换，未覆盖 torch.compile / NPUGraph 图模式路径。

注：参考量化模型输出为反量化后的浮点 Tensor，但无 NPU/CUDA 参考路径，
     故不做 CPU/NPU Golden 对照，仅校验 FX 图变换后的结构、dtype 与属性。
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

import torch.ao.quantization
import torch.ao.quantization.quantize_fx
from torch.ao.quantization.fx.custom_config import ConvertCustomConfig
from torch.fx import GraphModule

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_EXAMPLE = (torch.randn(1, 3, 8, 8),)


def _backend_engine():
    """Return a supported CPU quantized engine name, or skip when none exists."""
    supported = torch.backends.quantized.supported_engines
    for name in _ENGINE_CANDIDATES:
        if name in supported:
            return name
    raise unittest.SkipTest("No CPU quantized engine is available on this platform")


def _build_float_net():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    return _Net()


def _prepare_calibrated():
    engine = _backend_engine()
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(engine)
    prepared = torch.ao.quantization.quantize_fx.prepare_fx(
        _build_float_net().eval(), qconfig_mapping, _EXAMPLE
    )
    prepared(*_EXAMPLE)
    return prepared


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestConvertToReferenceFxUt(TestCase):
    """Functional tests for torch.ao.quantization.quantize_fx.convert_to_reference_fx."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_convert_returns_graph_module(self):
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(_prepare_calibrated())
        self.assertIsInstance(ref, GraphModule)
        self.assertEqual(ref.meta.get("_quantized", False), False)

    def test_convert_node_set_has_reference_ops(self):
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(_prepare_calibrated())
        nodes = [(n.op, str(n.target)) for n in ref.graph.nodes]
        ops = {op for op, _ in nodes}
        self.assertIn("get_attr", ops)
        self.assertTrue(any(op == "call_method" and t == "dequantize" for op, t in nodes))
        self.assertTrue(any(op == "call_function" and "quantize_per_tensor" in t for op, t in nodes))

    def test_convert_produces_float_output(self):
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(_prepare_calibrated())
        out = ref(*_EXAMPLE)
        # The reference model dequantizes back to float.
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (1, 4, 6, 6))

    def test_convert_removes_qconfig(self):
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(_prepare_calibrated())
        self.assertIsNone(getattr(ref.conv, "qconfig", None))

    def test_convert_keeps_qconfig_when_requested(self):
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(
            _prepare_calibrated(), _remove_qconfig=False
        )
        self.assertIsNotNone(getattr(ref.conv, "qconfig", None))

    def test_convert_preserves_attributes(self):
        prepared = _prepare_calibrated()
        prepared.my_attr = [7, 8]
        config = ConvertCustomConfig().set_preserved_attributes(["my_attr"])
        ref = torch.ao.quantization.quantize_fx.convert_to_reference_fx(
            prepared, convert_custom_config=config
        )
        self.assertEqual(ref.my_attr, [7, 8])

    def test_convert_rejects_non_graph_module(self):
        with self.assertRaises(ValueError):
            torch.ao.quantization.quantize_fx.convert_to_reference_fx(_build_float_net().eval())


if __name__ == "__main__":
    run_tests()
