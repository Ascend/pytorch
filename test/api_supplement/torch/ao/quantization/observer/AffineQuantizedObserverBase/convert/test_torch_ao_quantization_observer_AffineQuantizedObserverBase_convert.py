# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.AffineQuantizedObserverBase.convert 接口功能正确性
API 名称：torch.ao.quantization.observer.AffineQuantizedObserverBase.convert
API 签名：convert(self, model, observer_node)

覆盖维度表：
| 覆盖维度      | 说明                           | 覆盖情况                                                        |
|---------------|--------------------------------|-----------------------------------------------------------------|
| 空/非空       | block_size/original_dtype      | 已覆盖：test_convert_raises_without_block_size /                |
|               | 未填充（空）与已填充（非空）   | test_convert_replaces_observer_with_quantize_dequantize         |
| 枚举选项      | mapping_type/                  | 已覆盖：test_convert_replaces_observer_with_quantize_dequantize |
|               | zero_point_domain              |                                                                 |
|               | 枚举参与转换（ASYMMETRIC/INT） |                                                                 |
| 参数类型      | model 为 fx.GraphModule，      | 已覆盖：test_convert_replaces_observer_with_quantize_dequantize |
|               | observer_node 为 fx.Node       |                                                                 |
| 传参与不传参  | convert 的 model/              | 不适用                                                          |
|               | observer_node 均为必传，       |                                                                 |
|               | 无可选参数                     |                                                                 |
| 等价类/边界值 | 静态转换路径与动态转换路径     | 已覆盖：test_convert_dynamic_path_uses_choose_qparams_affine    |
| 正常传参场景  | 用 scale/zero_point 生成       | 已覆盖：test_convert_replaces_observer_with_quantize_dequantize |
|               | quantize/dequantize 节点       |                                                                 |
| 异常传参场景  | block_size/original_dtype      | 已覆盖：test_convert_raises_without_block_size /                |
|               | 未填充触发 AssertionError      | test_convert_raises_without_original_dtype                      |
| 混合设备类型  | 图变换接口，不涉及多 Tensor    | 不适用                                                          |
|               | 异构设备输入                   |                                                                 |

未覆盖项及原因：
- 传参与不传参：convert 无默认参数，model 与 observer_node 均必传，无「传参与不传参」对比维度。
- 混合设备类型：convert 为 FX 图变换接口，只做节点替换，不接收张量，无法构造异构设备输入。
- 真实量化内核：convert 产出的 quantize_affine/dequantize_affine 为 torchao（pt2e_quant）算子，
  本环境未安装 torchao，故用例用 unittest.mock 替换 pt2e_quant 命名空间，仅验证图变换契约（节点替换、
  缓冲区注册、动态/静态分支），不校验真实量化数值内核。

注：convert 仅做 FX 图结构变换，无 CPU/NPU 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证图结构（节点、缓冲区、输出边）语义。
"""
import os
import unittest
from unittest import mock

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

import torch.fx as fx
from torch.ao.quantization.observer import (
    AffineQuantizedObserverBase,
    MappingType,
    PerTensor,
    ZeroPointDomain,
)


class _DummyAffineObserver(AffineQuantizedObserverBase):
    """Minimal concrete observer used to drive convert()."""

    def forward(self, input):
        self.original_dtype = input.dtype
        self.block_size = tuple(input.shape)
        self.min_val = input.min().detach()
        self.max_val = input.max().detach()
        return input

    def calculate_qparams(self):
        scale = (self.max_val - self.min_val) / 255.0
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        return scale, zero_point


class _FakePt2eQuant:
    """Stand-in for the torchao pt2e_quant namespace used as FX call targets."""

    @staticmethod
    def quantize_affine(*args, **kwargs):
        return None

    @staticmethod
    def dequantize_affine(*args, **kwargs):
        return None

    @staticmethod
    def choose_qparams_affine(*args, **kwargs):
        return None


def _fresh_observer():
    return _DummyAffineObserver(
        MappingType.ASYMMETRIC, torch.uint8, PerTensor(), zero_point_domain=ZeroPointDomain.INT
    )


def _build_model(observer):
    """Build a GraphModule with a single call_module observer node."""
    root = torch.nn.Module()
    root.obs = observer
    graph = fx.Graph()
    x = graph.placeholder("x")
    obs_node = graph.call_module("obs", args=(x,))
    graph.output(obs_node)
    return fx.GraphModule(root, graph), obs_node


def _call_function_targets(graph_module):
    return [node.target for node in graph_module.graph.nodes if node.op == "call_function"]


def _getattr_names(graph_module):
    return [node.target for node in graph_module.graph.nodes if node.op == "get_attr"]


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverAffineQuantizedObserverBaseConvertUt(TestCase):
    """Functional tests for AffineQuantizedObserverBase.convert (FX graph transform)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.observer = _fresh_observer()

    def _populated_model(self):
        self.observer(torch.randn(3, 4))
        return _build_model(self.observer)

    def test_convert_replaces_observer_with_quantize_dequantize(self):
        """Static conversion splices quantize_affine and dequantize_affine into the graph."""
        model, obs_node = self._populated_model()
        with mock.patch.object(torch.ops, "pt2e_quant", _FakePt2eQuant):
            self.observer.convert(model, obs_node)
        targets = _call_function_targets(model)
        self.assertIn(_FakePt2eQuant.quantize_affine, targets)
        self.assertIn(_FakePt2eQuant.dequantize_affine, targets)
        # The observer node must be erased and its uses rewritten to the dequantize node.
        self.assertNotIn(obs_node, list(model.graph.nodes))
        out_node = next(node for node in model.graph.nodes if node.op == "output")
        self.assertIs(out_node.args[0].target, _FakePt2eQuant.dequantize_affine)

    def test_convert_registers_scale_and_zero_point_buffers(self):
        """Static conversion registers one _scale and one _zero_point buffer via get_attr."""
        model, obs_node = self._populated_model()
        with mock.patch.object(torch.ops, "pt2e_quant", _FakePt2eQuant):
            self.observer.convert(model, obs_node)
        attr_names = _getattr_names(model)
        scale_names = [name for name in attr_names if name.startswith("_scale")]
        zero_point_names = [name for name in attr_names if name.startswith("_zero_point")]
        self.assertEqual(len(scale_names), 1)
        self.assertEqual(len(zero_point_names), 1)
        state = model.state_dict()
        self.assertIn(scale_names[0], state)
        self.assertIn(zero_point_names[0], state)

    def test_convert_dynamic_path_uses_choose_qparams_affine(self):
        """The dynamic branch derives qparams at runtime via choose_qparams_affine."""
        self.observer.is_dynamic = True
        model, obs_node = self._populated_model()
        with mock.patch.object(torch.ops, "pt2e_quant", _FakePt2eQuant):
            self.observer.convert(model, obs_node)
        targets = _call_function_targets(model)
        self.assertIn(_FakePt2eQuant.choose_qparams_affine, targets)
        # No static _scale get_attr buffer should be registered on the dynamic path.
        self.assertEqual([name for name in _getattr_names(model) if name.startswith("_scale")], [])

    def test_convert_raises_without_block_size(self):
        """convert rejects an observer whose block_size is not yet populated."""
        model, obs_node = _build_model(self.observer)
        with mock.patch.object(torch.ops, "pt2e_quant", _FakePt2eQuant):
            with self.assertRaisesRegex(AssertionError, "Expecting block_size to be populated"):
                self.observer.convert(model, obs_node)

    def test_convert_raises_without_original_dtype(self):
        """convert rejects an observer whose original_dtype is not yet populated."""
        self.observer.block_size = (3, 4)
        model, obs_node = _build_model(self.observer)
        with mock.patch.object(torch.ops, "pt2e_quant", _FakePt2eQuant):
            with self.assertRaisesRegex(AssertionError, "Expecting original_dtype to be populated"):
                self.observer.convert(model, obs_node)


if __name__ == "__main__":
    run_tests()
