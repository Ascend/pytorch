# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.RNNCell 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.RNNCell
API 签名：RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | hx 为空(None)与非空两种前向 | 已覆盖：test_rnncell_forward_tanh、test_rnncell_forward_with_hx |
| 枚举选项 | nonlinearity 枚举 tanh/relu | 已覆盖：test_rnncell_forward_relu |
| 参数类型 | input/hx 为 2D Tensor | 已覆盖：test_rnncell_forward_tanh |
| 传参与不传参 | bias 默认 True 与显式 False | 已覆盖：test_rnncell_bias_disabled |
| 等价类/边界值 | 单样本 batch=1 输入 | 已覆盖：test_rnncell_forward_single_sample |
| 正常传参场景 | 前向输出 shape 正确 | 已覆盖：test_rnncell_forward_tanh |
| 异常传参场景 | 非法 nonlinearity/尺寸不匹配 | 已覆盖：rnncell_unknown_nonlinearity_raises、rnncell_check_forward_input_mismatch |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 RNNCell 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- float16 权重：fp16 动态内核仅 fbgemm/onednn 提供，qnnpack 平台不可用，故未覆盖。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 RNNCell 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
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

import torch.ao.nn.quantized.dynamic
from torch.ao.nn.quantized.dynamic.modules.rnn import RNNCell

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_BATCH = 4


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized RNNCell only provides kernels on the CPU quantized
    backend, so there is no NPU/CUDA execution path to compare against.
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
            # This engine has no kernel for the target op; try the next one.
            continue
        return name
    raise unittest.SkipTest("No CPU quantized engine can run the target op on this platform")


def _probe():
    RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)(torch.randn(_BATCH, _INPUT_SIZE))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRnncellUt(TestCase):
    """Functional tests for the dynamic quantized RNNCell module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_rnncell_construct_attributes(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell.input_size, _INPUT_SIZE)
        self.assertEqual(cell.hidden_size, _HIDDEN_SIZE)
        self.assertTrue(cell.bias)
        self.assertEqual(cell.nonlinearity, "tanh")
        self.assertEqual(cell.weight_dtype, torch.qint8)

    def test_rnncell_bias_disabled(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE, bias=False)
        self.assertFalse(cell.bias)
        self.assertIsNone(cell.bias_ih)
        self.assertIsNone(cell.bias_hh)

    def test_rnncell_forward_tanh(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        out = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_BATCH, _HIDDEN_SIZE))
        self.assertEqual(out.dtype, torch.float32)

    def test_rnncell_forward_relu(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE, nonlinearity="relu")
        out = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_BATCH, _HIDDEN_SIZE))

    def test_rnncell_forward_with_hx(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        x = torch.randn(_BATCH, _INPUT_SIZE)
        hx = torch.randn(_BATCH, _HIDDEN_SIZE)
        out = cell(x, hx)
        self.assertEqual(tuple(out.shape), (_BATCH, _HIDDEN_SIZE))

    def test_rnncell_forward_single_sample(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        out = cell(torch.randn(1, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (1, _HIDDEN_SIZE))

    def test_rnncell_unknown_nonlinearity_raises(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE, nonlinearity="sigmoid")
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE))

    def test_rnncell_check_forward_input_mismatch(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE + 1))

    def test_rnncell_check_forward_hidden_mismatch(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE), torch.randn(_BATCH, _HIDDEN_SIZE + 1))

    def test_rnncell_from_float_returns_rnncell(self):
        float_mod = torch.nn.RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        float_mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
        cell = RNNCell.from_float(float_mod)
        self.assertIsInstance(cell, RNNCell)
        self.assertEqual(cell.nonlinearity, "tanh")
        self.assertEqual(tuple(cell(torch.randn(_BATCH, _INPUT_SIZE)).shape), (_BATCH, _HIDDEN_SIZE))

    def test_rnncell_from_float_rejects_unsupported_type(self):
        with self.assertRaises(AssertionError):
            RNNCell.from_float(torch.nn.Linear(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_rnncell_get_name(self):
        cell = RNNCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell._get_name(), "DynamicQuantizedRNNCell")


if __name__ == "__main__":
    run_tests()
