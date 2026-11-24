# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.GRUCell 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.GRUCell
API 签名：GRUCell(input_size, hidden_size, bias=True, dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | hx 为空(None)与非空两种前向 | 已覆盖：test_grucell_forward_default_hx、test_grucell_forward_with_hx |
| 枚举选项 | 权重 dtype 枚举 qint8 | 已覆盖：test_grucell_construct_attributes |
| 参数类型 | input/hx 为 2D Tensor | 已覆盖：test_grucell_forward_returns_hidden_state |
| 传参与不传参 | bias 默认 True 与显式 False | 已覆盖：test_grucell_construct_attributes、test_grucell_bias_disabled |
| 等价类/边界值 | 单样本 batch=1 输入 | 已覆盖：test_grucell_forward_single_sample |
| 正常传参场景 | 前向输出 shape 正确 | 已覆盖：test_grucell_forward_returns_hidden_state |
| 异常传参场景 | 尺寸不匹配报错 | 已覆盖：grucell_check_forward_input_mismatch、grucell_check_forward_hidden_mismatch |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 GRUCell 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
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
from torch.ao.nn.quantized.dynamic.modules.rnn import GRUCell

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_BATCH = 4


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized GRUCell only provides kernels on the CPU quantized
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
    GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)(torch.randn(_BATCH, _INPUT_SIZE))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGrucellUt(TestCase):
    """Functional tests for the dynamic quantized GRUCell module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_grucell_construct_attributes(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell.input_size, _INPUT_SIZE)
        self.assertEqual(cell.hidden_size, _HIDDEN_SIZE)
        self.assertTrue(cell.bias)
        self.assertEqual(cell.weight_dtype, torch.qint8)

    def test_grucell_bias_disabled(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE, bias=False)
        self.assertFalse(cell.bias)
        self.assertIsNone(cell.bias_ih)
        self.assertIsNone(cell.bias_hh)

    def test_grucell_forward_returns_hidden_state(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        out = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_BATCH, _HIDDEN_SIZE))
        self.assertEqual(out.dtype, torch.float32)

    def test_grucell_forward_with_hx(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        x = torch.randn(_BATCH, _INPUT_SIZE)
        hx = torch.randn(_BATCH, _HIDDEN_SIZE)
        out = cell(x, hx)
        self.assertEqual(tuple(out.shape), (_BATCH, _HIDDEN_SIZE))

    def test_grucell_forward_default_hx(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        x = torch.randn(_BATCH, _INPUT_SIZE)
        with_hx = cell(x, torch.zeros(_BATCH, _HIDDEN_SIZE))
        default_hx = cell(x)
        torch.testing.assert_close(default_hx, with_hx)

    def test_grucell_forward_single_sample(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        out = cell(torch.randn(1, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (1, _HIDDEN_SIZE))

    def test_grucell_check_forward_input_mismatch(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE + 1))

    def test_grucell_check_forward_hidden_mismatch(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE), torch.randn(_BATCH, _HIDDEN_SIZE + 1))

    def test_grucell_from_float_returns_grucell(self):
        float_mod = torch.nn.GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        float_mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
        cell = GRUCell.from_float(float_mod)
        self.assertIsInstance(cell, GRUCell)
        self.assertEqual(cell.weight_dtype, torch.qint8)
        self.assertEqual(tuple(cell(torch.randn(_BATCH, _INPUT_SIZE)).shape), (_BATCH, _HIDDEN_SIZE))

    def test_grucell_from_float_rejects_unsupported_type(self):
        with self.assertRaises(AssertionError):
            GRUCell.from_float(torch.nn.Linear(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_grucell_get_name(self):
        cell = GRUCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell._get_name(), "DynamicQuantizedGRUCell")


if __name__ == "__main__":
    run_tests()
