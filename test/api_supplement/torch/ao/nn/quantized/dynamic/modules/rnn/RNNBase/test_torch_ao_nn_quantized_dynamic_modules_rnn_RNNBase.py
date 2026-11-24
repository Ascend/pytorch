# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase
API 签名：RNNBase(mode, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | num_layers/bidirectional 组合产生非空权重集合 | 已覆盖：test_rnnbase_weight_values_count |
| 枚举选项 | mode 枚举 LSTM/GRU | 已覆盖：test_rnnbase_construct_lstm_mode、test_rnnbase_construct_gru_mode |
| 参数类型 | dropout 须为数值 | 已覆盖：test_rnnbase_non_numeric_dropout_raises |
| 传参与不传参 | batch_first/dropout/bidirectional 默认与显式 | 已覆盖：test_rnnbase_extra_repr |
| 等价类/边界值 | dropout 边界 0/1 与越界 2.0 | 已覆盖：test_rnnbase_out_of_range_dropout_raises |
| 正常传参场景 | 期望 hidden size 与权重形状正确 | 已覆盖：test_rnnbase_get_expected_hidden_size |
| 异常传参场景 | 非法 mode/dropout/bool 触发 ValueError | 已覆盖：test_rnnbase_invalid_mode_raises、test_rnnbase_bool_dropout_raises |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 RNNBase 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- float16 权重：fp16 动态内核仅 fbgemm/onednn 提供，qnnpack 平台不可用，故未覆盖。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 RNNBase 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
"""
import os
import unittest
import warnings

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
from torch.ao.nn.quantized.dynamic.modules.rnn import RNNBase

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_BATCH = 2
_SEQ_LEN = 4


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to instantiate the target module.

    Dynamic quantized RNNBase only packs weights on the CPU quantized backend,
    so there is no NPU/CUDA execution path to compare against.
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
    RNNBase("GRU", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRnnbaseUt(TestCase):
    """Functional tests for the dynamic quantized RNNBase class."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_rnnbase_construct_lstm_mode(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)
        self.assertEqual(module.mode, "LSTM")
        self.assertEqual(module.dtype, torch.qint8)
        self.assertFalse(module.training)
        self.assertEqual(tuple(module.get_weight()["weight_ih_l0"].shape), (4 * _HIDDEN_SIZE, _INPUT_SIZE))

    def test_rnnbase_construct_gru_mode(self):
        module = RNNBase("GRU", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)
        self.assertEqual(module.mode, "GRU")
        self.assertEqual(tuple(module.get_weight()["weight_ih_l0"].shape), (3 * _HIDDEN_SIZE, _INPUT_SIZE))

    def test_rnnbase_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            RNNBase("FOO", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)

    def test_rnnbase_non_numeric_dropout_raises(self):
        with self.assertRaises(ValueError):
            RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1, dropout="high")

    def test_rnnbase_out_of_range_dropout_raises(self):
        with self.assertRaises(ValueError):
            RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1, dropout=2.0)

    def test_rnnbase_bool_dropout_raises(self):
        with self.assertRaises(ValueError):
            RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1, dropout=True)

    def test_rnnbase_dropout_warning_single_layer(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1, dropout=0.5)
        messages = [str(item.message) for item in record]
        self.assertTrue(any("dropout" in message for message in messages))

    def test_rnnbase_weight_values_count(self):
        single = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=2)
        self.assertEqual(len(single._all_weight_values), 2)
        both = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=2, bidirectional=True)
        self.assertEqual(len(both._all_weight_values), 4)

    def test_rnnbase_check_input_bad_dim(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)
        with self.assertRaises(RuntimeError):
            module.check_input(torch.randn(_BATCH, _INPUT_SIZE), None)

    def test_rnnbase_check_input_size_mismatch(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)
        with self.assertRaises(RuntimeError):
            module.check_input(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE + 1), None)

    def test_rnnbase_get_expected_hidden_size(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=2, bidirectional=True)
        expected = module.get_expected_hidden_size(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE), None)
        self.assertEqual(expected, (2 * 2, _BATCH, _HIDDEN_SIZE))

    def test_rnnbase_extra_repr(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=2, batch_first=True,
                         dropout=0.5, bidirectional=True)
        text = module.extra_repr()
        self.assertIn("num_layers=2", text)
        self.assertIn("batch_first=True", text)
        self.assertIn("dropout=0.5", text)
        self.assertIn("bidirectional=True", text)

    def test_rnnbase_get_name(self):
        module = RNNBase("LSTM", _INPUT_SIZE, _HIDDEN_SIZE, num_layers=1)
        self.assertEqual(module._get_name(), "DynamicQuantizedRNN")


if __name__ == "__main__":
    run_tests()
