# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantizable.modules.rnn.LSTMCell.from_params 接口功能正确性
API 名称：torch.ao.nn.quantizable.modules.rnn.LSTMCell.from_params
API 签名：LSTMCell.from_params(cls, wi, wh, bi=None, bh=None, split_gates=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | bi/bh 同时为 None（无偏置）与同时给出（有偏置） | 已覆盖：test_from_params_no_bias / test_from_params_copies_biases |
| 枚举选项 | split_gates 取 False/True 两态 | 已覆盖：test_from_params_returns_cell / test_from_params_split_gates |
| 参数类型 | wi/wh/bi/bh 为 Tensor，split_gates 为 bool | 已覆盖：test_from_params_copies_weights / test_from_params_split_gates |
| 传参与不传参 | 省略 bi/bh 用默认 None；显式传入偏置 | 已覆盖：test_from_params_no_bias / test_from_params_copies_biases |
| 等价类/边界值 | 不同 input/hidden 维度组合、split_gates 分支 | 已覆盖：test_from_params_returns_cell / test_from_params_split_gates |
| 正常传参场景 | 由权重/偏置构造 cell 并可前向 | 已覆盖：test_from_params_forward_shape / test_from_params_split_gates_forward |
| 异常传参场景 | 仅给 bi 或仅给 bh 时抛 AssertionError | 已覆盖：test_from_params_bi_without_bh_raises / test_from_params_bh_without_bi_raises |
| 混合设备类型 | 单模块构造接口，不涉及多 Tensor 异构设备输入 | 不适用 |

未覆盖项及原因：
- 混合设备类型：LSTMCell.from_params 仅按输入 Tensor 的 shape 构造 CPU 模块并做数值复制，
  无 NPU/CUDA 执行路径；输入权重均为同一设备上的 Tensor，无稳定的异构设备契约，故不单列。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为权重/偏置到 cell 模块的构造工具，其数值复制路径在 CPU 上即可完整验证，
     权重为普通 float Tensor（非浮点计算算子输出），故按精确相等断言，不做 CPU/NPU Golden 对照。
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

import torch.nn
from torch.ao.nn.quantizable.modules.rnn import LSTMCell

_INPUT_DIM = 3
_HIDDEN_DIM = 5


def _params(with_bias=True):
    wi = torch.randn(4 * _HIDDEN_DIM, _INPUT_DIM)
    wh = torch.randn(4 * _HIDDEN_DIM, _HIDDEN_DIM)
    bi = torch.randn(4 * _HIDDEN_DIM) if with_bias else None
    bh = torch.randn(4 * _HIDDEN_DIM) if with_bias else None
    return wi, wh, bi, bh


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLSTMCellFromParamsUt(TestCase):
    """Functional tests for torch.ao.nn.quantizable.modules.rnn.LSTMCell.from_params."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_from_params_returns_cell(self):
        """Dimensions are derived from the passed weights and bias is enabled."""
        wi, wh, bi, bh = _params()
        cell = LSTMCell.from_params(wi, wh, bi, bh)
        self.assertIsInstance(cell, LSTMCell)
        self.assertEqual(cell.input_size, _INPUT_DIM)
        self.assertEqual(cell.hidden_size, _HIDDEN_DIM)
        self.assertTrue(cell.bias)

    def test_from_params_copies_weights(self):
        """The input/hidden weights are copied verbatim into the gate linears."""
        wi, wh, bi, bh = _params()
        cell = LSTMCell.from_params(wi, wh, bi, bh)
        self.assertTrue(torch.equal(cell.igates.weight.data, wi))
        self.assertTrue(torch.equal(cell.hgates.weight.data, wh))

    def test_from_params_copies_biases(self):
        """The input/hidden biases are copied verbatim into the gate linears."""
        wi, wh, bi, bh = _params(with_bias=True)
        cell = LSTMCell.from_params(wi, wh, bi, bh)
        self.assertTrue(torch.equal(cell.igates.bias.data, bi))
        self.assertTrue(torch.equal(cell.hgates.bias.data, bh))

    def test_from_params_no_bias(self):
        """Omitting bi/bh builds a bias-free cell with no gate biases."""
        wi, wh, _, _ = _params(with_bias=False)
        cell = LSTMCell.from_params(wi, wh)
        self.assertFalse(cell.bias)
        self.assertIsNone(cell.igates.bias)
        self.assertIsNone(cell.hgates.bias)

    def test_from_params_split_gates(self):
        """split_gates=True rebuilds each gate as its own chunked Linear weight."""
        wi, wh, bi, bh = _params()
        cell = LSTMCell.from_params(wi, wh, bi, bh, split_gates=True)
        self.assertTrue(cell.split_gates)
        self.assertIsInstance(cell.igates, torch.nn.ModuleDict)
        self.assertEqual(list(cell.igates.keys()), ["input", "forget", "cell", "output"])
        wi_chunks = wi.chunk(4, dim=0)
        wh_chunks = wh.chunk(4, dim=0)
        bi_chunks = bi.chunk(4, dim=0)
        bh_chunks = bh.chunk(4, dim=0)
        for idx, name in enumerate(["input", "forget", "cell", "output"]):
            # igates consumes wi/bi, hgates consumes wh/bh, split per gate.
            self.assertTrue(torch.equal(cell.igates[name].weight.data, wi_chunks[idx]))
            self.assertTrue(torch.equal(cell.igates[name].bias.data, bi_chunks[idx]))
            self.assertTrue(torch.equal(cell.hgates[name].weight.data, wh_chunks[idx]))
            self.assertTrue(torch.equal(cell.hgates[name].bias.data, bh_chunks[idx]))

    def test_from_params_forward_shape(self):
        """A constructed cell forwards to (h, c) with the expected batch/hidden shape."""
        wi, wh, bi, bh = _params()
        cell = LSTMCell.from_params(wi, wh, bi, bh)
        hy, cy = cell(torch.randn(2, _INPUT_DIM))
        self.assertEqual(tuple(hy.shape), (2, _HIDDEN_DIM))
        self.assertEqual(tuple(cy.shape), (2, _HIDDEN_DIM))

    def test_from_params_split_gates_forward(self):
        """A split-gate cell still runs forward and returns the same output shapes."""
        wi, wh, bi, bh = _params()
        cell = LSTMCell.from_params(wi, wh, bi, bh, split_gates=True)
        hy, cy = cell(torch.randn(2, _INPUT_DIM))
        self.assertEqual(tuple(hy.shape), (2, _HIDDEN_DIM))
        self.assertEqual(tuple(cy.shape), (2, _HIDDEN_DIM))

    def test_from_params_bi_without_bh_raises(self):
        """Supplying only bi (bh missing) violates the paired-bias contract."""
        wi, wh, bi, _ = _params(with_bias=True)
        with self.assertRaisesRegex(AssertionError, "bi and bh must both be None or both have values"):
            LSTMCell.from_params(wi, wh, bi)

    def test_from_params_bh_without_bi_raises(self):
        """Supplying only bh (bi missing) violates the paired-bias contract."""
        wi, wh, _, bh = _params(with_bias=True)
        with self.assertRaisesRegex(AssertionError, "bi and bh must both be None or both have values"):
            LSTMCell.from_params(wi, wh, None, bh)


if __name__ == "__main__":
    run_tests()
