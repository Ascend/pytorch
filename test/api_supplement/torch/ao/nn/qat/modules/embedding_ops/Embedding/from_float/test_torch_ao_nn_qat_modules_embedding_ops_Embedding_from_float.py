# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.qat.modules.embedding_ops.Embedding.from_float 接口功能正确性
API 名称：torch.ao.nn.qat.modules.embedding_ops.Embedding.from_float
API 签名：Embedding.from_float(cls, mod, use_precomputed_fake_quant=False)

覆盖维度表：
| 覆盖维度      | 说明                                        | 覆盖情况                                           |
|---------------|---------------------------------------------|----------------------------------------------------|
| 空/非空       | mod 无 qconfig（空）与带合法 float-qparams  | 已覆盖：test_from_float_requires_qconfig /         |
|               | qconfig（非空）                             | test_from_float_returns_qat_embedding              |
| 枚举选项      | 无枚举参数；use_precomputed_fake_quant 取   | 已覆盖：test_from_float_use_precomputed_flag       |
|               | True/False                                  |                                                    |
| 参数类型      | mod 须为 torch.nn.Embedding 实例            | 已覆盖：test_from_float_rejects_non_embedding      |
| 传参与不传参  | 省略 use_precomputed_fake_quant 用默认      | 已覆盖：test_from_float_returns_qat_embedding /    |
|               | False；显式传 True                          | test_from_float_use_precomputed_flag               |
| 等价类/边界值 | padding_idx/max_norm/sparse 等属性组合      | 已覆盖：test_from_float_copies_attributes          |
| 正常传参场景  | 由 float Embedding 生成 QAT Embedding       | 已覆盖：test_from_float_forward_shape              |
|               | 并可前向                                    |                                                    |
| 异常传参场景  | 非 Embedding、缺 qconfig、qconfig=None、    | 已覆盖：四个 test_from_float_rejects_/requires_    |
|               | 错误 qscheme 各抛 AssertionError            | 方法                                               |
| 混合设备类型  | 单模块转换接口，不涉及多 Tensor             | 不适用                                             |
|               | 异构设备输入                                |                                                    |

未覆盖项及原因：
- 混合设备类型：QAT Embedding 的 fake-quant 权重仅由 CPU 量化后端提供，无 NPU/CUDA 执行路径，
  无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：QAT Embedding 只在 CPU 量化后端上有 fake-quant 语义，无 NPU/CUDA 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证转换正确性与属性/权重传递语义。
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
import torch.nn
from torch.ao.nn.qat.modules.embedding_ops import Embedding as QATEembedding

_EMB_QCONFIG = torch.ao.quantization.default_embedding_qat_qconfig


def _make_float_embedding(**kwargs):
    mod = torch.nn.Embedding(10, 5, **kwargs)
    mod.qconfig = _EMB_QCONFIG
    return mod


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEmbeddingFromFloatUt(TestCase):
    """Functional tests for torch.ao.nn.qat.modules.embedding_ops.Embedding.from_float."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_from_float_returns_qat_embedding(self):
        """A float Embedding is converted into a QAT Embedding with the same weight shape."""
        float_mod = _make_float_embedding()
        module = QATEembedding.from_float(float_mod)
        self.assertIsInstance(module, QATEembedding)
        self.assertEqual(tuple(module.weight.shape), tuple(float_mod.weight.shape))
        # from_float must carry the float module qconfig over to the QAT module.
        self.assertIs(module.qconfig, _EMB_QCONFIG)

    def test_from_float_copies_attributes(self):
        """Constructor attributes are propagated from the float module."""
        float_mod = _make_float_embedding(padding_idx=3, max_norm=2.0, norm_type=1.0, sparse=True)
        module = QATEembedding.from_float(float_mod)
        self.assertEqual(module.num_embeddings, 10)
        self.assertEqual(module.embedding_dim, 5)
        self.assertEqual(module.padding_idx, 3)
        self.assertEqual(module.max_norm, 2.0)
        self.assertEqual(module.norm_type, 1.0)
        self.assertTrue(module.sparse)

    def test_from_float_forward_shape(self):
        """The converted module forwards an index tensor to the expected output shape."""
        module = QATEembedding.from_float(_make_float_embedding())
        indices = torch.tensor([[1, 3, 5], [0, 2, 4]])
        out = module(indices)
        self.assertEqual(tuple(out.shape), (2, 3, 5))
        self.assertEqual(out.dtype, torch.float32)

    def test_from_float_use_precomputed_flag(self):
        """Passing use_precomputed_fake_quant=True is accepted and still converts."""
        module = QATEembedding.from_float(_make_float_embedding(), use_precomputed_fake_quant=True)
        self.assertIsInstance(module, QATEembedding)

    def test_from_float_rejects_non_embedding(self):
        """A non-Embedding float module is rejected."""
        other = torch.nn.EmbeddingBag(10, 5)
        other.qconfig = _EMB_QCONFIG
        with self.assertRaisesRegex(AssertionError, "only works for Embedding"):
            QATEembedding.from_float(other)

    def test_from_float_requires_qconfig(self):
        """A float module without a qconfig attribute is rejected."""
        float_mod = torch.nn.Embedding(10, 5)
        with self.assertRaisesRegex(AssertionError, "must have qconfig defined"):
            QATEembedding.from_float(float_mod)

    def test_from_float_rejects_invalid_qconfig(self):
        """A float module with qconfig=None is rejected."""
        float_mod = torch.nn.Embedding(10, 5)
        float_mod.qconfig = None
        with self.assertRaisesRegex(AssertionError, "must have a valid qconfig"):
            QATEembedding.from_float(float_mod)

    def test_from_float_rejects_wrong_qscheme(self):
        """A qconfig whose weight scheme is not per_channel_affine_float_qparams is rejected."""
        float_mod = torch.nn.Embedding(10, 5)
        float_mod.qconfig = torch.ao.quantization.default_qat_qconfig
        with self.assertRaisesRegex(
            AssertionError, "requires a qscheme of torch.per_channel_affine_float_qparams"
        ):
            QATEembedding.from_float(float_mod)


if __name__ == "__main__":
    run_tests()
