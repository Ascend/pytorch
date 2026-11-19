# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.sparse.Embedding.from_pretrained 接口功能正确性
API 名称：torch.nn.modules.sparse.Embedding.from_pretrained
API 签名：from_pretrained(embeddings, freeze=True, padding_idx=None, max_norm=None,
                          norm_type=2.0, scale_grad_by_freq=False, sparse=False) -> Embedding

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认权重与 padding_idx 场景（_weight 路径不清零 padding 行） | 已覆盖（padding_idx_keeps_pretrained_row） |
| 枚举选项 | freeze=True/False 两分支；max_norm 非 None 触发重归一化 | 已覆盖（freeze_flag/max_norm_renormalizes） |
| 参数类型 | embeddings 必须 2D；1D/3D 抛 AssertionError；dtype 按输入保持 | 已覆盖（invalid_dim_raises/dtype_preserved） |
| 传参与不传参 | 默认调用与显式 freeze/padding_idx/max_norm 传参 | 已覆盖（default_freeze/freeze_flag） |
| 等价类/边界值 | num_embeddings/embedding_dim 由输入形状推断；查找输出与预训练权重行一致 | 已覆盖（test_from_pretrained_forward_lookup） |
| 正常传参场景 | freeze=False 训练一步权重更新且梯度非空；freeze=True 无梯度 | 已覆盖（test_from_pretrained_training_updates_weight_on_device） |
| 异常传参场景 | embeddings 非 2D 抛 AssertionError(expected to be 2-dimensional) | 已覆盖（invalid_dim_raises） |
| 混合设备类型 | 设备权重 → 模块权重落设备，查找输出与 CPU Golden（同权重行取值）精确对照 | 已覆盖（test_from_pretrained_forward_lookup_on_device） |

未覆盖项及原因：
- norm_type 非 2、scale_grad_by_freq、sparse=True 未单独设例：三者仅在 backward/稀疏
  路径生效，from_pretrained 仅原样透传给 Embedding 构造，无独立契约面。
- 未覆盖存储共享断言（weight 与输入张量 data_ptr 一致）：文档未承诺存储共享语义，
  不作为契约断言。

注意：from_pretrained 为 classmethod，纯结构断言用 assertEqual/assertIs；
     查找输出等浮点结果以同权重行取值为 Golden，用 torch.testing.assert_close 对照；
     max_norm 前向在设备上对被查找行原地重归一化（原生契约仅作用于 input 中出现的行），
     按被查找行与输出行范数上界、其余行保持预训练值断言。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

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
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEmbeddingFromPretrained(TestCase):
    """Comprehensive cases for Embedding.from_pretrained (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_from_pretrained_default_freeze(self):
        """Default freeze=True builds an Embedding with frozen pretrained weights."""
        torch.manual_seed(3)
        embeddings = torch.randn(6, 3)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(embeddings)
        self.assertIsInstance(module, torch.nn.Embedding)
        self.assertEqual(module.num_embeddings, 6)
        self.assertEqual(module.embedding_dim, 3)
        self.assertFalse(module.weight.requires_grad)
        torch.testing.assert_close(module.weight.detach(), embeddings)

    def test_from_pretrained_freeze_flag(self):
        """freeze=False keeps the weight trainable; freeze=True freezes it."""
        torch.manual_seed(5)
        embeddings = torch.randn(4, 5)
        frozen = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=True)
        trainable = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=False)
        self.assertFalse(frozen.weight.requires_grad)
        self.assertTrue(trainable.weight.requires_grad)
        torch.testing.assert_close(frozen.weight.detach(), embeddings)
        torch.testing.assert_close(trainable.weight.detach(), embeddings)

    def test_from_pretrained_forward_lookup(self):
        """Forward lookup returns exactly the pretrained weight rows."""
        torch.manual_seed(7)
        embeddings = torch.randn(8, 4)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(embeddings)
        indices = torch.tensor([1, 3, 6], dtype=torch.long)
        out = module(indices)
        self.assertEqual(tuple(out.shape), (3, 4))
        torch.testing.assert_close(out, embeddings[[1, 3, 6]])

    def test_from_pretrained_invalid_dim_raises(self):
        """1D and 3D embeddings raise AssertionError per the source contract."""
        with self.assertRaisesRegex(AssertionError, "2-dimensional"):
            torch.nn.modules.sparse.Embedding.from_pretrained(torch.randn(6))
        with self.assertRaisesRegex(AssertionError, "2-dimensional"):
            torch.nn.modules.sparse.Embedding.from_pretrained(torch.randn(2, 3, 4))

    def test_from_pretrained_padding_idx_keeps_pretrained_row(self):
        """padding_idx does not zero the pretrained row (the _weight path skips fill)."""
        torch.manual_seed(11)
        embeddings = torch.randn(6, 3)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=0)
        self.assertEqual(module.padding_idx, 0)
        torch.testing.assert_close(module.weight.detach()[0], embeddings[0])
        torch.testing.assert_close(module.weight.detach(), embeddings)
        indices = torch.tensor([0, 2], dtype=torch.long)
        out = module(indices)
        torch.testing.assert_close(out, embeddings[[0, 2]])

    def test_from_pretrained_dtype_preserved(self):
        """The pretrained dtype is preserved on the module weight."""
        embeddings = torch.randn(4, 3, dtype=torch.float16)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(embeddings)
        self.assertEqual(module.weight.dtype, torch.float16)

    def test_from_pretrained_forward_lookup_on_device(self):
        """Device pretrained weights produce a device module matching the golden rows."""
        torch.manual_seed(13)
        embeddings = torch.randn(6, 4, device=self.device)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=True)
        self.assertEqual(module.weight.device.type, self.device_name)
        self.assertFalse(module.weight.requires_grad)
        indices = torch.tensor([0, 5, 2], dtype=torch.long, device=self.device)
        out = module(indices)
        self.assertEqual(out.device.type, self.device_name)
        self.assertEqual(tuple(out.shape), (3, 4))
        torch.testing.assert_close(out.detach().cpu(), embeddings.detach().cpu()[[0, 5, 2]])

    def test_from_pretrained_max_norm_renormalizes_on_device(self):
        """max_norm renormalizes looked-up rows above the bound before lookup.

        The native embedding renorm contract is scoped to the rows that appear
        in the input batch; rows outside the batch keep their pretrained values.
        """
        torch.manual_seed(17)
        embeddings = torch.randn(8, 3, device=self.device) * 3.0
        module = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=True, max_norm=1.0)
        pre = embeddings.detach().clone()
        indices = torch.tensor([1, 4], dtype=torch.long, device=self.device)
        out = module(indices)
        self.assertLessEqual(out.norm(dim=1).max().item(), 1.0 + 1e-5)
        looked_up = module.weight.detach()[[1, 4]].norm(dim=1)
        self.assertTrue(bool((looked_up <= 1.0 + 1e-5).all()))
        untouched = module.weight.detach()[[0, 2, 3, 5, 6, 7]]
        torch.testing.assert_close(untouched, pre[[0, 2, 3, 5, 6, 7]])

    def test_from_pretrained_freeze_true_no_grad(self):
        """A frozen pretrained module produces non-differentiable outputs."""
        torch.manual_seed(19)
        embeddings = torch.randn(6, 3, device=self.device)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=True)
        indices = torch.tensor([1, 2], dtype=torch.long, device=self.device)
        out = module(indices)
        self.assertFalse(out.requires_grad)
        self.assertFalse(module.weight.requires_grad)
        self.assertIsNone(module.weight.grad)

    def test_from_pretrained_training_updates_weight_on_device(self):
        """freeze=False trains on device: gradient exists and one SGD step updates."""
        torch.manual_seed(23)
        embeddings = torch.randn(6, 3, device=self.device)
        module = torch.nn.modules.sparse.Embedding.from_pretrained(
            embeddings, freeze=False)
        indices = torch.tensor([1, 2, 3], dtype=torch.long, device=self.device)
        module(indices).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        optimizer = torch.optim.SGD([module.weight], lr=0.1)
        weight_before = module.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(module.weight.detach(), weight_before))


if __name__ == "__main__":
    run_tests()
