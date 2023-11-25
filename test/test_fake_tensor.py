from typing import Any, Dict, List, Tuple, Union
import functools
import math
import numpy as np
from torch._functorch.aot_autograd import aot_module_simplified
import torch
from torch.library import Library, impl
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestFastGelu(TestCase):

    def test_fast_gelu(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu.fast_gelu(a)
            self.assertTrue(a.shape == result.shape)

    def test_fast_gelu_backward(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu.fast_gelu(a)
            result.sum().backward()
            self.assertTrue(a.shape == a.grad.shape)

    def test_npu_fast_gelu(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu.npu_fast_gelu(a)

            self.assertEqual(a.shape, result.shape)


class TestIncreFlashAttention(TestCase):
    def testIncreFlashAttention(self):
        with fake_mode:
            q = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            k = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            v = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
            res = torch.ops.npu.npu_incre_flash_attention(q, k, v)

            print("q.shape: ", q.shape)
            print("res.shape: ", res.shape)
            self.assertTrue(q.shape == res.shape)


class TestNpuBmmV2(TestCase):
    def test_npu_bmmV2(self):
        with fake_mode:
            npu_input1 = torch.randn(10, 3, 4).npu()
            npu_input2 = torch.randn(10, 4, 5).npu()
            output_size = []
            result = torch_npu.npu_bmmV2(npu_input1, npu_input2, output_size)

            self.assertEqual(result.dtype, npu_input1.dtype)
            self.assertEqual(result.shape, torch.matmul(npu_input1, npu_input2).shape)


class TestNpuDropout(TestCase):

    def test_npu_dropout(self):
        b = torch.randn(2, 3).npu()
        b.requires_grad = True
        result_b = torch_npu._npu_dropout(b, 0.5)

        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu._npu_dropout(a, 0.5)
            self.assertTrue(result[0].shape == result_b[0].shape)
            self.assertTrue(result[1].shape == result_b[1].shape)

    def test_npu_dropout_backward(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu._npu_dropout(a, 0.5)
            result[0].sum().backward()
            self.assertTrue(a.shape == a.grad.shape)


class TestNpuDtypeCast(TestCase):
    def test_npu_dtype_cast(self):
        with fake_mode:
            npu_input = torch.randn((2, 3), dtype=torch.float32).npu()
            dst_dtype = torch.float16
            result = torch_npu.npu_dtype_cast(npu_input, dst_dtype)

            self.assertEqual(result.dtype, dst_dtype)
            self.assertEqual(result.shape, npu_input.shape)

    def test_npu_dtype_cast_backward(self):
        with fake_mode:
            npu_input = torch.randn((2, 3), dtype=torch.float32).npu()
            npu_input.requires_grad = True
            dst_dtype = torch.float16
            result = torch_npu.npu_dtype_cast(npu_input, dst_dtype)
            result.sum().backward()
            self.assertEqual(result.dtype, dst_dtype)
            self.assertEqual(npu_input.shape, npu_input.grad.shape)


class TestNpuRotaryMul(TestCase):
    def test_npu_rotary_mul(self):
        with fake_mode:
            embedding = torch.randn(2, 8192, 5, 125, dtype=torch.float16, requires_grad=True).npu()
            cosine = torch.randn(1, 8192, 1, 128, dtype=torch.float16, requires_grad=True).npu()
            sine = torch.randn(1, 8192, 1, 128, dtype=torch.float16, requires_grad=True).npu()
            ret = torch.ops.npu.npu_rotary_mul(embedding, cosine, sine)

            self.assertEqual(embedding.shape, ret.shape)
            self.assertEqual(embedding.dtype, ret.dtype)


class TestNpuTranspose(TestCase):
    def test_npu_transpose(self):
        with fake_mode:
            npu_input = torch.randn((5, 3, 6, 4)).npu()
            perm = [1, 0, 2, 3]
            exp_shape = npu_input.permute(perm).shape
            result = torch_npu.npu_transpose(npu_input, perm)

            self.assertEqual(result.shape, exp_shape)


class TestPromptFlashAttention(TestCase):
    def testPromptFlashAttention(self):
        with fake_mode:
            q = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            k = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            v = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
            res = torch.ops.npu.npu_prompt_flash_attention(q, k, v)

            print("q.shape: ", q.shape)
            print("res.shape: ", res.shape)
            self.assertTrue(q.shape == res.shape)


class TestMaskedSoftmaxWithRelPosBias(TestCase):
    # meta shape推导
    def testMaskedSoftmaxWithRelPosBias(self):
        with fake_mode:
            x = torch.randn(96, 2, 2, 32, 32, dtype=torch.float)
            relative_pos_bias = torch.randn(1, 1, 2, 32, 32, dtype=torch.float)
            atten_mask = torch.randn(1, 2, 1, 32, 32, dtype=torch.float)
            x.requires_grad = True
            atten_mask.requires_grad = True
            relative_pos_bias.requires_grad = True
            res = torch.ops.npu.npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias)
            print("x.shape: ", x.shape)
            print("res.shape: ", res.shape)
            self.assertTrue(x.shape == res.shape)


class TestScatterUpdateMeta(TestCase):

    def test_scatter_update_meta(self):
        with FakeTensorMode() as mode:
            in_self = torch.randn(4, 4, 32, 256, dtype=torch.float16).npu()
            in_indices = torch.tensor([1, 1, 1, 1]).npu()
            in_updates = torch.randn(4, 4, 1, 256, dtype=torch.float16).npu()
            fake_self = mode.from_tensor(in_self)
            fake_indices = mode.from_tensor(in_indices)
            fake_updates = mode.from_tensor(in_updates)
            self.assertIsNotNone(fake_self)
            self.assertIsNotNone(fake_indices)
            self.assertIsNotNone(fake_updates)
            fake_result = torch.ops.npu.scatter_update(fake_self, fake_indices, fake_updates, -2)

            self.assertEqual(fake_result.shape, in_self.shape)
            self.assertEqual(fake_result.dtype, in_self.dtype)
            self.assertEqual(fake_result.device, in_self.device)
            self.assertTrue(isinstance(fake_result, FakeTensor))
            self.assertIsNot(fake_result, fake_self)
            self.assertIsNot(fake_result, in_self)


    def test_scatter_update__meta(self):
        with FakeTensorMode() as mode:
            in_self = torch.randn(4, 4, 32, 256, dtype=torch.float32).npu()
            in_indices = torch.tensor([1, 1, 1, 1]).npu()
            in_updates = torch.randn(4, 4, 1, 256, dtype=torch.float32).npu()
            fake_self = mode.from_tensor(in_self)
            fake_indices = mode.from_tensor(in_indices)
            fake_updates = mode.from_tensor(in_updates)
            self.assertIsNotNone(fake_self)
            self.assertIsNotNone(fake_indices)
            self.assertIsNotNone(fake_updates)
            fake_result = torch.ops.npu.scatter_update_(fake_self, fake_indices, fake_updates, -2)

            self.assertEqual(fake_result.shape, in_self.shape)
            self.assertEqual(fake_result.dtype, in_self.dtype)
            self.assertEqual(fake_result.device, in_self.device)
            self.assertTrue(isinstance(fake_result, FakeTensor))
            self.assertIs(fake_result, fake_self)
            self.assertIsNot(fake_result, in_self)

if __name__ == "__main__":
    run_tests()
