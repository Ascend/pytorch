import unittest
import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torch_npu.contrib.module import MultiheadAttention
from torch_npu.contrib.module.multihead_attention import _MHAConfig
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module.multihead_attention import _quant_noise, _NpuLinear, _MHAConfig, MultiheadAttention
FORMAT_ND = 2
FORMAT_NZ = 29
npu_device = "npu:0"
_MHAConfig.set_fussion()


class TestMultiheadAttention(unittest.TestCase):
    def test_MultiheadAttention(self):
        model = MultiheadAttention(embed_dim=1024,
                                   num_heads=16,
                                   dropout=0.1,
                                   kdim=1024,
                                   vdim=1024,
                                   self_attention=True,
                                   encoder_decoder_attention=True)
        _, query = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, key = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, value = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, key_padding_mask = create_common_tensor([np.float16, FORMAT_NZ, (16, 16, 64, 64)], -65504, 65504)
        bsz = 16
        tgt_len = 64
        s_len = 64
        model = model.to("npu")
        output = model(query, key, value, bsz, tgt_len, s_len, key_padding_mask)

    def test_multihead_attention_delf_attention_mismatch(self):
        with self.assertRaises(ValueError):
            MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                kdim=64,
                vdim=128,
                self_attention=True,
            )

    def test_multihead_attention_invalid_embed_dim(self):
        with self.assertRaises(ValueError):
            MultiheadAttention(embed_dim=10, num_heads=3)

    def test_npu_linear_forward_invalid_dim(self):
        module = _NpuLinear(10, 20)
        input_tensor = torch.randn(5, 10, 3)

        with self.assertRaises(RuntimeError):
            module(input_tensor)

    def test_multihead_attention_add_bias_kv(self):
        model = MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            add_bias_kv=True,
        )

        self.assertIsNotNone(model.bias_k)
        self.assertIsNotNone(model.bias_v)

        model.reset_parameters()

    def test_quant_noise_invalid_block_size_conv_large(self):
        conv = nn.Conv2d(10, 20, (3, 3))
        with self.assertRaises(ValueError):
            _quant_noise(conv, 0.1, 5)

    def test_quant_noise_invalid_block_size_conv_1x1(self):
        conv = nn.Conv2d(10, 20, (1, 1))
        with self.assertRaises(ValueError):
            _quant_noise(conv, 0.1, 3)

    def test_quant_noise_invalid_block_size_2d(self):
        linear = nn.Linear(10, 20)
        with self.assertRaises(ValueError):
            _quant_noise(linear, 0.1, 3)

    def test_quant_noise_invalid_module_type(self):
        with self.assertRaises(TypeError):
            _quant_noise(nn.Conv1d(10, 10, 3), 0.1, 8)

if __name__ == "__main__":
    run_tests()
