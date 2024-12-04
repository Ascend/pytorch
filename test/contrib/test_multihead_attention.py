import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.contrib.module import MultiheadAttention
from torch_npu.contrib.module.multihead_attention import _MHAConfig
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

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


if __name__ == "__main__":
    run_tests()
