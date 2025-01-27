import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestIncreFlashAttention(TestCase):

    def supported_op_exec(self, query_states1, past_key, past_value, head_dim, hidden_size):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) * (1.0 / math.sqrt(head_dim))
        attn_weights1 = torch.max(attn_weights1, torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)
        attn_output1 = torch.matmul(attn_weights1, past_value)
        attn_output1 = attn_output1.transpose(1, 2)
        attn_output1 = attn_output1.reshape(1, 1, hidden_size)  # IFA (1, 1, 4096)
        return attn_output1

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    def custom_op_exec(self, query, key, value, head_dim, hidden_size):
        scale = 1.0 / math.sqrt(head_dim)
        return torch_npu.npu_incre_flash_attention(query, key, value, num_heads=32, input_layout="BSH", scale_value=scale)

    @SupportedDevices(['Ascend910B'])
    def test_npu_incre_flash_attention(self, device="npu"):

        q = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        k = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        v = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        q_FA = self.trans_BNSD2BSH(q)
        k_FA = self.trans_BNSD2BSH(k)
        v_FA = self.trans_BNSD2BSH(v)

        head_dim = 128
        hidden_size = 4096

        supported_output = self.supported_op_exec(q, k, v, head_dim, hidden_size)
        custom_output = self.custom_op_exec(q_FA, k_FA, v_FA, head_dim, hidden_size)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
