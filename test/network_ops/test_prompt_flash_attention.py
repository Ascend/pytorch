import math
import unittest
import torch
import numpy as np
import torch_npu
import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestPromptFlashAttetion(TestCase):
    def baseline(self, query_states1, past_key, past_value, head_dim):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        attn_weights1 = torch.max(attn_weights1, torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)
        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1

    def prompt_flash_attention_npu(self, q, k, v, head_dim):
        scale = 1 / 0.0078125
        return torch_npu.npu_prompt_flash_attention(
            q, k, v, num_heads=32, input_layout="BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535, sparse_mode=0)

    @SupportedDevices(['Ascend910B'])
    def test_op_exec(self):
        q = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        k = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        v = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        print("input tensor size: q k v")
        print(q.size(), k.size(), v.size())

        head_dim = 128

        pfa_out = self.prompt_flash_attention_npu(q, k, v, head_dim)
        print("PFA output", pfa_out, pfa_out.shape)

        baseline_out = self.baseline(q, k, v, head_dim)
        print("baseline output", baseline_out, baseline_out.shape)

        self.assertRtolEqual(pfa_out, baseline_out)


if __name__ == "__main__":
    run_tests()
