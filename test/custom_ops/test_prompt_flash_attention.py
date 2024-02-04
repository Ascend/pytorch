import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestPromptFlashAttention(TestCase):

    def supported_op_exec(self, query_states1, past_key, past_value, head_dim):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        attn_weights1 = torch.max(attn_weights1, torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)
        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1

    def custom_op_exec(self, query, key, value, head_dim):
        scale = 1 / 0.0078125
        return torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=32, input_layout="BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535, sparse_mode=0)

    def custom_op_exec_test_quantscale2(self, query, key, value, head_dim):
        scale = 1 / 0.0078125
        deq_scale1 = torch.tensor([1], dtype=torch.float32).npu()
        quant_scale1 = torch.tensor([1], dtype=torch.float32).npu()
        deq_scale2 = torch.tensor([1], dtype=torch.float32).npu()
        quant_scale2 = torch.tensor([1], dtype=torch.float32).npu()
        quant_offset2 = torch.tensor([0], dtype=torch.float32).npu()
        return torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=32, deq_scale1=deq_scale1, quant_scale1=quant_scale1, deq_scale2=deq_scale2, quant_scale2=quant_scale2, quant_offset2=quant_offset2, input_layout="BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535, sparse_mode=0)

    def custom_op_exec_test_int8(self, query, key, value, head_dim):
        scale = 1 / 0.0078125
        deq_scale1 = torch.tensor([1], dtype=torch.float32).npu()
        quant_scale1 = torch.tensor([1], dtype=torch.float32).npu()
        deq_scale2 = torch.tensor([1], dtype=torch.float32).npu()
        quant_scale2 = None
        quant_offset2 = None
        return torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=32, deq_scale1=deq_scale1, quant_scale1=quant_scale1, deq_scale2=deq_scale2, quant_scale2=quant_scale2, quant_offset2=quant_offset2, input_layout="BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535, sparse_mode=0)

    @SupportedDevices(['Ascend910B'])
    def test_npu_prompt_flash_attention(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128

        supported_output = self.supported_op_exec(query, key, value, head_dim)
        custom_output = self.custom_op_exec(query, key, value, head_dim)
        self.assertRtolEqual(supported_output, custom_output)

        query_test2 = torch.randn(1, 32, 2048, 128, dtype=torch.int8).npu()
        key_test2 = torch.randn(1, 32, 2048, 128, dtype=torch.int8).npu()
        value_test2 = torch.randn(1, 32, 2048, 128, dtype=torch.int8).npu()

        supported_output = self.supported_op_exec(query_test2, key_test2, value_test2, head_dim)
        custom_output = self.custom_op_exec_test_quantscale2(query_test2, key_test2, value_test2, head_dim)
        self.assertRtolEqual(supported_output, custom_output)

        supported_output = self.supported_op_exec(query_test2, key_test2, value_test2, head_dim)
        custom_output = self.vcustom_op_exec_test_int8(query_test2, key_test2, value_test2, head_dim)
        self.assertRtolEqual(supported_output, custom_output)

if __name__ == "__main__":
    run_tests()
