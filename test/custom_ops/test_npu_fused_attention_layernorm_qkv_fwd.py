#
import unittest

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFusedAttentionQKV(TestCase):

    def confusion_transpose(self, x, new_shape):
        return torch_npu.npu_format_cast(x.view(new_shape).permute(0, 2, 1, 3), 29)

    def supported_op_exec(self, ln_input, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias):
        q_kernel = torch_npu.npu_format_cast(q_kernel.t().contiguous(), 29)
        k_kernel = torch_npu.npu_format_cast(k_kernel.t().contiguous(), 29)
        v_kernel = torch_npu.npu_format_cast(v_kernel.t().contiguous(), 29)

        norm_shape = (1024,)
        norm, mean, _ = torch.native_layer_norm(ln_input, norm_shape, gamma, beta, eps=1e-05)
        variance = torch.var(ln_input, -1, keepdim=False, unbiased=False)
        q_layer = self.confusion_transpose(torch.nn.functional.linear(norm, q_kernel, q_bias), (24, 512, 16, 64))
        k_layer = self.confusion_transpose(torch.nn.functional.linear(norm, k_kernel, k_bias), (24, 512, 16, 64))
        v_layer = self.confusion_transpose(torch.nn.functional.linear(norm, v_kernel, v_bias), (24, 512, 16, 64))
        return norm.cpu(), mean.cpu(), variance.cpu(), q_layer.cpu(), k_layer.cpu(), v_layer.cpu()

    def custom_op_exec(self, hidden_states, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias):
        hidden_states = torch_npu.npu_format_cast(hidden_states, 29)
        q_kernel = torch_npu.npu_format_cast(q_kernel, 29)
        k_kernel = torch_npu.npu_format_cast(k_kernel, 29)
        v_kernel = torch_npu.npu_format_cast(v_kernel, 29)
        gamma = torch_npu.npu_format_cast(gamma, 1)
        beta = torch_npu.npu_format_cast(beta, 1)

        seq_len = 512
        num_heads = 16
        norm, q_layer, k_layer, v_layer, mean, variance = torch_npu.npu_fused_attention_layernorm_qkv_fwd(
            hidden_states, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias, seq_len, num_heads)
        return norm.cpu(), mean.cpu(), variance.cpu(), q_layer.cpu(), k_layer.cpu(), v_layer.cpu()

    @unittest.skip("skipped this case")
    def test_npu_fused_attention_layernorm_qkv_fwd(self, device="npu"):
        ln_input = torch.rand(12288, 1024).uniform_(-6, 6).half().npu()
        q_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half().npu()
        k_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half().npu()
        v_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half().npu()
        gamma = torch.rand(1024).half().npu()
        beta = torch.rand(1024).half().npu()
        q_bias = torch.rand(1024).half().npu()
        k_bias = torch.rand(1024).half().npu()
        v_bias = torch.rand(1024).half().npu()

        supported_norm, supported_mean, supported_variance, \
            supported_q, supported_k, supported_v = self.supported_op_exec(
                ln_input, q_weight, k_weight, v_weight, gamma, beta, q_bias, k_bias, v_bias)

        custom_norm, custom_mean, custom_variance, \
            custom_q, custom_k, custom_v = self.custom_op_exec(
                ln_input, q_weight, k_weight, v_weight, gamma, beta, q_bias, k_bias, v_bias)

        self.assertRtolEqual(supported_norm, custom_norm)
        self.assertRtolEqual(supported_mean, custom_mean)
        self.assertRtolEqual(supported_variance, custom_variance)
        self.assertRtolEqual(supported_q, custom_q, prec16=0.003)
        self.assertRtolEqual(supported_k, custom_k, prec16=0.003)
        self.assertRtolEqual(supported_v, custom_v, prec16=0.003)


if __name__ == '__main__':
    run_tests()
