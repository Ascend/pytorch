# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestFusedAttentionQKV(TestCase):
    def transpose_new(self, x, new_shape):
        return x.npu_confusion_transpose((0, 2, 1, 3), new_shape, False).npu_format_cast(29)

    def npu_op_exec_ori(self, ln_input, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias):
        norm_shape = (1024,)
        norm, mean, variance = torch.native_layer_norm(ln_input, norm_shape, gamma, beta, eps=1e-05)
        q_layer = self.transpose_new(F.linear(norm, q_kernel, q_bias), (24, 512, 16, 64))
        k_layer = self.transpose_new(F.linear(norm, k_kernel, k_bias), (24, 512, 16, 64))
        v_layer = self.transpose_new(F.linear(norm, v_kernel, v_bias), (24, 512, 16, 64))
        return norm.cpu(), mean.cpu(), variance.cpu(), q_layer.cpu(), k_layer.cpu(), v_layer.cpu()

    def npu_op_exec_fused(self, hidden_states, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias):
        seq_len = 512
        num_heads = 16
        norm, q_layer, k_layer, v_layer, mean, variance = torch_npu.npu_fused_attention_layernorm_qkv_fwd(
            hidden_states, q_kernel, k_kernel, v_kernel, gamma, beta, q_bias, k_bias, v_bias, seq_len, num_heads)
        return norm.cpu(), mean.cpu(), variance.cpu(), q_layer.cpu(), k_layer.cpu(), v_layer.cpu()

    def test_fused_attention_ln_qkv_bert_large(self):
        ln_input = torch.rand(12288, 1024).uniform_(-6, 6).half()
        q_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        k_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        v_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        gamma = torch.rand(1024).half()
        beta = torch.rand(1024).half()
        q_bias = torch.rand(1024).half()
        k_bias = torch.rand(1024).half()
        v_bias = torch.rand(1024).half()

        ori_ln_input = ln_input.npu().npu_format_cast(29)
        ori_q_w = q_weight.npu().npu_format_cast(29)
        ori_k_w = k_weight.npu().npu_format_cast(29)
        ori_v_w = v_weight.npu().npu_format_cast(29)
        ori_gamma = gamma.npu()
        ori_beta = beta.npu()
        ori_q_b = q_bias.npu()
        ori_k_b = k_bias.npu()
        ori_v_b = v_bias.npu()

        fused_ln_input = ln_input.npu().npu_format_cast(29)
        fused_q_w = q_weight.npu().t().contiguous().npu_format_cast(29)
        fused_k_w = k_weight.npu().t().contiguous().npu_format_cast(29)
        fused_v_w = v_weight.npu().t().contiguous().npu_format_cast(29)
        fused_gamma = gamma.npu()
        fused_beta = beta.npu()
        fused_q_b = q_bias.npu()
        fused_k_b = k_bias.npu()
        fused_v_b = v_bias.npu()

        ori_norm, ori_mean, ori_variance, ori_q, ori_k, ori_v = self.npu_op_exec_ori(
            ori_ln_input, ori_q_w, ori_k_w, ori_v_w, ori_gamma, ori_beta, ori_q_b, ori_k_b, ori_v_b)
        fused_norm, fused_mean, fused_variance, fused_q, fused_k, fused_v = self.npu_op_exec_fused(
            fused_ln_input, fused_q_w, fused_k_w, fused_v_w, fused_gamma, fused_beta, fused_q_b, fused_k_b, fused_v_b)

        self.assertRtolEqual(ori_norm, fused_norm)
        self.assertRtolEqual(ori_mean, fused_mean)
        self.assertRtolEqual(ori_variance, fused_variance)
        self.assertRtolEqual(ori_q, fused_q, prec16=0.003)
        self.assertRtolEqual(ori_k, fused_k, prec16=0.003)
        self.assertRtolEqual(ori_v, fused_v, prec16=0.003)


if __name__ == '__main__':
    run_tests()
