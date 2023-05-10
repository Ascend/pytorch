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


def transpose_for_scores(x):
    new_x_shape = (24, 512, 16, 64)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3).contiguous()


class FusedAttentionQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias):
        q_layer = transpose_for_scores(F.linear(hidden_states, q_kernel, q_bias))
        k_layer = transpose_for_scores(F.linear(hidden_states, k_kernel, k_bias))
        v_layer = transpose_for_scores(F.linear(hidden_states, v_kernel, v_bias))

        ctx.save_for_backward(hidden_states, q_kernel, k_kernel, v_kernel)

        return q_layer, k_layer, v_layer

    @staticmethod
    def backward(ctx, grad_output_q, grad_output_k, grad_output_v):
        hidden_states, q_kernel, k_kernel, v_kernel = ctx.saved_tensors
        grad_output_q = grad_output_q.permute(0, 2, 1, 3).contiguous().view(12288, 1024).npu_format_cast(29)
        grad_output_k = grad_output_k.permute(0, 2, 1, 3).contiguous().view(12288, 1024).npu_format_cast(29)
        grad_output_v = grad_output_v.permute(0, 2, 1, 3).contiguous().view(12288, 1024).npu_format_cast(29)
        grad_hidden_states, grad_w_q, grad_w_k, grad_w_v, grad_b_q, grad_b_k, grad_b_v = \
            torch_npu.npu_fused_attention_qkv_grad(grad_output_q, grad_output_k, grad_output_v,
                                                   q_kernel, k_kernel, v_kernel, hidden_states,
                                                   torch.zeros_like(hidden_states).npu_format_cast(29))

        return grad_hidden_states, grad_w_q, grad_w_k, grad_w_v, grad_b_q, grad_b_k, grad_b_v


class TestFusedAttentionQKV(TestCase):
    def npu_op_exec_ori(self, hidden_states, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias):
        hidden_states.requires_grad = True
        q_kernel.requires_grad = True
        k_kernel.requires_grad = True
        v_kernel.requires_grad = True
        q_bias.requires_grad = True
        k_bias.requires_grad = True
        v_bias.requires_grad = True

        q_layer = transpose_for_scores(F.linear(hidden_states, q_kernel, q_bias))
        k_layer = transpose_for_scores(F.linear(hidden_states, k_kernel, k_bias))
        v_layer = transpose_for_scores(F.linear(hidden_states, v_kernel, v_bias))

        loss = (q_layer + k_layer + v_layer).mean() * 1000
        loss.backward()
        return hidden_states.grad.cpu(), q_kernel.grad.cpu(), k_kernel.grad.cpu(), v_kernel.grad.cpu(), \
            q_bias.grad.cpu(), k_bias.grad.cpu(), v_bias.grad.cpu()

    def npu_op_exec_fused(self, hidden_states, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias):
        hidden_states.requires_grad = True
        q_kernel.requires_grad = True
        k_kernel.requires_grad = True
        v_kernel.requires_grad = True
        q_bias.requires_grad = True
        k_bias.requires_grad = True
        v_bias.requires_grad = True

        fused_attn_qkv = FusedAttentionQKV.apply
        q_layer, k_layer, v_layer = fused_attn_qkv(
            hidden_states, q_kernel.t(), k_kernel.t(), v_kernel.t(), q_bias, k_bias, v_bias)

        loss = (q_layer + k_layer + v_layer).mean() * 1000
        loss.backward()
        return hidden_states.grad.cpu(), q_kernel.grad.cpu(), k_kernel.grad.cpu(), v_kernel.grad.cpu(), \
            q_bias.grad.cpu(), k_bias.grad.cpu(), v_bias.grad.cpu()

    def test_fused_attention_qkv_grad_bert_large(self):
        hidden_states = torch.rand(24, 512, 1024).uniform_(-5, 5).half()
        q_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        k_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        v_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
        q_bias = torch.rand(1024).half()
        k_bias = torch.rand(1024).half()
        v_bias = torch.rand(1024).half()

        ori_h = hidden_states.npu().view(12288, 1024).npu_format_cast(29)
        ori_q_w = q_weight.npu().npu_format_cast(29)
        ori_k_w = k_weight.npu().npu_format_cast(29)
        ori_v_w = v_weight.npu().npu_format_cast(29)
        ori_q_b = q_bias.npu()
        ori_k_b = k_bias.npu()
        ori_v_b = v_bias.npu()

        fused_h = hidden_states.npu().view(12288, 1024).npu_format_cast(29)
        fused_q_w = q_weight.npu().npu_format_cast(29)
        fused_k_w = k_weight.npu().npu_format_cast(29)
        fused_v_w = v_weight.npu().npu_format_cast(29)
        fused_q_b = q_bias.npu()
        fused_k_b = k_bias.npu()
        fused_v_b = v_bias.npu()

        ori_grad_h, ori_grad_q_w, ori_grad_k_w, ori_grad_v_w, ori_grad_q_b, ori_grad_k_b, ori_grad_v_b = \
            self.npu_op_exec_ori(ori_h, ori_q_w, ori_k_w, ori_v_w, ori_q_b, ori_k_b, ori_v_b)
        fused_grad_h, fused_grad_q_w, fused_grad_k_w, fused_grad_v_w, fused_grad_q_b, fused_grad_k_b, fused_grad_v_b = \
            self.npu_op_exec_fused(fused_h, fused_q_w, fused_k_w, fused_v_w, fused_q_b, fused_k_b, fused_v_b)

        self.assertRtolEqual(ori_grad_h, fused_grad_h)
        self.assertRtolEqual(ori_grad_q_w, fused_grad_q_w)
        self.assertRtolEqual(ori_grad_k_w, fused_grad_k_w)
        self.assertRtolEqual(ori_grad_v_w, fused_grad_v_w)
        self.assertRtolEqual(ori_grad_q_b, fused_grad_q_b)
        self.assertRtolEqual(ori_grad_k_b, fused_grad_k_b)
        self.assertRtolEqual(ori_grad_v_b, fused_grad_v_b)


if __name__ == '__main__':
    run_tests()
