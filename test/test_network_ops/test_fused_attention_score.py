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

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class FusedAttentionScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_layer, key_layer, value_layer, attention_mask, scale, keep_prob):
        context_layer, softmax_output, dropout_mask = torch_npu.npu_fused_attention_score_fwd(
            query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
        ctx.save_for_backward(query_layer, key_layer, value_layer, softmax_output, dropout_mask)
        ctx.scale = scale
        ctx.keep_prob = keep_prob
        return context_layer

    @staticmethod
    def backward(ctx, grad_output):
        query_layer, key_layer, value_layer, softmax_output, dropout_mask = ctx.saved_tensors
        query_grad, key_grad, value_grad = torch_npu.npu_fused_attention_score_grad(
            grad_output, softmax_output, query_layer, key_layer, value_layer, dropout_mask, ctx.scale, ctx.keep_prob)
        query_grad = query_grad.reshape(
            query_layer.shape[0], query_layer.shape[2], query_layer.shape[1], query_layer.shape[3]).permute(0, 2, 1, 3)
        key_grad = key_grad.reshape(
            query_layer.shape[0], query_layer.shape[2], query_layer.shape[1], query_layer.shape[3]).permute(0, 2, 1, 3)
        value_grad = value_grad.reshape(
            query_layer.shape[0], query_layer.shape[2], query_layer.shape[1], query_layer.shape[3]).permute(0, 2, 1, 3)
        return query_grad, key_grad, value_grad, None, None, None


class TestFusedAttentionScore(TestCase):
    def npu_op_exec_ori(self, query_layer, key_layer, value_layer, attention_mask, scale, keep_prob):
        query_layer.requires_grad = True
        key_layer.requires_grad = True
        value_layer.requires_grad = True

        # attention_score ori forward
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attention_mask + attention_scores * scale
        attn_probss = torch.nn.functional.softmax(attn_scores, dim=-1)
        drop_p = 1 - keep_prob
        drop = torch.nn.DropoutWithByteMask(p=drop_p).npu()
        attn_probs = drop(attn_probss)
        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (query_layer.shape[0] * query_layer.shape[2],
                                   query_layer.shape[1] * query_layer.shape[3])
        context_layer = context_layer.view(new_context_layer_shape)

        # loss
        loss = context_layer.mean() * 1000
        # backward
        loss.backward()

        return context_layer.cpu().detach(), query_layer.grad.cpu(), key_layer.grad.cpu(), value_layer.grad.cpu()

    def npu_fused_op_exec(self, query_layer, key_layer, value_layer, attention_mask, scale, keep_prob):
        query_layer.requires_grad = True
        key_layer.requires_grad = True
        value_layer.requires_grad = True

        # npu_attention_score forward
        context_layer = torch_npu.npu_fused_attention_score(
            query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
        # loss
        loss = context_layer.mean() * 1000
        # backward
        loss.backward()

        return context_layer.detach().cpu(), query_layer.grad.cpu(), key_layer.grad.cpu(), value_layer.grad.cpu()

    def npu_fused_grad_op_exec(self, query_layer, key_layer, value_layer, attention_mask, scale, keep_prob):
        fused_attention_score = FusedAttentionScore.apply
        query_layer.requires_grad = True
        key_layer.requires_grad = True
        value_layer.requires_grad = True

        # npu_attention_score forward
        context_layer = fused_attention_score(
            query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
        # loss
        loss = context_layer.mean() * 1000
        # backward
        loss.backward()

        return context_layer.detach().cpu(), query_layer.grad.cpu(), key_layer.grad.cpu(), value_layer.grad.cpu()

    def test_fused_attention_score_bert_large(self):
        q = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half()
        k = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half()
        v = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half()
        mask = torch.ones(512) * -10000.
        mask[:6] = -0.
        mask = mask.expand(24, 1, 512, 512).half()
        ori_q = q.npu()
        ori_k = k.npu()
        ori_v = v.npu()
        ori_mask = mask.npu()
        fused_q = q.npu()
        fused_k = k.npu()
        fused_v = v.npu()
        fused_mask = mask.npu()
        fused_grad_q = q.npu()
        fused_grad_k = k.npu()
        fused_grad_v = v.npu()
        fused_grad_mask = mask.npu()
        scale = 0.125
        keep_prob = 1

        ori_attention_score, ori_query_dx, ori_key_dw, ori_value_dw = self.npu_op_exec_ori(
            ori_q, ori_k, ori_v, ori_mask, scale, keep_prob)
        fused_attention_score, fused_query_dx, fused_key_dw, fused_value_dw = self.npu_fused_op_exec(
            fused_q, fused_k, fused_v, fused_mask, scale, keep_prob)
        fused_grad_attn_score, fused_grad_query, fused_grad_key, fused_grad_value = self.npu_fused_grad_op_exec(
            fused_grad_q, fused_grad_k, fused_grad_v, fused_grad_mask, scale, keep_prob)

        self.assertRtolEqual(ori_attention_score, fused_attention_score, prec16=0.006)
        self.assertRtolEqual(ori_query_dx, fused_query_dx)
        self.assertRtolEqual(ori_key_dw, fused_key_dw)
        self.assertRtolEqual(ori_value_dw, fused_value_dw)

        self.assertRtolEqual(ori_attention_score, fused_grad_attn_score, prec16=0.006)
        self.assertRtolEqual(ori_query_dx, fused_grad_query)
        self.assertRtolEqual(ori_key_dw, fused_grad_key)
        self.assertRtolEqual(ori_value_dw, fused_grad_value)


if __name__ == '__main__':
    run_tests()
