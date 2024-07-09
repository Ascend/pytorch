#

import torch

import torch_npu
from torch_npu.testing._testcase import TestCase, run_tests


class TestFusedAttentionScoreFwd(TestCase):

    def supported_op_exec(self, q, k, v, mask, scale, keep_prob):
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = mask + attention_scores * scale
        attn_probss = torch.nn.functional.softmax(attn_scores, dim=-1)
        drop_p = 1 - keep_prob
        drop = torch.nn.DropoutWithByteMask(p=drop_p).npu()
        attn_probs = drop(attn_probss)
        context_layer = torch.matmul(attn_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (q.shape[0] * q.shape[2],
                                   q.shape[1] * q.shape[3])
        context_layer = context_layer.view(new_context_layer_shape)

        # drop_mask[i] = 1 for inferencing
        drop_mask = torch.ones(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[2])
        drop_mask = drop_mask.to(torch.uint8)

        return context_layer.cpu(), attn_probss.cpu(), drop_mask.cpu()

    def custom_op_exec(self, q, k, v, mask, scale, keep_prob):
        attention_score, softmax_output, drop_mask = torch_npu.npu_fused_attention_score_fwd(q, k, v, mask, scale,
                                                                                             keep_prob)
        return attention_score.cpu(), softmax_output.cpu(), drop_mask.cpu()

    def test_npu_fused_attention_score_fwd(self):
        q = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        k = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        v = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        mask = torch.ones(512) * -10000.
        mask[:6] = -0.
        mask = mask.expand(24, 1, 512, 512).half().npu()
        scale = 0.125
        # keep_prob = 1 for inferencing
        keep_prob = 1

        supported_attention_score, supported_softmax_output, supported_drop_mask = self.supported_op_exec(q, k, v,
                                                                                                          mask,
                                                                                                          scale,
                                                                                                          keep_prob)
        custom_attention_score, custom_softmax_output, custom_drop_mask = self.custom_op_exec(q, k, v, mask, scale,
                                                                                              keep_prob)

        self.assertRtolEqual(supported_attention_score, custom_attention_score, prec16=0.006)
        self.assertRtolEqual(supported_softmax_output, custom_softmax_output)
        self.assertRtolEqual(supported_drop_mask, custom_drop_mask)


if __name__ == '__main__':
    run_tests()
