import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


FORMAT_ND = 2
FORMAT_NZ = 29


class TestMultiHeadAttention(TestCase):
    def matmul_transpose(self, tensor1, tensor2):
        return torch.matmul(tensor1, tensor2.transpose(-2, -1))

    def non_convergence_exec(self, query, key, value, query_weight, key_weight, value_weight, attn_mask,
                             out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, drop_mask, batch,
                             attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float):
        embed_dim = query.size()[-1]
        q = torch_npu.npu_linear(query, query_weight, query_bias)
        k = torch_npu.npu_linear(key, key_weight, key_bias)
        v = torch_npu.npu_linear(value, value_weight, value_bias)
        q *= (attn_dim_per_head ** -0.5)
        new_shape = (batch, tgt_len, attn_head_num, attn_dim_per_head)
        perm = (0, 2, 1, 3)
        if k is not None:
            key_shape = (batch, src_len, attn_head_num, attn_dim_per_head)
        q = torch_npu.npu_confusion_transpose(q, perm, new_shape, False)
        if k is not None:
            k = torch_npu.npu_confusion_transpose(k, perm, new_shape, False)
        if v is not None:
            v = torch_npu.npu_confusion_transpose(v, perm, new_shape, False)
        attn_batch1 = self.matmul_transpose(q, k)

        attn_weights = attn_batch1.view(batch, attn_head_num, tgt_len, src_len)
        attn_adds = attn_weights + attn_mask
        attn_weights_float = F.softmax(attn_adds, dim=-1, dtype=torch.float32)
        attn_softmax = attn_weights_float.to(attn_weights.dtype)
        attn_probs, dropout_mask = torch_npu._npu_dropout(
            attn_softmax, p=dropout_prob)
        attn_batch2 = torch.matmul(attn_probs, v)
        context = torch_npu.npu_confusion_transpose(attn_batch2,
                                                    perm,
                                                    (attn_batch2.size()[0] * attn_batch2.size()[2], embed_dim),
                                                    True)
        attn = torch_npu.npu_linear(context, out_proj_weight, out_proj_bias)

        return attn, dropout_mask, q, k, v, attn_weights_float, attn_probs, context

    def npu_exec(self, query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight,
                 query_bias, key_bias, value_bias, out_proj_bias, drop_mask, batch,
                 attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float):

        return torch_npu.npu_multi_head_attention(
            query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight,
            query_bias, key_bias, value_bias, out_proj_bias, drop_mask,
            attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)

    def test_npu_multi_head_attention(self):
        batch = 8
        attn_head_num = 16
        attn_dim_per_head = 64
        src_len = 64
        tgt_len = 64
        dropout_prob = 0.0
        softmax_use_float = True
        weight_col = attn_head_num * attn_dim_per_head

        _, query = create_common_tensor(["float16", FORMAT_NZ, (batch * tgt_len, weight_col)], -1, 1)
        _, key = create_common_tensor(["float16", FORMAT_NZ, (batch * src_len, weight_col)], -1, 1)
        _, value = create_common_tensor(["float16", FORMAT_NZ, (batch * src_len, weight_col)], -1, 1)
        _, query_weight = create_common_tensor(["float16", FORMAT_NZ, (weight_col, weight_col)], -1, 1)
        _, key_weight = create_common_tensor(["float16", FORMAT_NZ, (weight_col, weight_col)], -1, 1)
        _, value_weight = create_common_tensor(["float16", FORMAT_NZ, (weight_col, weight_col)], -1, 1)
        _, out_proj_weight = create_common_tensor(["float16", FORMAT_NZ, (weight_col, weight_col)], -1, 1)
        _, attn_mask = create_common_tensor(["float16", FORMAT_ND, (batch, attn_head_num, tgt_len, src_len)], -1, 1)
        _, query_bias = create_common_tensor(["float16", FORMAT_ND, (weight_col,)], -1, 1)
        _, key_bias = create_common_tensor(["float16", FORMAT_ND, (weight_col,)], -1, 1)
        _, value_bias = create_common_tensor(["float16", FORMAT_ND, (weight_col,)], -1, 1)
        _, out_proj_bias = create_common_tensor(["float16", FORMAT_ND, (weight_col,)], -1, 1)

        # cpu
        cpu_result, cpu_dropout_mask, cpu_query_res, cpu_key_res, cpu_value_res, cpu_attn_scores, \
            cpu_attn_res, cpu_context = self.non_convergence_exec(
                query, key, value, query_weight, key_weight, value_weight, attn_mask,
                out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, None, batch,
                attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)
        cpu_result = cpu_result.cpu().numpy()
        # npu
        npu_result, npu_dropout_mask, npu_query_res, npu_key_res, npu_value_res, npu_attn_scores, \
            npu_attn_res, npu_context = self.npu_exec(
                query, key, value, query_weight, key_weight, value_weight, attn_mask,
                out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, cpu_dropout_mask, batch,
                attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)
        npu_result = npu_result.cpu().numpy()

        self.assertRtolEqual(cpu_result, npu_result)


if __name__ == "__main__":
    run_tests()
