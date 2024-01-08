import numpy as np
import torch
import torch.nn.functional as F

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


FORMAT_ND = 2
FORMAT_NZ = 29
npu_device = "npu:0"


class MatmulApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat1, mat2):
        ctx.save_for_backward(mat1, mat2)
        result = torch.matmul(mat1, mat2.transpose(-2, -1))
        return result.detach()

    @staticmethod
    def backward(ctx, grad):
        mat1, mat2 = ctx.saved_tensors
        mat1_grad = torch_npu.npu_bmmV2(grad, mat2, [])
        mat2_grad = torch_npu.npu_bmmV2(grad.transpose(-2, -1), mat1, [])
        return mat1_grad, mat2_grad


def Matmul_transpose(tensor1, tensor2):
    return MatmulApply.apply(tensor1, tensor2)


class DropoutApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, prob):
        attn_probs, dropout_mask = torch_npu._npu_dropout(input1, prob)
        ctx.save_for_backward(dropout_mask)
        ctx.prob = prob
        return attn_probs, dropout_mask

    @staticmethod
    def backward(ctx, grad1, grad2):
        prob = ctx.prob
        grad1_nz = torch_npu.npu_format_cast(grad1, FORMAT_NZ)
        mask = ctx.saved_tensors[0]
        attn_probs, dropout_mask = torch_npu.npu_dropout_do_mask(grad1_nz, mask, prob)
        return attn_probs, None


def Dropout(tensor1, prob):
    return DropoutApply.apply(tensor1, prob)


def create_common_tensor(item, minValue, maxValue, need_grad=True):
    dtype1 = item[0]
    format1 = item[1]
    shape1 = item[2]

    input1 = np.random.uniform(minValue, maxValue, shape1).astype(dtype1)
    cpu_input = torch.from_numpy(input1).to(npu_device)
    npu_input = torch.from_numpy(input1).to(npu_device)
    if format1 != -1:
        cpu_input = torch_npu.npu_format_cast(cpu_input, format1)
        npu_input = torch_npu.npu_format_cast(npu_input, format1)
    cpu_input.requires_grad = need_grad
    npu_input.requires_grad = need_grad
    return cpu_input, npu_input


class TestMultiHeadAttention(TestCase):
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
        attn_batch1 = Matmul_transpose(q, k)

        attn_weights = attn_batch1.view(batch, attn_head_num, tgt_len, src_len)
        attn_adds = attn_weights + attn_mask
        attn_adds_nz = torch_npu.npu_format_cast(attn_adds, FORMAT_NZ)
        attn_weights_float = F.softmax(attn_adds_nz, dim=-1, dtype=torch.float32)
        attn_softmax = attn_weights_float.to(attn_weights.dtype)
        attn_probs, dropout_mask = Dropout(attn_softmax, dropout_prob)
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

    def result_equal(self, cpu_result, cpu_query, cpu_grad, npu_grad, cpu_key, cpu_value, cpu_query_weight,
                     cpu_key_weight, cpu_value_weight, cpu_out_proj_weight, cpu_query_bias, cpu_key_bias,
                     cpu_value_bias, cpu_out_proj_bias, npu_result, npu_query, npu_key, npu_value, npu_query_weight,
                     npu_key_weight, npu_value_weight, npu_out_proj_weight, npu_query_bias,
                     npu_key_bias, npu_value_bias, npu_out_proj_bias):
        self.assertRtolEqual(cpu_result.cpu().detach(), npu_result.cpu().detach())
        cpu_result.backward(cpu_grad)
        npu_result.backward(npu_grad)
        self.assertRtolEqual(cpu_query.grad.cpu(), npu_query.grad.cpu())
        self.assertRtolEqual(cpu_key.grad.cpu(), npu_key.grad.cpu())
        self.assertRtolEqual(cpu_value.grad.cpu(), npu_value.grad.cpu())
        self.assertRtolEqual(cpu_query_weight.grad.cpu(), npu_query_weight.grad.cpu())
        self.assertRtolEqual(cpu_key_weight.grad.cpu(), npu_key_weight.grad.cpu())
        self.assertRtolEqual(cpu_value_weight.grad.cpu(), npu_value_weight.grad.cpu())
        self.assertRtolEqual(cpu_out_proj_weight.grad.cpu(), npu_out_proj_weight.grad.cpu())
        self.assertRtolEqual(cpu_query_bias.grad.cpu(), npu_query_bias.grad.cpu())
        self.assertRtolEqual(cpu_key_bias.grad.cpu(), npu_key_bias.grad.cpu())
        self.assertRtolEqual(cpu_value_bias.grad.cpu(), npu_value_bias.grad.cpu())
        self.assertRtolEqual(cpu_out_proj_bias.grad.cpu(), npu_out_proj_bias.grad.cpu())

    def test_mv_out_shape_format(self):

        shape_format = [
            {"batch": 8, "attn_head_num": 16, "attn_dim_per_head": 64, "src_len": 64, "tgt_len": 64,
             "dropout_prob": 0.5, "softmax_use_float": True}]
        for item in shape_format:
            batch = item["batch"]
            attn_head_num = item["attn_head_num"]
            attn_dim_per_head = item["attn_dim_per_head"]
            src_len = item["src_len"]
            tgt_len = item["tgt_len"]
            dropout_prob = item["dropout_prob"]
            softmax_use_float = item["softmax_use_float"]

            weight_col = attn_head_num * attn_dim_per_head
            cpu_query, npu_query = create_common_tensor([np.float16, FORMAT_NZ, (batch * tgt_len, weight_col)], -1, 1)
            cpu_key, npu_key = create_common_tensor([np.float16, FORMAT_NZ, (batch * src_len, weight_col)], -1, 1)
            cpu_value, npu_value = create_common_tensor([np.float16, FORMAT_NZ, (batch * src_len, weight_col)], -1, 1)
            cpu_query_weight, npu_query_weight = create_common_tensor([np.float16, FORMAT_NZ, (weight_col, weight_col)],
                                                                      -1, 1)
            cpu_key_weight, npu_key_weight = create_common_tensor([np.float16, FORMAT_NZ, (weight_col, weight_col)], -1,
                                                                  1)
            cpu_value_weight, npu_value_weight = create_common_tensor([np.float16, FORMAT_NZ, (weight_col, weight_col)],
                                                                      -1, 1)
            cpu_out_proj_weight, npu_out_proj_weight = create_common_tensor(
                [np.float16, FORMAT_NZ, (weight_col, weight_col)], -1, 1)
            cpu_attn_mask, npu_attn_mask = create_common_tensor(
                [np.float16, FORMAT_ND, (batch, attn_head_num, tgt_len, src_len)], -1, 1)
            cpu_query_bias, npu_query_bias = create_common_tensor([np.float16, FORMAT_ND, (weight_col,)], -1, 1)
            cpu_key_bias, npu_key_bias = create_common_tensor([np.float16, FORMAT_ND, (weight_col,)], -1, 1)
            cpu_value_bias, npu_value_bias = create_common_tensor([np.float16, FORMAT_ND, (weight_col,)], -1, 1)
            cpu_out_proj_bias, npu_out_proj_bias = create_common_tensor([np.float16, FORMAT_ND, (weight_col,)], -1, 1)
            cpu_grad, npu_grad = create_common_tensor(
                [np.float16, FORMAT_NZ, (batch * tgt_len, attn_dim_per_head * attn_head_num)], -1, 1)
            cpu_result, cpu_dropout_mask, cpu_query_res, cpu_key_res, cpu_value_res, cpu_attn_scores, \
                cpu_attn_res, cpu_context = self.non_convergence_exec(
                    cpu_query, cpu_key, cpu_value, cpu_query_weight, cpu_key_weight, cpu_value_weight, cpu_attn_mask,
                    cpu_out_proj_weight,
                    cpu_query_bias, cpu_key_bias, cpu_value_bias, cpu_out_proj_bias, None, batch,
                    attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float
                )
            npu_attn_mask.requires_grad_(False)
            npu_result, npu_dropout_mask, npu_query_res, npu_key_res, npu_value_res, npu_attn_scores, \
                npu_attn_res, npu_context = self.npu_exec(
                    npu_query, npu_key, npu_value, npu_query_weight, npu_key_weight, npu_value_weight, npu_attn_mask,
                    npu_out_proj_weight,
                    npu_query_bias, npu_key_bias, npu_value_bias, npu_out_proj_bias, cpu_dropout_mask, batch,
                    attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)
            self.result_equal(cpu_result, cpu_query, cpu_grad, npu_grad, cpu_key, cpu_value, cpu_query_weight,
                              cpu_key_weight, cpu_value_weight, cpu_out_proj_weight, cpu_query_bias, cpu_key_bias,
                              cpu_value_bias, cpu_out_proj_bias, npu_result, npu_query, npu_key, npu_value, npu_query_weight,
                              npu_key_weight, npu_value_weight, npu_out_proj_weight, npu_query_bias,
                              npu_key_bias, npu_value_bias, npu_out_proj_bias)


if __name__ == "__main__":
    run_tests()
