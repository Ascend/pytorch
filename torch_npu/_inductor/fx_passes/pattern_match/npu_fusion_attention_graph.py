# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import sympy  # noqa: F401
import torch
import torch.nn.functional as F  # noqa: F401
from torch.autograd import Function
from torch.library import Library, impl
import torch_npu

npu_def = Library("npu_graph", "DEF")
npu_lib = Library("npu_graph", "IMPL", "PrivateUse1")
meta_lib = Library("npu_graph", "IMPL", "Meta")

npu_def.define(
    "npu_fa(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, "
    "Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, "
    "float scale=1., float keep_prob=1., int pre_tockens=2147483647, "
    "int next_tockens=2147483647, int inner_precise=0, int[]? prefix=None, "
    "int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, "
    "bool gen_mask_parallel=True, bool sync=False, str softmax_layout=\"\", "
    "Tensor? sink=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
npu_def.define(
    "npu_fa_backward(Tensor query, Tensor key, Tensor value, Tensor dy, "
    "int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, "
    "Tensor? atten_mask=None, Tensor? softmax_max=None, Tensor? softmax_sum=None, "
    "Tensor? softmax_in=None, Tensor? attention_in=None, float scale_value=1., "
    "float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, "
    "int inner_precise=0, Tensor? seed=None, Tensor? offset=None, Tensor? numels=None, "
    "int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, "
    "int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False, "
    "str softmax_layout=\"\", Tensor? sink=None) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor)")


@impl(npu_lib, "npu_fa")
def npu_fa(*args, **kwargs):
    if len(args) > 8:
        args = list(args)
        # for scale
        try:
            args[8] = 1.0 / args[8]
        except IndexError:
            args[8] = 1.0 / (args[8] + 1e-6)
    r1, r2, r3, r4, seed, offset, numel = torch_npu.npu_fusion_attention(*args, **kwargs)
    r2.requires_grad = False
    r3.requires_grad = False
    r4.requires_grad = False
    return r1, r2, r3, r4, torch.tensor([seed], requires_grad=False), torch.tensor([offset],
                                                                                   requires_grad=False), torch.tensor(
        [numel], requires_grad=False)


@impl(npu_lib, "npu_fa_backward")
def npu_fa_backward(*args, **kwargs):
    if 'scale_value' in kwargs:
        kwargs['scale_value'] = 1.0 / kwargs['scale_value']
    return torch_npu.npu_fusion_attention_grad(*args, **kwargs)


@impl(meta_lib, "npu_fa")
def npu_fa(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
           atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
           inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
           gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    attention_score = torch.empty_like(query, dtype=query.dtype, device='meta').contiguous()
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(attention_score),
            torch.empty_like(softmax_max),
            torch.empty_like(softmax_sum),
            torch.empty_like(softmax_out),
            torch.tensor([0], device='meta', requires_grad=False),
            torch.tensor([0], device='meta', requires_grad=False),
            torch.tensor([0], device='meta', requires_grad=False))


@impl(meta_lib, "npu_fa_backward")
def npu_fa_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                    softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                    keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                    numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                    gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    dq = torch.empty_like(query, dtype=query.dtype, device='meta').contiguous()
    dk = torch.empty_like(key, dtype=query.dtype, device='meta').contiguous()
    dv = torch.empty_like(value, dtype=query.dtype, device='meta').contiguous()
    dpse = torch.empty([0], dtype=query.dtype, device='meta').contiguous()
    dsink = torch.empty([], device='meta') if sink is None else torch.empty_like(sink, dtype=sink.dtype, device='meta').contiguous()
    return (torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dv),
            torch.empty_like(dpse) if pse else None, dsink)


class NpuGraphAttentionFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, scale=1.0,
                keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
        # 前向传播逻辑
        # 这里假设有一个实现前向传播的函数 `npu_fusion_attention_forward`
        result0, result1, result2, result3, result4, result5, result6 = torch.ops.npu_graph.npu_fa(
            query, key, value, head_num, input_layout, pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
            scale=scale, keep_prob=keep_prob, pre_tockens=pre_tockens, next_tockens=next_tockens,
            inner_precise=inner_precise, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode, gen_mask_parallel=gen_mask_parallel, sync=sync
        )
        # 保存中间结果，以便在反向传播中使用
        ctx.save_for_backward(query, key, value, pse, padding_mask, atten_mask, result1, result2, result3, result0,
                              result4, result5, result6)
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        ctx.scale = scale
        ctx.keep_prob = keep_prob
        ctx.pre_tockens = pre_tockens
        ctx.next_tockens = next_tockens
        ctx.inner_precise = inner_precise
        ctx.prefix = prefix
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.sparse_mode = sparse_mode
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync

        return result0, result1, result2, result3, result4, result5, result6

    @staticmethod
    def backward(ctx, grad_result0, grad_result1, grad_result2, grad_result3, grad_result4, grad_result5, grad_result6):
        # 获取保存的中间结果
        (query, key, value, pse, padding_mask, atten_mask,
         result1, result2, result3, result0, result4, result5, result6) = ctx.saved_tensors
        # 反向传播逻辑
        # 这里假设有一个实现反向传播的函数 `npu_fusion_attention_backward`
        grad_query, grad_key, grad_value, grad_pse, grad_sink = torch.ops.npu_graph.npu_fa_backward(
            query, key, value, grad_result0, ctx.head_num, ctx.input_layout, pse=pse, padding_mask=padding_mask,
            atten_mask=atten_mask, softmax_max=result1, softmax_sum=result2, softmax_in=result3, attention_in=result0,
            scale_value=ctx.scale, keep_prob=ctx.keep_prob, pre_tockens=ctx.pre_tockens, next_tockens=ctx.next_tockens,
            inner_precise=ctx.inner_precise, seed=result4, offset=result5, numels=result6, prefix=ctx.prefix,
            actual_seq_qlen=ctx.actual_seq_qlen, actual_seq_kvlen=ctx.actual_seq_kvlen, sparse_mode=ctx.sparse_mode,
            gen_mask_parallel=ctx.gen_mask_parallel, sync=ctx.sync
        )
        return (
            grad_query, grad_key, grad_value, None, None, grad_pse, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None)


def npu_fusion_attention_graph(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                               atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                               next_tockens=2147483647,
                               inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                               gen_mask_parallel=True, sync=False):
    return NpuGraphAttentionFunction.apply(query, key, value, head_num, input_layout, pse, padding_mask,
                                           atten_mask, scale, keep_prob, pre_tockens, next_tockens,
                                           inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                                           gen_mask_parallel, sync)


torch_npu.npu_fusion_attention_graph = npu_fusion_attention_graph
