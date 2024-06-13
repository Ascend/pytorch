import functools
import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error

__all__ = ['npu_fused_attention', 'npu_fused_attention_with_layernorm']


def _exec_once(func):
    @functools.wraps(func)
    def wrapper_exec_once(*args, **kwargs):
        if not wrapper_exec_once.called:
            wrapper_exec_once.called = True
            func(*args, **kwargs)

    wrapper_exec_once.called = False
    return wrapper_exec_once


VALID_FORMAT = [29, 29, 29, 29, 29, 2, 2, 2]


def _is_format_matched(input_list):
    format_list = list(map(torch_npu.get_npu_format, input_list))
    return format_list == VALID_FORMAT


@_exec_once
def _check_compatibility_once(hidden_states,
                             attention_mask,
                             query_kernel,
                             key_kernel,
                             value_kernel,
                             query_bias,
                             key_bias,
                             value_bias,
                             gamma=None,
                             beta=None):
    if not _is_format_matched(
            [hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias]):
        raise RuntimeError(
            'fused attention check compatibility failed, format not matches' + ops_error(ErrCode.VALUE))
    if gamma is not None and beta is not None:
        if torch_npu.get_npu_format(gamma) != 2 or torch_npu.get_npu_format(
                beta) != 2:
            raise RuntimeError(
                'fused attention check compatibility failed, gamma or beta format not matches' +
                ops_error(ErrCode.VALUE)
            )
    if len(hidden_states.size()) != 2 or hidden_states.shape[
        0] % 32 != 0 or hidden_states.shape[1] not in (1024, 768):
        raise RuntimeError(
            'fused attention check compatibility failed, shape of hidden_states not matches' + ops_error(ErrCode.VALUE)
        )
    if len(attention_mask.size()) != 4 or attention_mask.shape[1] != 1 or (
            attention_mask.shape[2] != attention_mask.shape[3]):
        raise RuntimeError(
            'fused attention check compatibility failed, shape of attention_mask not matches' + ops_error(ErrCode.VALUE)
        )
    if query_kernel.shape[0] not in (1024, 768) or key_kernel.shape[0] not in (
            1024, 768) or value_kernel.shape[0] not in (1024, 768):
        raise RuntimeError(
            'fused attention check compatibility failed, shape of kernel not matches' + ops_error(ErrCode.VALUE)
        )


def _permute_with_reshape(x, new_shape):
    return torch_npu.npu_format_cast(torch_npu.npu_confusion_transpose(x,
                                                                       (0, 2, 1, 3),
                                                                       new_shape, False), 29)


class _FusedAttentionWithLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                hidden_states,
                attention_mask,
                query_kernel,
                key_kernel,
                value_kernel,
                query_bias,
                key_bias,
                value_bias,
                gamma,
                beta,
                scale=1,
                keep_prob=0):
        _check_compatibility_once(hidden_states, attention_mask, query_kernel,
                                 key_kernel, value_kernel, query_bias,
                                 key_bias, value_bias, gamma, beta)

        ctx.bsnc = [
            attention_mask.shape[0],
            hidden_states.shape[0] // attention_mask.shape[0],
            hidden_states.shape[1] // 64, 64
        ]

        norm, query_layer, key_layer, value_layer, mean, variance = torch_npu.npu_fused_attention_layernorm_qkv_fwd(
            hidden_states, query_kernel, key_kernel, value_kernel, gamma, beta,
            query_bias, key_bias, value_bias, ctx.bsnc[1], ctx.bsnc[2])

        context_layer, softmax_output, dropout_mask = torch_npu.npu_fused_attention_score_fwd(
            query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)

        ctx.scale = scale
        ctx.keep_prob = keep_prob
        ctx.save_for_backward(query_kernel, key_kernel, value_kernel,
                              query_layer, key_layer, value_layer,
                              hidden_states, softmax_output, dropout_mask,
                              norm, mean, variance, gamma, beta)
        return context_layer, norm

    @staticmethod
    def backward(ctx, grad_output, grad_norm):
        q_w, k_w, v_w, q_l, k_l, v_l, h_s, s_o, d_m, norm, mean, variance, gamma, beta = ctx.saved_variables
        query_grad, key_grad, value_grad = torch_npu.npu_fused_attention_score_grad(
            grad_output, s_o, q_l, k_l, v_l, d_m, ctx.scale, ctx.keep_prob)

        g_h_s, g_w_q, g_w_k, g_w_v, g_b_q, g_b_k, g_b_v = torch_npu.npu_fused_attention_qkv_grad(
            query_grad, key_grad, value_grad, q_w, k_w, v_w, norm,
            torch_npu.npu_format_cast(grad_norm, 29))

        g_h_s, g_gamma, g_beta = torch_npu.npu_layernorm_grad(
            g_h_s, h_s, (g_h_s.shape[1],), mean, variance, gamma, beta)

        return g_h_s, None, g_w_q, g_w_k, g_w_v, g_b_q, g_b_k, g_b_v, g_gamma, g_beta, None, None


npu_fused_attention_with_layernorm = _FusedAttentionWithLayerNorm.apply


class _FusedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                hidden_states,
                attention_mask,
                query_kernel,
                key_kernel,
                value_kernel,
                query_bias,
                key_bias,
                value_bias,
                scale=1,
                keep_prob=0):
        _check_compatibility_once(hidden_states, attention_mask, query_kernel,
                                 key_kernel, value_kernel, query_bias,
                                 key_bias, value_bias, None, None)

        ctx.bsnc = [
            attention_mask.shape[0],
            hidden_states.shape[0] // attention_mask.shape[0],
            hidden_states.shape[1] // 64, 64
        ]

        with torch.no_grad():
            query_layer = _permute_with_reshape(
                torch_npu.npu_linear(hidden_states, query_kernel.t(), query_bias),
                (ctx.bsnc[0], ctx.bsnc[1], ctx.bsnc[2], ctx.bsnc[3]))
            key_layer = _permute_with_reshape(
                torch_npu.npu_linear(hidden_states, key_kernel.t(), key_bias),
                (ctx.bsnc[0], ctx.bsnc[1], ctx.bsnc[2], ctx.bsnc[3]))
            value_layer = _permute_with_reshape(
                torch_npu.npu_linear(hidden_states, value_kernel.t(), value_bias),
                (ctx.bsnc[0], ctx.bsnc[1], ctx.bsnc[2], ctx.bsnc[3]))

        context_layer, softmax_output, dropout_mask = torch_npu.npu_fused_attention_score_fwd(
            query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)

        ctx.scale = scale
        ctx.keep_prob = keep_prob
        ctx.save_for_backward(query_kernel, key_kernel, value_kernel,
                              query_layer, key_layer, value_layer,
                              hidden_states, softmax_output, dropout_mask)
        return context_layer

    @staticmethod
    def backward(ctx, grad_output):
        q_w, k_w, v_w, q_l, k_l, v_l, h_s, s_o, d_m = ctx.saved_variables
        query_grad, key_grad, value_grad = torch_npu.npu_fused_attention_score_grad(
            grad_output, s_o, q_l, k_l, v_l, d_m, ctx.scale, ctx.keep_prob)

        g_h_s, g_w_q, g_w_k, g_w_v, g_b_q, g_b_k, g_b_v = torch_npu.npu_fused_attention_qkv_grad(
            query_grad, key_grad, value_grad, q_w, k_w, v_w, h_s,
            torch_npu.npu_format_cast(torch.zeros_like(h_s), 29))

        return g_h_s, None, g_w_q, g_w_k, g_w_v, g_b_q, g_b_k, g_b_v, None, None


npu_fused_attention = _FusedAttention.apply
