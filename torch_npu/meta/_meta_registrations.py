import math
import torch
from torch.library import Library, impl

'''
Registering Meta implementations for custom ops
'''
BIT_NUMBER = 128
UINT8_BIT_NUMBER = 8
INPUTS_DIM_LIMIT_QUANTCONV2D = 4
ATTR_DIM_LIMIT_QUANTCONV2D = 2
#meta register implementation
m = Library("npu", "IMPL", "Meta")


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None,
                                      antiquant_scale=None, antiquant_offset=None, block_table=None,
                                      dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                                      quant_offset2=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH",
                                      num_key_value_heads=0, block_size=0, inner_precise=1):
    if quant_scale2 is not None:
        return torch.empty_like(query, dtype=torch.int8)
    elif query.dtype == torch.int8:
        return torch.empty_like(query, dtype=torch.half)
    else:
        return torch.empty_like(query)


@impl(m, "npu_prompt_flash_attention")
def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
    if quant_scale2 is not None:
        return torch.empty_like(query, dtype=torch.int8)
    elif query.dtype == torch.int8:
        return torch.empty_like(query, dtype=torch.half)
    else:
        return torch.empty_like(query, dtype=query.dtype)


@impl(m, "npu_fusion_attention")
def npu_fusion_attention_forward(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
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

    seed = 0
    offset = 0
    numels = 0
    attention_score = torch.empty_like(query, dtype=query.dtype, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(attention_score),
            torch.empty_like(softmax_max),
            torch.empty_like(softmax_sum),
            torch.empty_like(softmax_out),
            seed,
            offset,
            numels)


@impl(m, "npu_fusion_attention_grad")
def npu_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    dq = torch.empty_like(query, dtype=query.dtype, device='meta')
    dk = torch.empty_like(key, dtype=query.dtype, device='meta')
    dv = torch.empty_like(value, dtype=query.dtype, device='meta')
    dpse = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dv), torch.empty_like(dpse))


@impl(m, "npu_rotary_mul")
def npu_rotary_mul_meta(embedding, cosine, sine):
    return torch.empty_like(embedding)


@impl(m, "npu_rotary_mul_backward")
def npu_rotary_mul_backward(grad, embedding, cosine, sine):
    dx = torch.empty_like(embedding, dtype=embedding.dtype, device='meta')
    dr1 = torch.empty_like(cosine, dtype=embedding.dtype, device='meta')
    dr2 = torch.empty_like(sine, dtype=embedding.dtype, device='meta')
    return (dx, dr1, dr2)


@impl(m, "selu_backward")
def selu_backward_meta(self, result):
    return torch.empty_like(self)


@impl(m, "fast_gelu")
def fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu_backward")
def npu_fast_gelu_backward_meta(grad, self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu")
def npu_fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_dtype_cast")
def npu_dtype_cast_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "npu_dtype_cast_backward")
def npu_dtype_cast_backward_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "npu_bmmV2")
def npu_bmmV2_meta(self, mat2, output_sizes):
    dim1 = self.size(0)
    dim2 = self.size(1)
    dim3 = mat2.size(2)
    return self.new_empty((dim1, dim2, dim3))


@impl(m, "npu_transpose")
def npu_transpose_meta(self, perm, require_contiguous=True):
    output = self.permute(perm)
    return torch.empty_like(output, dtype=self.dtype)


@impl(m, "npu_deep_norm")
def npu_deep_norm_meta(self, gx, beta, gamma, alpha=0.3, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(rstd), torch.empty_like(rstd), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_rms_norm")
def npu_rms_norm_meta(self, gamma, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(rstd))


@impl(m, "npu_add_rms_norm")
def npu_add_rms_norm_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(x1, dtype=x1.dtype), torch.empty_like(rstd), torch.empty_like(x1, dtype=x1.dtype))


@impl(m, "npu_rms_norm_backward")
def npu_rms_norm_backward_meta(dy, self, gamma, rstd):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=gamma.dtype))


@impl(m, "scatter_update")
def scatter_update_meta(self, indices, updates, axis):
    return torch.empty_like(self)


@impl(m, "scatter_update_")
def scatter_update__meta(self, indices, updates, axis):
    return self


@impl(m, "_npu_dropout")
def _npu_dropout_meta(self, p):
    mask = math.floor(math.floor((self.numel() + BIT_NUMBER - 1) / BIT_NUMBER) * BIT_NUMBER / UINT8_BIT_NUMBER)
    return (torch.empty_like(self, dtype=self.dtype), torch.empty(mask, dtype=torch.uint8, device='meta'))


@impl(m, "npu_quant_scatter")
def npu_quant_scatter_meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1,
                           reduce='update'):
    return torch.empty_like(self)


@impl(m, "npu_quant_scatter_")
def npu_quant_scatter__meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1,
                            reduce='update'):
    return self


@impl(m, "npu_scatter_list_")
def scatter_list__meta(self, indices, updates, mask, reduce='update', axis=-2):
    return self


@impl(m, "npu_scatter_list")
def scatter_list_meta(self, indices, updates, mask, reduce='update', axis=-2):
    var_list = []
    for item in self:
        var_list.append(torch.empty_like(item))
    return var_list


@impl(m, "npu_scatter_nd_update")
def scatter_nd_update_meta(self, indices, updates):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_scatter_nd_update_")
def scatter_nd_update__meta(self, indices, updates):
    return self


@impl(m, "npu_geglu")
def npu_geglu_meta(self, dim, approximate, activate_left=False):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_geglu_grad")
def npu_geglu_backward_meta(grad_output, self, gelu, dim, approximate, activate_left=False):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_dropout_backward")
def npu_dropout_backward_meta(grad_output, mask, p):
    return torch.empty_like(grad_output, dtype=grad_output.dtype)


@impl(m, "npu_masked_softmax_with_rel_pos_bias")
def npu_masked_softmax_with_rel_pos_bias_meta(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return torch.empty_like(x, dtype=x.dtype)


@impl(m, "npu_ffn")
def npu_ffn_meta(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None,
                 bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None,
                 antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=0,
                 output_dtype=None):
    dim_list = []
    for i in range(0, x.dim() - 1):
        dim_list.append(x.size(i))
    dim_list.append(weight2.size(weight2.dim() - 1))
    if x.dtype == torch.int8:
        if output_dtype is not None and output_dtype == torch.bfloat16:
            return x.new_empty(tuple(dim_list), dtype=torch.bfloat16)
        else:
            return x.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x.new_empty(tuple(dim_list))


@impl(m, "npu_grouped_matmul")
def npu_grouped_matmul_meta(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None,
                            antiquant_offset=None, group_list=None, split_item=0, output_dtype=None):
    y = []
    num_x = len(x)
    if num_x > 0 and output_dtype is None:
        output_dtype = x[0].dtype
    if split_item == 0:
        for i in range(num_x):
            y.append(x[i].new_empty((*x[i].shape[:-1], weight[i].shape[1]), dtype=output_dtype))
    elif split_item == 1:
        num_group_list = len(group_list)
        y.append(x[0].new_empty((group_list[0], weight[0].shape[1]), dtype=output_dtype))
        for i in range(1, num_group_list):
            y.append(x[0].new_empty((group_list[i] - group_list[i - 1], weight[i].shape[1]), dtype=output_dtype))
    elif split_item == 2:
        dim_m = 0
        for i in range(num_x):
            dim_m += x[i].shape[0]
        y.append(x[0].new_empty((dim_m, weight[0].shape[1]), dtype=output_dtype))
    elif split_item == 3:
        y.append(x[0].new_empty((x[0].shape[0], weight[0].shape[1]), dtype=output_dtype))

    return y


@impl(m, "npu_group_norm_silu")
def group_norm_silu_meta(self, gemma, beta, group, eps=0.00001):
    N = self.size(1)
    if gemma is None or beta is None:
        return (torch.empty_like(self, dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype))
    else:
        return (torch.empty_like(self, dtype=self.dtype), gemma.new_empty((N, group), dtype=gemma.dtype), beta.new_empty((N, group), dtype=beta.dtype))


@impl(m, "npu_mm_all_reduce_base")
def npu_mm_all_reduce_base_forward(x1, x2, hcom, reduce_op='sum', bias=None, antiquant_scale=None,
                                   antiquant_offset=None, x3=None, dequant_scale=None, antiquant_group_size=0,
                                   comm_turn=0):
    dim_list = []
    for i in range(x1.dim()):
        dim_list.append(x1.size(i))
    dim_list[-1] = x2.size(1)
    if dequant_scale is not None:
        if dequant_scale.dtype == torch.bfloat16:
            return x1.new_empty(tuple(dim_list), dtype=torch.bfloat16)
        else:
            return x1.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x1.new_empty(tuple(dim_list))



@impl(m, "npu_weight_quant_batchmatmul")
def npu_weight_quant_batchmatmul_meta(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0):
    dim_m = x.size(0)
    dim_n = weight.size(1)
    if quant_scale is not None:
        return x.new_empty((dim_m, dim_n), dtype=torch.int8)
    return x.new_empty((dim_m, dim_n), dtype=x.dtype)


def bias_shape_check(x2, bias, batch_val):
    bias_dim_num = bias.dim()
    torch._check(
        bias_dim_num == 1 or bias_dim_num == 3,
        lambda: "bias_dim_num should be 1 or 3, please check bias dim num ",
    )
    x2_dim_num = x2.dim()
    x2_n_dim = x2.size(x2_dim_num - 1)
    bias_first_dim = bias.size(0)
    if bias_dim_num == 1:
        torch._check(
            bias_first_dim == x2_n_dim,
            lambda: "bias_first_dim should be equal to x2 n dim, please check bias 1st dim value ",
        )
        return
    bias_second_dim = bias.size(1)
    bias_third_dim = bias.size(2)
    torch._check(
        bias_first_dim == batch_val,
        lambda: "infered batch value should be equal to bias batch dim value, please check bias batch dim value",
    )
    torch._check(
        bias_second_dim == 1,
        lambda: "bias_second_dim should be 1, please check bias second dim value ",
    )
    torch._check(
        bias_third_dim == x2_n_dim,
        lambda: "bias_third_dim should be equal to x2_n_dim, please check bias third dim value ",
    )


def quant_matmul_shape_check(x1, x2, scale, offset, pertoken_scale):
    X_MAX_DIM = 6
    X_MIN_DIM = 2
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    x1_m_dim = x1.size(x1_dim_num - 2)
    x1_k_dim = x1.size(x1_dim_num - 1)
    x2_k_dim = x2.size(x2_dim_num - 2)
    x2_n_dim = x2.size(x2_dim_num - 1)
    torch._check(
        x1_dim_num >= X_MIN_DIM and x1_dim_num <= X_MAX_DIM,
        lambda: "x1 dim num should be 2 ~ 6, please check x1 dim num",
    )
    torch._check(
        x2_dim_num >= X_MIN_DIM and x2_dim_num <= X_MAX_DIM,
        lambda: "x2 dim num should be 2 ~ 6, please check x2 dim num",
    )
    torch._check(
        x1_k_dim == x2_k_dim,
        lambda: "k dim of x1 and x2 need be same, please check k dim of x1 and x2",
    )
    if offset is not None:
        offset_dim_num = offset.dim()
        torch._check(
            offset_dim_num == 1,
            lambda: "the offset dim num must be 1, please check offset dim num ",
        )
        offset_first_dim = offset.size(0)
        torch._check(
            offset_first_dim == 1 or offset_first_dim == x2_n_dim,
            lambda: "the offset 1st dim value must be 1 or x2 n dim value, please check offset 1st dim value",
        )
    if pertoken_scale is not None:
        pertoken_scale_dim_num = pertoken_scale.dim()
        torch._check(
            pertoken_scale_dim_num == 1,
            lambda: "the pertoken_scale dim num must be 1, please check scale dim num",
        )
        pertoken_scale_first_dim = pertoken_scale.size(0)
        torch._check(
            pertoken_scale_first_dim == x1_m_dim,
            lambda: "the pertoken_scale 1st dim value must be x1 m dim value, please check scale 1st dim value ",
        )

    scale_dim_num = scale.dim()
    torch._check(
        scale_dim_num == 1,
        lambda: "the scale dim num must be 1, please check scale dim num",
    )
    scale_first_dim = scale.size(0)
    torch._check(
        scale_first_dim == 1 or scale_first_dim == x2_n_dim,
        lambda: "the scale 1st dim value must be 1 or x2 n dim value, please check scale 1st dim value ",
    )


def quant_matmul_dtype_check(*args):
    x1, x2, scale, offset, pertoken_scale, bias = args
    torch._check(
        x1.dtype == torch.int8 and x2.dtype == torch.int8,
        lambda: "x1'type and x2's type should be int8, but x1.dtype is " + str(x1.dtype) + " and x2.dtype is " +
                str(x2.dtype),
    )
    scale_dtype_supported_lst = [torch.float32, torch.int64, torch.bfloat16]
    torch._check(
        scale.dtype in scale_dtype_supported_lst,
        lambda: "scale's type supported for float32, int64 and bfloat16, but scale.dtype is " + str(scale.dtype),
    )
    if offset is not None:
        torch._check(
            offset.dtype == torch.float32,
            lambda: "offset's type supported for float32, but offset.dtype is " + str(offset.dtype),
        )
    if pertoken_scale is not None:
        torch._check(
            pertoken_scale.dtype == torch.float32,
            lambda: "pertoken_scale's type supported for float32, but pertoken_scale.dtype is " + str(offset.dtype),
        )
    if bias is not None:
        torch._check(
            bias.dtype == torch.int32 or bias.dtype == torch.bfloat16,
            lambda: "bias's type supported for int32 and bfloat16, but bias.dtype is " + str(bias.dtype),
        )


def quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype):
    if scale.dtype == torch.bfloat16:
        torch._check(
            output_dtype == torch.bfloat16,
            lambda: "When scale's dtype is bfloat16, output_dtype must be bfloat16, but output_dtype is " + str(output_dtype),
        )
    if output_dtype == torch.bfloat16:
        torch._check(
            scale.dtype == torch.bfloat16,
            lambda: "When output_dtype is bfloat16, scale's dtype must be bfloat16, but scale's dtype is " +
            str(scale.dtype),
        )
    if offset is not None:
        torch._check(
            output_dtype is None or output_dtype == torch.int8,
            lambda: "offset only exists when output_dtype is int8, but output_dtype is " + str(output_dtype),
        )
    if pertoken_scale is not None:
        if output_dtype == torch.float16:
            torch._check(
                scale.dtype == torch.float32,
                lambda: "When output_dtype is float16 and pertoken_scale is not none, scale's dtype must be float32, but scale's dtype is " +
                str(scale.dtype),
            )
        torch._check(
            output_dtype == torch.float16 or output_dtype == torch.bfloat16,
            lambda: "When pertoken_scale is not none, output_dtype must be float16 or bfloat16, but output_dtype is " + str(output_dtype),
        )


@impl(m, "npu_quant_matmul")
def npu_quant_matmul_meta(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None):
    batch_val = 1
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    out_dim_num = max(x1_dim_num, x2_dim_num)
    shape_long = x1 if x1_dim_num > x2_dim_num else x2
    shape_short = x2 if x1_dim_num > x2_dim_num else x1
    vaild_offset = out_dim_num - min(x1_dim_num, x2_dim_num)
    dim_list = []
    for i in range(0, out_dim_num - 2):
        short_dim = 1 if i < vaild_offset else shape_short.size(i - vaild_offset)
        long_dim = shape_long.size(i)
        torch._check(
            not (short_dim > 1 and long_dim > 1 and short_dim != long_dim),
            lambda: "the batch shape cannnot be boardcast",
        )
        cur_batch_val = max(short_dim, long_dim)
        batch_val = batch_val * cur_batch_val
        dim_list.append(cur_batch_val)
    dimm = x1.size(x1.dim() - 2)
    dimn = x2.size(x2.dim() - 1)
    dim_list.append(dimm)
    dim_list.append(dimn)
    quant_matmul_shape_check(x1, x2, scale, offset, pertoken_scale)
    if bias is not None:
        if bias.dtype == torch.bfloat16:
            torch._check(
                output_dtype == torch.bfloat16,
                lambda: "When bias dtype is bfloat16, output_dtype must be bfloat16, but it is " + str(output_dtype),
            )
        if bias.dim() == 3:
            torch._check(
                len(dim_list) == 3,
                lambda:"when bias dim is 3, out dim need to be 3",
            )
        bias_shape_check(x2, bias, batch_val)
    quant_matmul_dtype_check(x1, x2, scale, offset, pertoken_scale, bias)
    quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype)
    if output_dtype == torch.float16:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.float16)
    elif output_dtype == torch.bfloat16:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.bfloat16)
    elif output_dtype is None or output_dtype == torch.int8:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.int8)
    else:
        raise RuntimeError("Not supportted output dtype is " + str(output_dtype))


@impl(m, "npu_trans_quant_param")
def npu_trans_quant_param_meta(scale, offset=None):
    scale_dim_num = scale.dim()
    torch._check(
        scale_dim_num == 1 or (scale_dim_num == 2 and scale.size(0) == 1),
        lambda: "the scale shape support only (1, ) and (1, n)",
    )
    output_shape = scale.size()
    if scale_dim_num == 1:
        scale_first_dim = scale.size(0)
        dim_max = scale_first_dim
        if offset is not None:
            offset_first_dim = offset.size(0)
            dim_max = max(dim_max, offset_first_dim)
            if offset_first_dim != 1 and scale_first_dim != 1:
                torch._check(
                    offset_first_dim == scale_first_dim,
                    lambda: "offset first dim should be equal to scale first dim if none of them are equal to one",
                )
        output_shape = (dim_max)
    else:
        if offset is not None:
            torch._check(
                scale.size() == offset.size(),
                lambda: "when the input shape of scale is (1, n), shape of scale and offset should be equal",
            )
    return scale.new_empty(output_shape, dtype=torch.int64)


@impl(m, "npu_quantize")
def npu_quantize_meta(self, scales, zero_points, dtype, axis=1, div_mode=True):
    if dtype == torch.quint8:
        return torch.empty_like(self, dtype=torch.uint8)
    elif dtype == torch.qint8:
        return torch.empty_like(self, dtype=torch.int8)
    elif dtype == torch.qint32:
        return torch.empty_like(self, dtype=torch.int32)
    return torch.empty_like(self, dtype=torch.int8)


@impl(m, "npu_dynamic_quant")
def npu_dynamic_quant(input_dummy, *, smooth_scales=None):
    return (torch.empty_like(input_dummy, dtype=torch.int8),
             torch.empty_like(input_dummy.sum(dim=-1), dtype=torch.float32))


@impl(m, "npu_anti_quant")
def npu_anti_quant_meta(x, scale, *, offset=None, dst_dtype=None, src_dtype=None):
    if dst_dtype is None:
        return torch.empty_like(x, dtype=torch.float16)
    return torch.empty_like(x, dtype=dst_dtype)


@impl(m, "npu_apply_rotary_pos_emb")
def npu_apply_rotary_pos_emb_meta(query, key, cos, sin, layout=1):
    return (torch.empty_like(query, dtype=query.dtype), torch.empty_like(key, dtype=key.dtype))


@impl(m, "npu_quant_conv2d")
def npu_quant_conv2d(input_, weight, scale, strides, pads, dilations,
                     groups=1, offset_x=0, round_mode='rint', output_dtype=None, bias=None, offset=None):

    input_shape = input_.size()
    weight_shape = weight.size()
    scale_shape = scale.size()

    input_dim = input_.dim()
    weight_dim = weight.dim()
    scale_dim = scale.dim()

    def check_basic_inputs_dim_shape():

        torch._check(
            input_dim == weight_dim and weight_dim == INPUTS_DIM_LIMIT_QUANTCONV2D,
            lambda: "input dim or weight dim is not equal to 4, but now input dim is " + str(input_dim) + ", and weight dim is "
                     + str(weight_dim),
        )

        torch._check(
            scale_dim == 1,
            lambda: "scale dim is not equal to 1, but now scale dim is " + str(scale_dim),
        )

        torch._check(
            input_shape[1] == weight_shape[1],
            lambda: "input cin should equal to weight cin, but now input cin is " + str(input_shape[1]) + ", and weight cin is "
                    + str(weight_shape[1]),
        )

        torch._check(
            scale_shape[0] == weight_shape[0],
            lambda: "scale shape should equal to cout, but now scale shape is " + str(scale_shape[0]) + ", and cout is " +
                    str(weight_shape[0]),
        )

    def check_basic_inputs_dtype():
        torch._check(
            input_.dtype == torch.int8 and weight.dtype == torch.int8,
            lambda: "input's dtype and weight's dtype should be int8, but input.dtype is " + str(input_.dtype) + ", and weight.dtype is " +
                    str(weight.dtype),
        )

        torch._check(
            scale.dtype == torch.int64,
            lambda: "scale's dtype should be int64, but scale.dtype is " + str(scale.dtype),
        )

        torch._check(
            output_dtype == torch.float16,
            lambda: "output dtype should be float16, but now dtype is " + str(output_dtype),
        )

    def check_bias_dim_shape_dtype():
        bias_dim = bias.dim()
        bias_shape = bias.size()
        torch._check(
            bias_dim == 1,
            lambda: "bias dim is not equal to 1, but now bias dim is " + str(bias_dim),
        )

        torch._check(
            bias.dtype == torch.int32,
            lambda: "bias' dtype should be int32, but bias.dtype is " + str(input_.dtype),
        )

        torch._check(
            bias_shape[0] == weight_shape[0],
            lambda: "bias shape should equal to cout, but now bias shape is " + str(bias_shape[0]) + ", and cout is " +
                    str(weight_shape[0]),
        )

    def check_attrs():
        pads_dim = len(pads)
        strides_dim = len(strides)
        dilations_dim = len(dilations)
        torch._check(
            pads_dim == ATTR_DIM_LIMIT_QUANTCONV2D and strides_dim == ATTR_DIM_LIMIT_QUANTCONV2D and
            dilations_dim == ATTR_DIM_LIMIT_QUANTCONV2D,
            lambda: "attrs's dim should be 2, but pads dim is " + str(pads_dim) + ", strides dim is "
                    + str(strides_dim) + ", dilations dim is " + str(dilations_dim),
        )
        torch._check(
            pads[0] >= 0 and pads[1] >= 0,
            lambda: "pads's value should large or equal to 0, but pads is " + str(pads[0]) + ", "
                    + str(pads[1]),
        )
        torch._check(
            strides[0] > 0 and strides[1] > 0,
            lambda: "strides's value should large than 0, but strides is " + str(strides[0]) + ", "
                    + str(strides[1]),
        )
        torch._check(
            dilations[0] > 0 and dilations[1] > 0,
            lambda: "dilations's value should large than 0, but dilations is " + str(dilations[0]) + ", "
                    + str(dilations[1]),
        )
        torch._check(
            groups == 1,
            lambda: "groups should be 1, but now " + str(groups),
        )
        torch._check(
            offset_x <= 127 and offset_x >= -128,
            lambda: "offset_x should be [-128,127], but offset_x is " + str(offset_x),
        )
        torch._check(
            round_mode == 'rint',
            lambda: "round_mode should be rint, but round_mode is " + str(round_mode),
        )

    check_basic_inputs_dim_shape()
    check_basic_inputs_dtype()
    if bias is not None:
        check_bias_dim_shape_dtype()
    check_attrs()

    nout = input_shape[0]
    cout = weight_shape[0]
    hout = (input_shape[2] + pads[0] * 2 - dilations[0] * (weight_shape[2] - 1) - 1) // strides[0] + 1
    wout = (input_shape[3] + pads[1] * 2 - dilations[1] * (weight_shape[3] - 1) - 1) // strides[1] + 1

    torch._check(
        hout > 0 and wout > 0,
        lambda: "ho, wo should larger than 0, but now ho is " + str(hout) + ", and wo is " + str(wout),
    )

    output_dim_list = [nout, cout, hout, wout]

    return scale.new_empty(tuple(output_dim_list), dtype=output_dtype)


@impl(m, "npu_linear")
def npu_linear_meta(input_, weight, bias=None):
    dimm = input_.size(0)
    dimn = weight.size(0)
    return input_.new_empty((dimm, dimn))


@impl(m, "npu_moe_finalize_routing")
def npu_moe_finalize_routing_meta(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row,
                                  expert_for_source_row):
    return torch.empty_like(skip1, dtype=skip1.dtype)
