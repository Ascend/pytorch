import math
import torch
from torch.library import Library, impl

'''
Registering Meta implementations for custom ops
'''
BIT_NUMBER = 128
UINT8_BIT_NUMBER = 8
#meta register implementation
m = Library("npu", "IMPL", "Meta")


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None,
                                      num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0):
    return torch.empty_like(query)


@impl(m, "npu_prompt_flash_attention")
def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
    return torch.empty_like(query, dtype=query.dtype)


@impl(m, "npu_fusion_attention")
def npu_fusion_attention_forward(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    return torch.empty_like(query)


@impl(m, "npu_fusion_attention_grad")
def npu_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    return torch.empty_like(query)


@impl(m, "npu_rotary_mul")
def npu_rotary_mul_meta(embedding, cosine, sine):
    return torch.empty_like(embedding)


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


@impl(m, "npu_scatter_list_")
def scatter_list__meta(self, indices, updates, mask, reduce='update', axis=-2):
    return self


@impl(m, "npu_scatter_list")
def scatter_list_meta(self, indices, updates, mask, reduce='update', axis=-2):
    var_list = []
    for item in self:
        var_list.append(torch.empty_like(item))
    return var_list


@impl(m, "npu_dropout_backward")
def npu_dropout_backward_meta(grad_output, mask, p):
    return torch.empty_like(grad_output, dtype=grad_output.dtype)


@impl(m, "npu_masked_softmax_with_rel_pos_bias")
def npu_masked_softmax_with_rel_pos_bias_meta(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return torch.empty_like(x, dtype=x.dtype)


@impl(m, "npu_ffn")
def npu_ffn_meta(x, weight1, weight2, activation, *, expert_tokens=None, bias1=None, bias2=None, scale=None,
                 offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None, antiquant_scale2=None,
                 antiquant_offset1=None, antiquant_offset2=None, inner_precise=0):
    dim_list = []
    for i in range(0, x.dim() - 1):
        dim_list.append(x.size(i))
    dim_list.append(weight2.size(weight2.dim() - 1))
    if x.dtype == torch.int8:
        return x.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x.new_empty(tuple(dim_list))


@impl(m, "npu_group_norm_silu")
def group_norm_silu_meta(self, gemma, beta, group, eps=0.00001):
    N = self.size(1)
    if gemma is None or beta is None:
        return (torch.empty_like(self, dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype))
    else:
        return (torch.empty_like(self, dtype=self.dtype), gemma.new_empty((N, group), dtype=gemma.dtype), beta.new_empty((N, group), dtype=beta.dtype))


@impl(m, "npu_mm_all_reduce_base")
def npu_mm_all_reduce_base_forward(self, x2, hcom, reduce_op='sum', bias=None, comm_turn=0):
    dim_list = []
    for i in range(self.dim()):
        dim_list.append(self.size(i))
    dim_list[-1] = x2.size(1)
    return self.new_empty(tuple(dim_list))


@impl(m, "npu_weight_quant_batchmatmul")
def npu_weight_quant_batchmatmul_meta(x, weight, antiquant_scale, antiquant_offset=None, quant_offset=None, quant_scale=None, bias=None, antiquant_group_size=0):
    dim_m = x.size(0)
    dim_n = weight.size(1)
    if quant_scale is not None:
        return x.new_empty((dim_m, dim_n), dtype=torch.int8)
    return x.new_empty((dim_m, dim_n), dtype=x.dtype)


@impl(m, "npu_quant_matmul")
def npu_quant_matmul_meta(x1, x2, scale, offset=None, bias=None, output_dtype=None):
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
        cur_batch_val = max(short_dim, long_dim)
        dim_list.append(cur_batch_val)
    dimm = x1.size(x1.dim() - 2)
    dimn = x2.size(x2.dim() - 1)
    dim_list.append(dimm)
    dim_list.append(dimn)
    if output_dtype == "float16":
        return shape_long.new_empty(tuple(dim_list), dtype=torch.float16)
    elif output_dtype == "bfloat16":
        return shape_long.new_empty(tuple(dim_list), dtype=torch.bfloat16)
    return shape_long.new_empty(tuple(dim_list), dtype=torch.int8)


@impl(m, "npu_trans_quant_param")
def npu_trans_quant_param_meta(scale, offset=None):
    dim_max = scale.size(0)
    if offset is not None:
        dim_offset = offset.size(0)
        dim_max = max(dim_max, dim_offset)
    return scale.new_empty((dim_max), dtype=torch.int64)


@impl(m, "npu_quantize")
def npu_quantize_meta(self, scales, zero_points, dtype, axis=1):
    if dtype == torch.quint8:
        return torch.empty_like(self, dtype=torch.uint8)
    elif dtype == torch.qint8:
        return torch.empty_like(self, dtype=torch.int8)
    elif dtype == torch.qint32:
        return torch.empty_like(self, dtype=torch.int32)
    return torch.empty_like(self, dtype=torch.int8)
