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
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None,
                                      antiquant_scale=None, antiquant_offset=None, block_table=None, num_heads=1,
                                      scale_value=1.0, input_layout="BSH", num_key_value_heads=0, block_size=0,
                                      inner_precise=1):
    return torch.empty_like(query)


@impl(m, "npu_prompt_flash_attention")
def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
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


@impl(m, "scatter_update")
def scatter_update_meta(self, indices, updates, axis):
    return torch.empty_like(self)


@impl(m, "scatter_update_")
def scatter_update__meta(self, indices, updates, axis):
    return self


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


@impl(m, "npu_rms_norm_backward")
def npu_rms_norm_backward_meta(dy, self, gamma, rstd):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=gamma.dtype))


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
def npu_ffn_meta(x, weight1, weight2, activation, expert_tokens=None, bias1=None, bias2=None, inner_precise=0):
    dim1 = x.size(0)
    dim2 = weight2.size(1)
    return x.new_empty((dim1, dim2))


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
