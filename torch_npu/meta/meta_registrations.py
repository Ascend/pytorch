import torch
from torch.library import Library, impl

#meta register implementation
m = Library("npu", "IMPL", "Meta")


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0):
    return torch.empty_like(query, dtype=query.dtype)


@impl(m, "npu_prompt_flash_attention")
def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0):
    return torch.empty_like(query, dtype=query.dtype)


@impl(m, "scatter_update")
def npu_scatter_update_forward(data, indices, updates, axis):
    return torch.empty_like(data, dtype=data.dtype)


@impl(m, "npu_rotary_mul")
def npu_rotary_mul_forward(embedding, cosine, sine):
    return torch.empty_like(embedding, dtype=embedding.dtype)


@impl(m, "fast_gelu")
def fast_gelu_meta(self):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_fast_gelu_backward")
def npu_fast_gelu_backward_meta(grad, self):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_fast_gelu")
def npu_fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_dtype_cast")
def npu_dtype_cast_meta(self, dtype):
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
