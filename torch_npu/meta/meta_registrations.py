import torch
from torch.library import Library, impl

#meta register implementation
m = Library("npu", "IMPL", "Meta")


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0):
    return torch.empty_like(query, dtype=query.dtype)


@impl(m, "npu_prompt_flash_attention")
# pre_tokens, default value, INT_MAX 2147473647
def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, actual_seq_lengths=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0):
    return torch.empty_like(query, dtype=query.dtype)