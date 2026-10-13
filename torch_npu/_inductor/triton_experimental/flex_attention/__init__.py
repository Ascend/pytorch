"""Experimental FlexAttention backend."""

from .flex_attention import _register_npu_inductor_flex_attention, _validate_device, patch_flex_attention
