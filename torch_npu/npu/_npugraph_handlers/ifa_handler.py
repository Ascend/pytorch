"""IFA (Infer Fused Attention) v1 / v2 Graph Handlers.

This module defines the NPU Graph operator handlers for the
``npu_fused_infer_attention_score`` (v1) and
``npu_fused_infer_attention_score_v2`` (v2) operator families.

Structure: ``_TensorListOutHandler`` provides ``postprocess_result`` (return
kwargs["out"]).  ``IFAv1DefaultHandler`` and ``IFAv2DefaultHandler`` inherit
it and each implement ``update_args`` and ``prepare_capture``; both
``.default`` and ``.out`` are registered on the same handler class.
"""
__all__ = []

import torch_npu
from .npugraph_handler import NpuGraphOpHandler, register_npu_graph_handler


class _TensorListOutHandler(NpuGraphOpHandler):
    """Base for operators whose ``out`` kwarg is a ``TensorList``.

    Returns ``kwargs["out"]`` from ``postprocess_result`` so callers get a
    Python list instead of the raw C++ return.
    """

    @classmethod
    def postprocess_result(cls, result, kwargs):
        return kwargs["out"]


# ========================= IFA v1 ================================

@register_npu_graph_handler([
    "npu_fused_infer_attention_score",
    "npu_fused_infer_attention_score.default",
    "npu_fused_infer_attention_score.out",
])
class _IFAv1DefaultHandler(_TensorListOutHandler):
    """IFA v1: ``.default`` pre-allocates and swaps to ``.out``; ``.out`` passthrough."""

    @classmethod
    def update_args(cls, record, update_input):
        if "actual_seq_lengths_kv" in update_input and len(record.args) >= 7:
            record.args[6] = update_input["actual_seq_lengths_kv"]

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        func_out = torch_npu.npu_fused_infer_attention_score.out
        if func is func_out:
            return func, args, kwargs

        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            *args, **kwargs
        )
        out_args = [args[0], args[2]]
        out_kwargs_keys = [
            "input_layout",
            "quant_scale2",
            "block_table",
            "num_heads",
            "num_key_value_heads",
            "softmax_lse_flag",
            "query_rope",
        ]
        out_kwargs = {k: kwargs[k] for k in out_kwargs_keys if k in kwargs}
        output, softmax_lse = (
            torch_npu._npu_fused_infer_attention_score_infer_output(
                *out_args, **out_kwargs
            )
        )
        kwargs["workspace"] = workspace
        kwargs["out"] = [output, softmax_lse]
        return func_out, args, kwargs


# ========================= IFA v2 ================================

@register_npu_graph_handler([
    "npu_fused_infer_attention_score_v2",
    "npu_fused_infer_attention_score_v2.default",
    "npu_fused_infer_attention_score_v2.out",
])
class _IFAv2DefaultHandler(_TensorListOutHandler):
    """IFA v2: ``.default`` pre-allocates and swaps to ``.out``; ``.out`` passthrough."""

    @classmethod
    def update_args(cls, record, update_input):
        if "actual_seq_kvlen" in update_input and len(record.args) >= 9:
            record.args[8] = update_input["actual_seq_kvlen"]

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        func_out = torch_npu.npu_fused_infer_attention_score_v2.out
        if func is func_out:
            return func, args, kwargs

        workspace = (
            torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                *args, **kwargs
            )
        )
        out_args = [args[0], args[2]]
        out_kwargs_keys = [
            "query_dtype",
            "value_dtype",
            "input_layout",
            "quant_scale_out",
            "block_table",
            "num_query_heads",
            "num_key_value_heads",
            "return_softmax_lse",
            "query_rope",
            "out_dtype",
        ]
        out_kwargs = {k: kwargs[k] for k in out_kwargs_keys if k in kwargs}
        output, softmax_lse = (
            torch_npu._npu_fused_infer_attention_score_v2_infer_output(
                *out_args, **out_kwargs
            )
        )
        kwargs["workspace"] = workspace
        kwargs["out"] = [output, softmax_lse]
        return func_out, args, kwargs
