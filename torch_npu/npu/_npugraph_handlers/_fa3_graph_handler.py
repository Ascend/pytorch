"""FA v3 (npu_fusion_attention_v3) Graph Handlers.

This module defines the NPU Graph operator handlers for the
``npu_fusion_attention_v3`` (forward) and
``npu_fusion_attention_grad_v3`` (backward) operator families.

Structure: ``_FA3TensorListOutHandler`` provides ``postprocess_result`` (return
kwargs["out"]).  ``FA3ForwardHandler`` and ``FA3BackwardHandler`` inherit
it and each implement ``update_args`` and ``prepare_capture``; both
``.default`` and ``.out`` are registered on the same handler class.
"""
__all__ = []

import torch
import torch_npu
from .npugraph_handler import NpuGraphOpHandler, register_npu_graph_handler


class _FA3TensorListOutHandler(NpuGraphOpHandler):
    """Base for FA v3 handlers whose ``out`` kwarg is a ``TensorList``.

    Returns ``kwargs["out"]`` from ``postprocess_result`` so callers get a
    Python list instead of the raw C++ return.
    """

    @classmethod
    def postprocess_result(cls, result, kwargs):
        return kwargs["out"]


# ========================= FA v3 Forward ================================

@register_npu_graph_handler([
    "npu_fusion_attention_v3",
    "npu_fusion_attention_v3.default",
    "npu_fusion_attention_v3.out",
])
class FA3ForwardHandler(_FA3TensorListOutHandler):
    """FA v3 forward: ``.default`` pre-allocates and swaps to ``.out``; ``.out`` passthrough."""

    @classmethod
    def update_args(cls, record, update_input):
        if "actual_seq_qlen" in update_input and len(record.args) > 14:
            record.args[14] = update_input["actual_seq_qlen"]
        if "actual_seq_kvlen" in update_input and len(record.args) > 15:
            record.args[15] = update_input["actual_seq_kvlen"]

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        func_out = torch_npu.npu_fusion_attention_v3.out
        if func is func_out:
            return func, args, kwargs

        # BNSD does not need update; keep original op path.
        input_layout = kwargs.get('input_layout') or (args[4] if len(args) > 4 else None)
        if input_layout == 'BNSD':
            return func, args, kwargs

        # TND + keep_prob in [0,1): not supported for ACLgraph, should fallback via graph partition
        keep_prob = kwargs.get('keep_prob', 1.0) or (args[9] if len(args) > 9 else 1.0)
        if input_layout == 'TND' and 0 <= keep_prob < 1:
            raise RuntimeError(
                "TND layout with dropout (keep_prob < 1) is not supported in ACLgraph mode. "
                "Please use graph partition to fallback this op to eager mode."
            )

        # step2: compute max workspace
        workspace = torch_npu._npu_fusion_attention_v3_get_max_workspace(*args, **kwargs)

        # step3: infer output shapes
        head_num = kwargs.get('head_num') if kwargs else None
        if head_num is None and len(args) > 3:
            head_num = args[3]
        input_layout = kwargs.get('input_layout') if kwargs else None
        if input_layout is None and len(args) > 4:
            input_layout = args[4]
        attention_score, softmax_max, softmax_sum = (
            torch_npu._npu_fusion_attention_v3_infer_output(
                args[0], args[1], args[2], head_num, input_layout)
        )

        # softmax_out shares shape with softmax_max/softmax_sum; seed and offset are 1-element int64
        softmax_out = torch.empty_like(softmax_max)
        seed = torch.empty(1, dtype=torch.int64, device=attention_score.device)
        offset = torch.empty(1, dtype=torch.int64, device=attention_score.device)

        kwargs["workspace"] = workspace
        kwargs["out"] = [attention_score, softmax_max, softmax_sum, softmax_out, seed, offset]
        return func_out, args, kwargs


# ========================= FA v3 Backward ================================

@register_npu_graph_handler([
    "npu_fusion_attention_grad_v3",
    "npu_fusion_attention_grad_v3.default",
    "npu_fusion_attention_grad_v3.out",
])
class FA3BackwardHandler(_FA3TensorListOutHandler):
    """FA v3 backward: ``.default`` pre-allocates and swaps to ``.out``; ``.out`` passthrough."""

    @classmethod
    def update_args(cls, record, update_input):
        if "actual_seq_qlen" in update_input and len(record.args) > 21:
            record.args[21] = update_input["actual_seq_qlen"]
        if "actual_seq_kvlen" in update_input and len(record.args) > 22:
            record.args[22] = update_input["actual_seq_kvlen"]

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        func_out = torch_npu.npu_fusion_attention_grad_v3.out
        if func is func_out:
            return func, args, kwargs

        # BNSD does not need update; keep original op path.
        input_layout = kwargs.get('input_layout') or (args[5] if len(args) > 5 else None)
        if input_layout == 'BNSD':
            return func, args, kwargs

        # TND + keep_prob in [0,1): not supported for ACLgraph, should fallback via graph partition
        keep_prob = kwargs.get('keep_prob', 1.0) or (args[14] if len(args) > 14 else 1.0)
        if input_layout == 'TND' and 0 <= keep_prob < 1:
            raise RuntimeError(
                "TND layout with dropout (keep_prob < 1) is not supported in ACLgraph mode. "
                "Please use graph partition to fallback this op to eager mode."
            )

        # step2: compute max workspace
        workspace = torch_npu._npu_fusion_attention_grad_v3_get_max_workspace(*args, **kwargs)

        # step3: infer output shapes
        pse = kwargs.get('pse') if kwargs else None
        if pse is None and len(args) > 6:
            pse = args[6]
        sink = kwargs.get('sink') if kwargs else None
        if sink is None and len(args) > 27:
            sink = args[27]
        dq, dk, dv, dpse, dsink = (
            torch_npu._npu_fusion_attention_grad_v3_infer_output(
                args[0], args[1], args[2], pse, sink)
        )

        kwargs["workspace"] = workspace
        kwargs["out"] = [dq, dk, dv, dpse, dsink]
        return func_out, args, kwargs
