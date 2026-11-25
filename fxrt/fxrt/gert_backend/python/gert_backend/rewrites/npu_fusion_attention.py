"""Converter overlay for torch.ops.npu.npu_fusion_attention.

The catalog row launches FlashAttentionScore through aclnnFlashAttentionScore.
aclnn writes softmax_max, softmax_sum, softmax_out, attention_out. Torch getitem 0
is attention_out, so the output hook rebuilds the IR tuple in that order and the
getitem hook remaps the slot. Calls that carry a mask, varlen, dropout or a host
int output stay on torch fallback.
"""

from __future__ import annotations

import operator

import torch

from fxrt.ir import Op
from fxrt import fx_backend as fxb

# pylint: disable=protected-access

_GE_ORDER_FLAG = "_fx_fusion_attention_ge_order"
# Torch slot -> GE slot. Torch is (attention, softmax_max, softmax_sum, softmax_out).
_TORCH_TO_GE = {0: 3, 1: 0, 2: 1, 3: 2}
_LAYOUTS = frozenset({"BSH", "SBH", "BNSD", "BSND"})
_OPTIONAL_SLOTS = (5, 6, 7, 13, 14, 15, 20, 21)
_TAIL = (
    None, None, None,
    1.0, 1.0, 2147483647, 2147483647, 0,
    None, None, None,
    0, True, False, "",
    None, None, 0, 0,
)
_ARG_COUNT = 5 + len(_TAIL)


def _fusion_op():
    return getattr(Op, "npu_fusion_attention", None)


def _runtime(value):
    if isinstance(value, torch.fx.Node):
        return value.meta.get("example_value", None)
    return value


def _pad(flat_args):
    args = list(flat_args)
    if len(args) < 5:
        return None
    if len(args) < _ARG_COUNT:
        args.extend(_TAIL[len(args) - 5:])
    return args[:_ARG_COUNT]


def _is_int(value, expected):
    value = _runtime(value)
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _keep_prob_is_one(value):
    value = _runtime(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return abs(float(value) - 1.0) <= 1e-6


def _layout(value):
    value = _runtime(value)
    if not isinstance(value, str):
        return ""
    return value.upper()


def _optionals_absent(args):
    return all(_runtime(args[index]) is None for index in _OPTIONAL_SLOTS)


def _getitem_index(user):
    if user.target is not operator.getitem or len(user.args) < 2:
        return None
    index = _runtime(user.args[1])
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return index


def _users_are_device_tensors(node):
    for user in node.users:
        index = _getitem_index(user)
        if index is None or index not in _TORCH_TO_GE:
            return False
    return True


def _is_bool(value, expected):
    value = _runtime(value)
    return isinstance(value, bool) and value is expected


def _is_number(value):
    value = _runtime(value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _can_lower(node, args):
    if args is None or not _optionals_absent(args):
        return False
    if _layout(args[4]) not in _LAYOUTS or not _is_positive_int(args[3]):
        return False
    if not _is_number(args[8]) or not _keep_prob_is_one(args[9]):
        return False
    if not _is_int(args[10], 2147483647) or not _is_int(args[11], 2147483647):
        return False
    if not all(_is_int(args[index], 0) for index in (12, 16, 22, 23)):
        return False
    if not _is_bool(args[17], True) or not _is_bool(args[18], False):
        return False
    if _runtime(args[19]) != "":
        return False
    return _users_are_device_tensors(node)


def _is_positive_int(value):
    value = _runtime(value)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def npu_fusion_attention_ops_hook(op, node, flat_args, executor):
    """Keep the catalog row only for the no-mask, keep_prob=1, tensor-getitem path."""
    del executor
    if _can_lower(node, _pad(flat_args)):
        return op
    return Op.custom_call


def npu_fusion_attention_arg_hook(node, flat_args, executor):
    """Reduce the schema to query, key, value, head_num, layout, scale."""
    del node, executor
    args = _pad(flat_args)
    if args is None:
        return list(flat_args)
    head_num = _runtime(args[3]) if isinstance(args[3], torch.fx.Node) else args[3]
    scale = args[8]
    if not isinstance(scale, torch.fx.Node):
        scale = _runtime(scale)
        if isinstance(scale, int) and not isinstance(scale, bool):
            scale = float(scale)
    return [args[0], args[1], args[2], head_num, _layout(args[4]), scale]


def npu_fusion_attention_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Publish softmax_max, softmax_sum, softmax_out, attention_out."""
    example = fxb._get_node_meta_value(node)
    if isinstance(example, (tuple, list)) and len(example) >= 4:
        # This hook runs only after the support check kept the logical op, so the
        # getitem remap below is safe even if a later fallback never sees the flag.
        example = (example[1], example[2], example[3], example[0])
        node.meta[_GE_ORDER_FLAG] = True
    return executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(example))


def npu_fusion_attention_getitem_hook(node, flat_args, executor):
    """Map a torch getitem slot onto the aclnn output order."""
    del node, executor
    if len(flat_args) < 2 or not isinstance(flat_args[0], torch.fx.Node):
        return flat_args
    if not flat_args[0].meta.get(_GE_ORDER_FLAG):
        return flat_args
    mapped = _TORCH_TO_GE.get(_runtime(flat_args[1]))
    if mapped is None:
        return flat_args
    return [flat_args[0], mapped]


def register() -> None:
    """Overlay the torch op and its hooks. No-op until the IR enum exists."""
    op = _fusion_op()
    if op is None:
        return
    packet = getattr(getattr(torch.ops, "npu", None), "npu_fusion_attention", None)
    if packet is not None:
        fxb._OP_MAP[packet] = op
        default = getattr(packet, "default", None)
        if default is not None:
            fxb._OP_MAP[default] = op
    fxb.register_ops_mapping_hook(op, npu_fusion_attention_ops_hook)
    fxb.register_arg_mapping_hook(op, npu_fusion_attention_arg_hook)
    fxb.register_output_mapping_hook(op, npu_fusion_attention_output_hook)
    fxb.register_arg_mapping_hook(Op.tuple_getitem, npu_fusion_attention_getitem_hook)
