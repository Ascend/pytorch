"""Ops-mapping hooks: pick the IR op for an FX node by its input forms.

Slice / narrow retain view semantics for the geometry output hook; a tensor slice
getitem lowers as Op.getitem_slice instead of tuple_getitem. mul/sub/where demote
unsupported forms (bool arithmetic, integer tensor-scalar math, and scalar where
branches) to Op.custom_call, which Convert lowers through
the torch fallback runtime. Anything a hook declines keeps the default op; mul/sub
then delegate to fx_backend's own scalar/tensor dispatch.
"""

from __future__ import annotations

import torch
from torch.fx import Node

from fxrt.gert_backend.rewrites._indexing import getitem_axes
from fxrt.ir import Op
from fxrt import fx_backend as fxb


# This module intentionally reaches into fx_backend internals for the converter-only
# hook overrides, under the same contract as the other rewrites modules.
# pylint: disable=protected-access


# pylint: disable=unused-argument
def slice_ops_hook(op, node, flat_args, executor):
    """Ops-mapping: preserve slice / narrow as view metadata for the edge planner."""
    del node, flat_args, executor
    return op


# pylint: disable=unused-argument
def getitem_ops_hook(op, node, flat_args, executor):
    """Ops-mapping: a tensor slice getitem lowers as getitem_slice, not tuple_getitem."""
    if getitem_axes(node) is None:
        return op
    return Op.getitem_slice


def view_ops_hook(op, node, flat_args, executor):
    """Keep reshape's copy permission distinct from strict view, converter only."""
    del flat_args, executor
    reshape_targets = (torch.reshape, torch.ops.aten.reshape.default, torch.ops.aten.reshape)
    if node.target == "reshape" or node.target in reshape_targets:
        return Op.reshape
    return op


# Path C Muls stores a scalar as a Float CNI attr; integer tensor-scalar
# arithmetic stays on torch fallback to avoid losing integer precision.
_INT_DTYPES = (
    torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
)


def _tensor_dtypes(node: Node):
    """Tensor dtypes of FX node arguments."""
    dtypes = []
    for arg in list(node.args) + list(node.kwargs.values()):
        if not isinstance(arg, Node):
            continue
        example = fxb._get_node_meta_value(arg)
        if isinstance(example, torch.Tensor):
            dtypes.append(example.dtype)
    return dtypes


def _has_integer_tensor(node: Node) -> bool:
    """True when a tensor argument is an integer dtype (bool excluded)."""
    return any(
        dt in _INT_DTYPES and dt != torch.bool
        for dt in _tensor_dtypes(node)
    )


def _scalar_operand(node: Node) -> bool:
    """True when one of the first two arguments is a Python scalar."""
    if len(node.args) < 2:
        return False
    return fxb._is_scalar_arg(node.args[0]) or fxb._is_scalar_arg(node.args[1])


def _mul_sub_ops_hook(op, node, flat_args, executor, default_hook):
    """Bool tensors and integer tensor-scalar math stay on torch fallback."""
    if not fxb._CONVERTER_GRAPH_BUILD:
        return default_hook(op, node, flat_args, executor)
    if any(dt == torch.bool for dt in _tensor_dtypes(node)):
        return Op.custom_call
    if _scalar_operand(node) and _has_integer_tensor(node):
        return Op.custom_call
    return default_hook(op, node, flat_args, executor)


def mul_ops_hook(op, node, flat_args, executor):
    """Integer tensor-scalar mul must not use Path C Muls (Float CNI attr)."""
    return _mul_sub_ops_hook(op, node, flat_args, executor, fxb.mul_op_hook)


def sub_ops_hook(op, node, flat_args, executor):
    """Integer tensor-scalar sub stays on torch fallback."""
    return _mul_sub_ops_hook(op, node, flat_args, executor, fxb.sub_op_hook)


def where_ops_hook(op, node, flat_args, executor):
    """Use SelectV2 for tensor branches, including its supported integer dtypes."""
    del op, flat_args, executor
    # The condition is argument zero. Scalar-branch overloads need their own
    # promotion/materialization contract; do not confuse the condition with x/y.
    if len(node.args) < 3 or any(fxb._is_scalar_arg(arg) for arg in node.args[1:3]):
        return Op.custom_call
    return Op.where


def bitwise_and_ops_hook(op, node, flat_args, executor):
    """Boolean bitwise-and uses LogicalAnd; integer inputs keep BitwiseAnd."""
    del flat_args, executor
    dtypes = _tensor_dtypes(node)
    if dtypes and all(dtype == torch.bool for dtype in dtypes):
        return Op.logical_and
    return op


def gt_ops_hook(op, node, flat_args, executor):
    """Use Greater for scalar comparisons only when its scalar cast preserves ATen semantics."""
    del flat_args, executor
    other = node.args[1] if len(node.args) > 1 else node.kwargs.get("other")
    example = fxb._get_node_meta_value(other) if isinstance(other, Node) else other
    if isinstance(example, torch.Tensor):
        return op
    # Symbolic scalars and mixed integer/float comparisons need ATen promotion.
    # Greater's scalar input otherwise follows the leading tensor dtype.
    if type(other) not in (int, float):
        return Op.custom_call
    source = node.args[0] if node.args else node.kwargs.get("input", node.kwargs.get("self"))
    tensor = fxb._get_node_meta_value(source) if isinstance(source, Node) else source
    if not isinstance(tensor, torch.Tensor):
        return Op.custom_call
    if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return op
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) and type(other) is int:
        limits = torch.iinfo(tensor.dtype)
        if limits.min <= other <= limits.max:
            return op
    return Op.custom_call


def index_ops_hook(op, node, flat_args, executor):
    """Gather handles one index; native Index handles leading integer groups."""
    del flat_args, executor
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (tuple, list)):
        return Op.custom_call
    active = [(axis, item) for axis, item in enumerate(indices) if item is not None]
    if not active or [axis for axis, _ in active] != list(range(len(active))):
        return Op.custom_call
    for _, item in active:
        example = fxb._get_node_meta_value(item)
        if not isinstance(example, torch.Tensor) or example.dtype not in (torch.int32, torch.int64):
            return Op.custom_call
    return op


def _literal_scale(value) -> bool:
    """True when beta/alpha is a Python number an IR float attr can store."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _addmm_scale(node: Node, name: str, positional: int):
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > positional:
        return node.args[positional]
    return 1


def addmm_ops_hook(op, node, flat_args, executor):
    """Keep addmm on Op.addmm when beta and alpha are literal numbers."""
    del flat_args, executor
    beta = _addmm_scale(node, "beta", 3)
    alpha = _addmm_scale(node, "alpha", 4)
    if _literal_scale(beta) and _literal_scale(alpha):
        return op
    return Op.custom_call


def register() -> None:
    """Register the ops-mapping hooks into fx_backend's tables (idempotent).

    Must run after fx_backend's _init_ops_mapping_hooks(): the init re-registers
    its mul/sub defaults with plain assignment on every build.
    """
    fxb.register_ops_mapping_hook(Op.view, view_ops_hook)
    fxb.register_ops_mapping_hook(Op.slice_view, slice_ops_hook)
    fxb.register_ops_mapping_hook(Op.narrow_view, slice_ops_hook)
    fxb.register_ops_mapping_hook(Op.tuple_getitem, getitem_ops_hook)
    fxb.register_ops_mapping_hook(Op.mul, mul_ops_hook)
    fxb.register_ops_mapping_hook(Op.sub, sub_ops_hook)
    fxb.register_ops_mapping_hook(Op.where, where_ops_hook)
    fxb.register_ops_mapping_hook(Op.bitwise_and_tensor, bitwise_and_ops_hook)
    fxb.register_ops_mapping_hook(Op.index_tensor, index_ops_hook)
    fxb.register_ops_mapping_hook(Op.addmm, addmm_ops_hook)
    fxb.register_ops_mapping_hook(Op.gt, gt_ops_hook)
