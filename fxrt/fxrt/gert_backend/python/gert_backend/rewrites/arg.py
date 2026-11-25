"""Arg-mapping hooks: rewrite FX args into the vectors the IR ops consume.

slice / getitem replace dim/start/end (or the getitem index) with the ``(x, offsets,
size)`` int tuples; constant_pad_nd stores the pad list in GE contiguous order;
expand / view / repeat / as_strided concretize their size/stride lists without
pinning SymInt to the example batch.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
from torch.fx import Node

from fxrt.gert_backend.rewrites._common import (
    concrete_int,
    concretize_int_sequence,
    concretize_shape_dim,
    example_of,
    shape_of,
)
from fxrt.gert_backend.rewrites._indexing import (
    getitem_axes,
    getitem_input_vectors,
    slice_input_vectors,
)
from fxrt.ir import Op
from fxrt import fx_backend as fxb
from fxrt.view_offset import ViewOffsetSemantics, view_offset_semantics

def _pytorch_pad_to_ge(pt_pad: Sequence[int], rank: int) -> List[int]:
    """PyTorch pairs run from the last dim. GE PadV3 wants ``[left0, right0, ...]``."""
    ge_pad = [0] * (rank * 2)
    pair_count = len(pt_pad) // 2
    for pair_i in range(min(pair_count, rank)):
        src = 2 * pair_i
        dim = rank - 1 - pair_i
        ge_pad[2 * dim] = pt_pad[src]
        ge_pad[2 * dim + 1] = pt_pad[src + 1]
    return ge_pad


def _concrete_pad(pad: Any) -> Optional[List[int]]:
    """Return concrete pad widths. A symbolic width declines the rewrite."""
    resolved = example_of(pad)
    if not isinstance(resolved, (list, tuple)):
        return None
    widths: List[int] = []
    for item in resolved:
        width = concrete_int(item)
        if width is None:
            return None
        widths.append(width)
    return widths


# pylint: disable=unused-argument
def shape_list_concretize_arg_hook(node, flat_args, executor):
    """Pass size lists through; static ints stay ints, SymInt stays symbolic."""
    del executor
    if len(flat_args) < 2:
        return flat_args
    args = list(flat_args)
    if node.target in ("expand_as", torch.ops.aten.expand_as.default):
        args[1] = shape_of(node.args[1])
        if args[1] is None:
            raise RuntimeError("expand_as requires target shape metadata")
    args[1] = concretize_int_sequence(args[1])
    return args


# pylint: disable=unused-argument
def as_strided_concretize_arg_hook(node, flat_args, executor):
    """Normalize as_strided size/stride/offset lists without specializing SymInt.

    Static Python ints are kept. Dynamic dims stay SymInt so ShapeEnv is not
    pinned to the example batch. ViewAlias / Path C AsStrided receive concrete
    H/W when those dims are already Python ints.
    """
    del executor
    if len(flat_args) < 3:
        return flat_args
    args = list(flat_args)
    args[1] = concretize_int_sequence(args[1])
    args[2] = concretize_int_sequence(args[2])
    explicit_offset = len(node.args) >= 4 and node.args[3] is not None
    explicit_offset = explicit_offset or node.kwargs.get("storage_offset") is not None
    # -1 is an internal sentinel for ATen's omitted/None offset, which means
    # "use this input's current storage_offset". The runtime resolves it from
    # the current feed TensorV2 instead of freezing the trace example here.
    offset = args[3] if explicit_offset and len(args) >= 4 else -1
    if explicit_offset and len(args) < 4:
        offset = node.kwargs.get("storage_offset")
    offset = concretize_shape_dim(offset)
    concrete_offset = concrete_int(offset)
    if (explicit_offset and concrete_offset is not None and concrete_offset < 0
            and view_offset_semantics(node) == ViewOffsetSemantics.ABSOLUTE):
        raise NotImplementedError("as_strided cannot address storage before the converter feed data_ptr")
    if len(args) >= 4:
        args[3] = offset
    else:
        args.append(offset)
    return args


# pylint: disable=unused-argument
def slice_arg_hook(node, flat_args, executor):
    """Arg-mapping: replace dim/start/end with the offsets and size tuples."""
    vectors = slice_input_vectors(node)
    if vectors is None:
        return flat_args
    return [vectors[0], vectors[1], vectors[2]]


# pylint: disable=unused-argument
def getitem_arg_hook(node, flat_args, executor):
    """Arg-mapping: step 1 replaces the index with offsets and size. A stride keeps the tensor."""
    vectors = getitem_input_vectors(node)
    if vectors is not None:
        return [vectors[0], vectors[1], vectors[2]]
    if getitem_axes(node) is not None:
        return [node.args[0]]
    return flat_args


# pylint: disable=unused-argument
def pad_arg_hook(node, flat_args, executor):
    """Arg-mapping: store the pad list in GE contiguous order."""
    del node, executor
    if len(flat_args) < 2:
        raise RuntimeError("constant_pad_nd rewrite requires a pad list")
    widths = _concrete_pad(flat_args[1])
    shape = shape_of(flat_args[0])
    if widths is None or shape is None:
        raise RuntimeError("constant_pad_nd rewrite requires a tensor and a concrete pad")
    args = list(flat_args)
    args[1] = _pytorch_pad_to_ge(widths, len(shape))
    return args


def _static_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# pylint: disable=unused-argument
def sub_drop_alpha_arg_hook(node, flat_args, executor):
    """Drop a statically verified default alpha; Path C Sub is x1,x2 only."""
    del executor
    alpha = node.kwargs.get("alpha", 1) if hasattr(node, "kwargs") else 1
    if _static_scalar(alpha) != 1.0:
        raise ValueError("Path C Sub requires alpha=1")
    if len(flat_args) >= 3 and _static_scalar(flat_args[2]) == 1.0:
        return list(flat_args[:2])
    return flat_args


# pylint: disable=unused-argument
def amin_full_reduce_arg_hook(node, flat_args, executor):
    """Normalize amin arguments the way fx_backend's amax_arg_hook does for amax.

    PyTorch uses dim=None / dim=[] as all-dimension reduction, while the Op.amin
    row (like Op.amax) consumes the concrete dim list, so expand both forms to
    [0, ..., rank - 1]. A single int dim becomes a one-element list. amin lives
    only in the converter overlay, so fx_backend never registers this for it.
    """
    del executor
    args = list(flat_args)
    if not args:
        raise ValueError("amin requires at least a self operand")
    self_arg = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False
    example = self_arg.meta.get("example_value", None) if isinstance(self_arg, Node) else self_arg
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        rank = int(example.dim()) if hasattr(example, "dim") else 0
        dim = list(range(rank))
    elif isinstance(dim, int):
        dim = [dim]
    return [self_arg, dim, keepdim]


# pylint: disable=unused-argument
def stack_flatten_arg_hook(node, flat_args, executor):
    """Flatten stack's leading tensor list into individual args so the variadic Pack
    collect (AiCore packed-x) sees each tensor as its own IR input instead of one IR
    tuple it cannot unwrap."""
    if not flat_args or not isinstance(flat_args[0], (list, tuple)):
        raise RuntimeError("stack rewrite requires a leading tensor list")
    return list(flat_args[0]) + list(flat_args[1:])


# pylint: disable=unused-argument
def factory_fill_arg_hook(node, flat_args, executor):
    """Normalize zeros/full to Fill's (symbolic dimensions, scalar value) inputs."""
    del executor
    is_full = node.target in (torch.full, torch.ops.aten.full.default)
    fill_value = flat_args[1] if is_full else 0
    # The normalized scalar channel omits bools. Keep their exact 0/1 value as
    # an integer and materialize it using the factory's output dtype.
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    return [flat_args[0], fill_value]


# pylint: disable=unused-argument
def argsort_axis_arg_hook(node, flat_args, executor):
    """Normalize argsort inputs to [input, stable, dim, descending] and gate the axis.

    Converter-overlay only: the classic fx_backend aclnnSort path accepts any axis,
    while the Path C AiCore Sort kernel sorts along the innermost axis (its binaries
    carry no axis-variant key) and bakes the axis into a Sort attr at Convert, so a
    symbolic or non-innermost axis would silently sort the wrong axis.
    """
    del node, executor
    if len(flat_args) == 4:
        normalized = list(flat_args)
    elif len(flat_args) == 3:
        # aten::argsort(Tensor self, int dim=-1, bool descending=False)
        # -> aclnnSort expects stable before dim/descending.
        normalized = [flat_args[0], False, flat_args[1], flat_args[2]]
    else:
        raise ValueError(f"Unsupported argsort input size: {len(flat_args)}")
    dim = normalized[2]
    self_arg = normalized[0]
    if isinstance(dim, Node):
        raise NotImplementedError(
            "torch.argsort with a symbolic dim (e.g. a dim passed as a compiled function "
            "argument under dynamic=True) is not supported on the Path C aicore backend: "
            "the axis is baked into a Sort attr at Convert. Keep the dim a literal in the "
            "traced body, compile with dynamic=False, or mark the dim static."
        )
    example = self_arg.meta.get("example_value", None) if isinstance(self_arg, Node) else None
    if isinstance(dim, int) and example is not None and hasattr(example, "dim"):
        rank = int(example.dim())
        if dim not in (-1, rank - 1):
            raise NotImplementedError(
                f"torch.argsort with dim={dim} is only supported along the innermost axis "
                f"(dim=-1 or dim={rank - 1} for a rank-{rank} input) on the Path C aicore "
                "backend; sort along the last axis or transpose the target axis first."
            )
    return normalized


def clone_arg_hook(node, input_nodes, executor):
    """Keep a contiguous clone's layout requirement in converter-only IR."""
    del executor
    memory_format = node.kwargs.get("memory_format", torch.preserve_format)
    example = fxb._get_node_meta_value(node)
    dense_output = isinstance(example, torch.Tensor) and example.is_contiguous()
    return [input_nodes[0], int(memory_format == torch.contiguous_format or dense_output)]


def register() -> None:
    """Register the arg-mapping hooks into fx_backend's tables (idempotent)."""
    fxb.register_arg_mapping_hook(Op.clone, clone_arg_hook)
    for op in (Op.expand, Op.view, Op.reshape, Op.repeat):
        fxb.register_arg_mapping_hook(op, shape_list_concretize_arg_hook)
    fxb.register_arg_mapping_hook(Op.stack, stack_flatten_arg_hook)
    fxb.register_arg_mapping_hook(Op.zeros, factory_fill_arg_hook)
    fxb.register_arg_mapping_hook(Op.as_strided_view, as_strided_concretize_arg_hook)
    fxb.register_arg_mapping_hook(Op.slice, slice_arg_hook)
    fxb.register_arg_mapping_hook(Op.getitem_slice, getitem_arg_hook)
    fxb.register_arg_mapping_hook(Op.constant_pad_nd, pad_arg_hook)
    # argsort axis validation is a Path C aicore Sort constraint; the classic
    # fx_backend hook normalizes only, since aclnnSort accepts any axis.
    fxb.register_arg_mapping_hook(Op.argsort, argsort_axis_arg_hook)
    # aten.pow.Scalar (scalar base, e.g. 2 ** x) lands on pow_scalar via the aten.pow
    # namespace fallback and would silently compute x ** base; the classic backend
    # never routes it there, so the order guard is converter-overlay only.
    fxb.register_arg_mapping_hook(Op.pow_scalar, fxb.binary_scalar_order_hook)
    # Tensor-tensor sub keeps the alpha=1 gate (a non-default alpha would need a
    # fused mul+sub the Sub opp cannot spell). sub_scalar must NOT be wrapped here:
    # the exporter's add_scalar/sub_scalar collector folds a trailing aten alpha into
    # the materialized scalar, so the alpha arg passes through like add_scalar's.
    fxb.register_arg_mapping_hook(Op.sub, sub_drop_alpha_arg_hook)
    # amin is converter-overlay only (op_map.py); fx_backend registers the equivalent
    # hook for Op.amax itself.
    if hasattr(Op, "amin"):
        fxb.register_arg_mapping_hook(Op.amin, amin_full_reduce_arg_hook)
