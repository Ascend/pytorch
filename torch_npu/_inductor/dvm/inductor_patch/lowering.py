# mypy: allow-untyped-defs
"""Shared traced-graph lowering for NPU backends.

Backend loaders install this module through lowering_patch. Backend-specific
policies are supplied by those loaders; this module must not import a backend
package, including through lazy imports, because that can install global patches.
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
import os
from typing import Any, Callable, List, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec

import sympy

import torch
import torch.ao.quantization.fx._decomposed
import torch.fx
import torch.utils._pytree as pytree
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    get_computation_dtype,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    Identity,
    ModularIndexing,
)

from torch._inductor import config, ir, test_operators  # NOQA: F401
from torch._inductor.decomposition import decompositions, get_decompositions
from torch._inductor.ir import (
    ExpandView,
    IRNode,
    is_triton,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    TensorBox,
    validate_ir,
    View,
)
# Import helpers used by local lowerings. The shared installer inherits missing
# operator registrations from the active PyTorch registry.
from torch._inductor.lowering import (
    _make_reduction_inner,
    is_boolean_type,
    is_integer_type,
    squeeze,
    transform_args,
    make_foreach_pointwise,
    sdpa_constraint,
    constant_like,
    create_tensor_like,
    full_like,
    _unwrap,
    _validate_dim,
    _validate_reduction_axis,
    assert_nyi,
    broadcast_tensors,
    constrain_to_fx_strides,
    decode_dtype,
    div,
    get_promoted_dtype,
    in_namespace,
    lowerings as _upstream_lowerings,
    mean,
    pow_recursive,
    promote_constants,
    require_channels_last,
    require_contiguous,
    require_dense,
)
from torch._inductor.utils import (
    decode_device,
    is_pointwise_use,
    register_op_dtype_propagation_rules,
    sympy_product,
)
from torch._inductor.virtualized import ops, V


if TYPE_CHECKING:
    from torch._inductor.ops_handler import ReductionType


_T = TypeVar("_T")
_P = ParamSpec("_P")

log = logging.getLogger(__name__)
lowerings: dict[Union[Callable[..., Any], str], Callable[..., Any]] = {}
# Use maybe_layout_constraints to access this dict, we lazily register tag-based layout constraints
_maybe_layout_constraints: dict[
    torch._ops.OpOverload, Optional[Callable[..., Any]]
] = {}
fallbacks = OrderedSet[torch._ops.OpOverload]()
aten = torch.ops.aten
prims = torch.ops.prims
needs_realized_inputs = OrderedSet[torch._ops.OpOverload]()
foreach_ops = OrderedSet[torch._ops.OpOverload](
    [torch._higher_order_ops._foreach_map]  # type: ignore[list-item]
)
# TODO(rec): torch._higher_order_ops._foreach_map is not an OpOverload
# so why is it in foreach_ops?
inplace_foreach_ops = OrderedSet[torch._ops.OpOverload]()
inplaceable_foreach_ops: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}

from torch_npu._inductor.lowering_common import (
    TracedGraph,
    MLIR_OPERATOR_MAPPING,
    fetch_graphs as _fetch_graphs,
    get_reduction_type_to_aten_fn,
    merge_traced_graphs as _merge_traced_graphs,
    register_fn_to_aten_fn as _register_fn_to_aten_fn,
    register_to_aten as _register_to_aten,
)

# Backend loaders may supply a sum-fallback predicate without importing their
# private configuration here.
sum_fallback_predicate: Callable[[TensorBox, Any], bool] | None = None

fn_to_aten_fn = {}
node_id = itertools.count(0)

def register_fn_to_aten_fn(fn, aten_fn=None):
    return _register_fn_to_aten_fn(fn_to_aten_fn, fn, aten_fn)

def register_to_aten(aten_fn=None):
    return _register_to_aten(fn_to_aten_fn, aten_fn)

reduction_type_to_aten_fn = get_reduction_type_to_aten_fn()

def fetch_graphs(inputs: Optional[List[TensorBox]]):
    return _fetch_graphs(inputs, MLIR_OPERATOR_MAPPING, use_npu_meta=False)

def merge_traced_graphs(input_graphs: List[TracedGraph], origin_fn, node_name, **kwargs):
    return _merge_traced_graphs(
        input_graphs, origin_fn, node_name, MLIR_OPERATOR_MAPPING, **kwargs
    )


# group by device, whether any of the inputs are dynamic
# note arg_pairs may or may not be a pair
# foreach_map for example just passes output buffers here
# The installer binds this by name for the upstream GraphLowering.call_function.
def maybe_layout_constraints(fn: Callable[..., Any]) -> Optional[Callable[..., Any]]:
    """Get layout constraints. Returns None if there are no layout constraints."""
    if not isinstance(fn, torch._ops.OpOverload):
        # Only OpOverloads have layout constraints.
        return None
    if fn in _maybe_layout_constraints:
        return _maybe_layout_constraints[fn]
    # OpOverload with custom lowerings override tag-based layout constraints
    if fn in lowerings:
        _maybe_layout_constraints[fn] = None
        return None
    # We lazily register tag-based layout constraints.

    def handle_layout_constraint_tag(tag):
        if tag is torch._C.Tag.needs_fixed_stride_order:
            _maybe_layout_constraints[fn] = constrain_to_fx_strides
            return _maybe_layout_constraints[fn]
        elif tag is torch._C.Tag.flexible_layout:
            _maybe_layout_constraints[fn] = None
            return None
        else:
            raise AssertionError(f"Unknown layout constraint tag: {tag}")

    tag = get_layout_constraint_tag(fn)
    return handle_layout_constraint_tag(tag)


def get_layout_constraint_tag(fn):
    tags_by_priority = [
        torch._C.Tag.needs_fixed_stride_order,
        torch._C.Tag.flexible_layout,
    ]
    for tag in tags_by_priority:
        if tag in fn.tags:
            return tag
    if torch._library.utils.is_builtin(fn):
        return torch._C.Tag.flexible_layout
    return getattr(torch._C.Tag, config.custom_op_default_layout_constraint)


def add_needs_realized_inputs(fn):
    if isinstance(fn, (list, set, tuple, OrderedSet)):  # noqa: set_linter
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)
    if isinstance(fn, torch._ops.OpOverloadPacket):
        needs_realized_inputs.update(
            getattr(fn, overload) for overload in fn.overloads()
        )


def add_layout_constraint(fn, constraint):
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            _maybe_layout_constraints[getattr(fn, overload)] = constraint
    else:
        _maybe_layout_constraints[fn] = constraint


add_needs_realized_inputs(
    [
        aten.as_strided,
        aten.as_strided_copy,
        aten.avg_pool2d,
        aten.avg_pool2d_backward,
        aten.bmm,
        aten.convolution,
        aten.convolution_backward,
        aten.max_pool2d_with_indices,
        aten.max_pool2d_with_indices_backward,
        aten.mm,
        aten.upsample_nearest2d,
        aten._upsample_nearest_exact2d,
        aten._int_mm,
    ]
)


def get_overloads(aten_fn):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    return aten_fn


def _register_foreach_lowering(aten_fn, decomp_fn):
    """
    Add a foreach lowering to lowerings dict.

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        assert len(args) <= 2
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out

    aten_fns = get_overloads(aten_fn)
    foreach_ops.update(aten_fns)
    lowerings.update(dict.fromkeys(aten_fns, wrapped))
    return wrapped


def _register_lowering(
    aten_fn,
    decomp_fn,
    broadcast,
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND],
    convert_input_to_bool,
):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args: list[Any] = list(args)
        kwargs: dict[str, Any] = dict(kwargs)
        unpacked = False
        # TODO maybe we need to use pytrees here
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional")) for fn in aten_fn
        ):
            # explicitly assert for "out=" ops for better error messages
            assert not any(x == "out" for x in kwargs.keys()), (
                "out= ops aren't yet supported"
            )

        args, kwargs = transform_args(
            args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        out = decomp_fn(*args, **kwargs)
        validate_ir(out)

        return out

    aten_fn = get_overloads(aten_fn)

    lowerings.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped


def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind: Optional[
        ELEMENTWISE_TYPE_PROMOTION_KIND
    ] = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )


def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    override_fn_when_gpu_float64=None,
    allow_alpha=False,
    triton_fallback=None,
    **kwargs
):
    def inner(*inputs: TensorBox, alpha=None):
        if triton_fallback is not None and any(
            isinstance(inp, IRNode) and is_triton(inp) for inp in inputs
        ):
            assert not allow_alpha  # not implemented
            return triton_fallback(*inputs)

        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_gpu_device = ir.is_gpu(decode_device(inputs[0].get_device()).type)

        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        low_pr_fp = (torch.bfloat16, torch.float16)
        emulate_precision_casts = (
            V.graph is not None
            and getattr(V.graph, "current_node", None) is not None
            and V.graph.current_node.meta is not None
            and V.graph.current_node.meta.get("low_precision_pointwise_barrier", False)
            and dtype in low_pr_fp
        )

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif (
                override_fn_when_gpu_float64
                and is_gpu_device
                and dtype == torch.float64
            ):
                return override_fn_when_gpu_float64(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for inp_index, load in enumerate(loaders):
                    out = load(index)
                    inp_dtype = inputs[inp_index].get_dtype()
                    if emulate_precision_casts and inp_dtype in low_pr_fp:
                        downcast = ops.to_dtype(out, inp_dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, inp_dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_precision_casts:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            device = None
            for i in inputs:
                if ir.is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        device = override_device or device

        from torch._inductor.subgraph_lowering import PointwiseSubgraphLowering
        from torch_npu.utils._dynamo import _InductorNpuRegistry

        if (
            _InductorNpuRegistry._loaded_backend == "dvm"
            and isinstance(V.graph, PointwiseSubgraphLowering)
        ):
            # DVM template subgraphs consume pointwise IR directly. Even upstream
            # lowerings can reach this builder through patched helpers such as
            # to_dtype during type promotion. Their intermediate values have no
            # MLIR traced graph (or realized strides) to extract.
            return Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=ranges,
            )

        input_graphs = fetch_graphs(inputs)
        node_name = f'pointwise_{next(node_id)}'
        origin_fn = fn_to_aten_fn[fn]
        new_graph = merge_traced_graphs(input_graphs, origin_fn, node_name, **kwargs)

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            node_name=node_name,
            traced_graph=new_graph,
        )

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype, copy=False):
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype, src_dtype=src_dtype)
    register_fn_to_aten_fn(_to_dtype, aten.to.dtype)
    return make_pointwise(_to_dtype, override_return_dtype=dtype, dtype=dtype)(x)


@register_lowering(torch.ops.npu.npu_dtype_cast_backward, type_promotion_kind=None)
@register_lowering(torch.ops.npu.npu_dtype_cast, type_promotion_kind=None)
@register_lowering(torch.ops.npu._npu_dtype_cast_backward, type_promotion_kind=None)
@register_lowering(torch.ops.npu._npu_dtype_cast, type_promotion_kind=None)
@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    if dtype.is_complex or x.get_dtype().is_complex:
        if x.get_size():
            # Decompose since aa aten fallback is more friendly for c++ codegen.
            # This decomposition doesn't work for empty tensor, which needs more investigation.
            dst = empty_like(x, dtype=dtype)
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        else:
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)
    return to_dtype(x, dtype, copy=True)


# Reserve the dtype overload before the local view packet registers its overloads.
lowerings[aten.view.dtype] = _upstream_lowerings[aten.view.dtype]


def to_device(x: TensorBox, device: torch.device, *, copy=False, non_blocking=False):
    src_dtype = x.get_dtype()
    device = decode_device(device)
    if x.get_device() == device:
        return clone(x) if copy else x

    input_graphs = fetch_graphs([x, device])
    node_name = f'to_device_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.to.device, node_name, dtype=src_dtype, copy=copy)
    return TensorBox.create(ir.DeviceCopy.create(x, device, non_blocking, traced_graph=new_graph, node_name=node_name))


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    override_return_dtype=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    use_libdevice_for_f64=False,
    triton_fallback=None,
):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)
    if use_libdevice_for_f64:
        fn_libdevice = ops_wrapper("libdevice_" + name)
        register_op_dtype_propagation_rules(
            "libdevice_" + name, type_promotion_kind, override_return_dtype
        )

    register_op_dtype_propagation_rules(
        name, type_promotion_kind, override_return_dtype
    )

    if override_fn_when_input_bool is not None:
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

    fn = register_fn_to_aten_fn(fn, aten_fn)

    fn = make_pointwise(
        fn,
        override_return_dtype=override_return_dtype,
        override_fn_when_input_bool=override_fn_when_input_bool,
        override_fn_when_gpu_float64=fn_libdevice if use_libdevice_for_f64 else None,  # type: ignore[possibly-undefined]
        allow_alpha=allow_alpha,
        triton_fallback=triton_fallback,
    )
    fn = register_lowering(
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)
    return fn


def register_foreach_pointwise(
    aten_fn,
    pointwise_lowering_fn,
    allow_alpha=False,
):
    fn = make_foreach_pointwise(pointwise_lowering_fn, allow_alpha=allow_alpha)
    fn = _register_foreach_lowering(aten_fn, fn)
    return fn


@register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    def fn(*args):
        return ops.where(*args)

    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)

    args = [cond, a, b]
    dtype = get_promoted_dtype(
        args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    register_fn_to_aten_fn(fn, aten.where)
    return make_pointwise(fn, override_return_dtype=dtype)(
        args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype)
    )


@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes, *, implicit=False, graph_fanout=False):
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    (x,) = promote_constants([x])
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    if tuple(x.get_size()) == tuple(sizes):
        return x

    if not free_unbacked_symbols(x.get_size()):
        x_size_product = V.graph.sizevars.guarding_hint_or_throw(sympy_product(x.get_size()))
        # TODO: It would be better to realize the input if any of its sizes
        # are unbacked, because typically the size will be non-zero.  However,
        # this cannot be done directly as below as we'll choke on the size_hint
        # here
        if x_size_product > 0 and not free_unbacked_symbols(sizes):
            # maybe realize input before broadcasting it
            x.mark_reuse(V.graph.sizevars.guarding_hint_or_throw(sympy_product(sizes)) // x_size_product)
    input_graphs = fetch_graphs([x.data, tuple(sizes)])
    if x.get_device().type == "cpu":
        new_graph = input_graphs[0]
        return TensorBox(ExpandView.create(x.data, tuple(sizes), traced_graph=new_graph))
    else:
        node_name = f'expand_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.expand, node_name)
        return TensorBox(ExpandView.create(x.data, tuple(sizes), traced_graph=new_graph, node_name=node_name))


@register_lowering(aten.repeat)
def repeat(x, repeats):
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    input_graphs = fetch_graphs([x, repeats])
    node_name = f'repeat_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.repeat, node_name)

    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.S.One] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    assert len(repeats) == len(x.get_size())

    new_size = list(x.get_size())

    zero_tensor = False
    for i in range(len(repeats)):
        if repeats[i] == 0:
            zero_tensor = True
        new_size[i] = new_size[i] * repeats[i]

    if zero_tensor:
        return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return clone(expand(x, new_size))

    x_loader: Callable[[Any], Any]

    def inner_fn(index):
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] != 1:
                if old_size[i] == 1:
                    index[i] = sympy.S.Zero
                else:
                    index[i] = ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    if not free_unbacked_symbols(old_size) and not free_unbacked_symbols(new_size):
        old_size_product = V.graph.sizevars.guarding_hint_or_throw(sympy_product(old_size))
        if old_size_product > 0:
            x.mark_reuse(V.graph.sizevars.guarding_hint_or_throw(sympy_product(new_size)) // old_size_product)

    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
        traced_graph=new_graph,
        node_name=node_name
    )


@register_lowering(aten._unsafe_view, type_promotion_kind=None)
@register_lowering(aten.view, type_promotion_kind=None)
@register_lowering(aten.reshape, type_promotion_kind=None)
def view(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    input_graphs = fetch_graphs([x.data, sizes])
    node_name = f'view_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.reshape, node_name)
    return TensorBox(View.create(x.data, sizes, traced_graph=new_graph, node_name=node_name))


@register_lowering(aten.permute, type_promotion_kind=None)
def permute(x, dims):
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    input_graphs = fetch_graphs([x.data, dims])
    node_name = f'permute_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.permute, node_name)
    return TensorBox(PermuteView.create(x.data, tuple(dims), traced_graph=new_graph, node_name=node_name))


@register_lowering(aten.slice, type_promotion_kind=None)
def slice_(x, dim=0, start=0, end=2**63, step=1, clamp=True):
    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    x.realize_hint()
    input_graphs = fetch_graphs([x.data])
    node_name = f'slice_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.slice, node_name, dim=dim, start=start, end=end, step=step)

    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step, clamp=clamp, traced_graph=new_graph, node_name=node_name))


def pointwise_cat(inputs, dim=0):
    # (inclusive, exclusive)
    inputs_ranges: list[tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))  # type: ignore[arg-type]
        prev_end = inputs_ranges[-1][-1]  # type: ignore[assignment]

    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        idx_dim = ops.index_expr(idx[dim], torch.int64)

        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            start = (
                ops.constant(0, torch.int64)
                if i == 0
                else ops.index_expr(inputs_ranges[i][0], torch.int64)
            )
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)

            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)

            masks.append(mask)
            idx_load = list(idx)

            # if we're concatting [4], [2]
            # when we index the second tensor for 5 we want to index 5 - 4
            # Use Identity to prevent expansion of index * stride to keep expression
            # in same int bitwidth as shape
            idx_load[dim] = Identity(idx_load[dim] - inputs_ranges[i][0])

            masked_loads.append(
                ops.masked(
                    mask,
                    lambda: inputs_loaders[i](idx_load),
                    0.0,  # this value should be unused
                ),
            )

        next_val = masked_loads[-1]
        for i in range((len(inputs)) - 2, -1, -1):
            next_val = ops.where(
                masks[i],
                masked_loads[i],
                next_val,
            )
        return next_val

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    input_graphs = fetch_graphs([inputs])
    node_name = f'cat_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.cat, node_name, dim=dim)

    return Pointwise.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
        traced_graph=new_graph,
        node_name=node_name
    )

@register_lowering(aten.cat)
def cat(inputs, dim=0):
    cpu_device = inputs[0].get_device().type == "cpu"
    if cpu_device and all(
        input.get_dtype() in [torch.int8, torch.uint8] for input in inputs
    ):
        # TODO <leslie> Remove this fallback when we support vectorization
        # code gen with uint8 data type directly.
        for input in inputs:
            input.realize()
        if all(len(input.get_size()) == 4 for input in inputs):
            inputs, _ = require_channels_last(aten.cat, *inputs)
        return fallback_handler(aten.cat.default)(inputs, dim)

    if len(inputs) == 1:
        return clone(inputs[0])

    # ATen skips known-empty 1-D tensors when validating the concat dimension.
    def cat_should_skip(tensor):
        size = tensor.get_size()
        return len(size) == 1 and V.graph.sizevars.statically_known_equals(size[0], 0)

    valid_inputs = [tensor for tensor in inputs if not cat_should_skip(tensor)]
    if not valid_inputs:
        return clone(inputs[0])
    if len(valid_inputs) == 1:
        return clone(valid_inputs[0])
    inputs = valid_inputs

    dim = _validate_dim(inputs[0], dim, 0)
    dtype = get_promoted_dtype(
        *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    def unwrap_tensor(x: Union[TensorBox, ir.StorageBox]) -> ir.IRNode:
        if isinstance(x, TensorBox):
            if isinstance(x.data, ir.BaseView):
                return x.data.unwrap_view()
            else:
                return x.data

        if isinstance(x, ir.StorageBox):
            return x.data

        return x

    def is_reduction(t):
        return isinstance(t, ir.ComputedBuffer) and isinstance(t.data, ir.Reduction)

    def can_fuse_reduction(t):
        if isinstance(t, (TensorBox, ir.StorageBox)):
            return can_fuse_reduction(unwrap_tensor(t))
        return (
            is_reduction(t)
            or isinstance(t, ir.Pointwise)
            and any(
                can_fuse_reduction(V.graph.get_buffer(read))
                for read in t.get_read_names()
            )
        )

    # fusing reducutions into computed concat buffer can cause regressions.
    fusable_reduction = any(can_fuse_reduction(t) for t in inputs)

    def should_lower_cat_input(x) -> bool:
        # Unrealized inputs will not be storage and layouts, and we dont want to realize
        # them in case we want to fuse
        if ir.is_storage_and_layout(x):
            storage, _ = ir.as_storage_and_layout(x, freeze=False)
            return not ir.ConcatKernel.can_realize_into_without_copy(storage)

        if isinstance(x, (TensorBox, ir.StorageBox)):
            return should_lower_cat_input(unwrap_tensor(x))

        if isinstance(x, ir.Pointwise):
            return True

        return False

    if config.force_pointwise_cat:
        return pointwise_cat(inputs, dim)

    # TODO: We observed negative performance impact of pointwise_cat optimization on CPU so disabled it.
    #             We will revisit this later after enabling vectorization on index_expr.
    if cpu_device:
        return TensorBox(ir.ConcatKernel.create(inputs, dim))

    def op_count(x):
        if isinstance(x, (TensorBox, ir.StorageBox)):
            return op_count(unwrap_tensor(x))

        # this will correspond to a direct memory read
        if not isinstance(x, ir.Pointwise):
            return 0

        count = x.inner_fn_opcount().num_ops
        for read in x.get_read_names():
            count += op_count(V.graph.get_buffer(read))

        return count

    # as of inputs increase, possibility for register spilling also increases
    # past a certain threshold of inputs we only fuse if the if the input kernels
    # are simple
    # not sure if we want to expose to users via config since logic may change in future
    MAX_COMPLEX_POINTWISE_CAT = 8
    MAX_SIMPLE_OP_COUNT = 2

    def additional_pointwise_ops(op: torch._ops.OpOverload):
        return op in (aten.cat.default, aten.constant_pad_nd.default)

    if len(inputs) <= MAX_COMPLEX_POINTWISE_CAT or (
        (len(inputs) <= config.max_pointwise_cat_inputs)
        and all(op_count(t) <= MAX_SIMPLE_OP_COUNT for t in inputs)
    ):
        pointwise_uses = all(
            is_pointwise_use(use, additional_pointwise_ops)
            for use in V.current_node.users
        )
        # fuse in case we will be used in a pointwise node, and there are any inputs we
        # we can prevent materialization of.
        fuse_pointwise_use = (
            any(should_lower_cat_input(inp) for inp in inputs) and pointwise_uses
        )

        # horizontal fuse in case all inputs will require a copy kernel anyway.
        # only horizontally fuse pointwise kernels
        horizontal_fuse_cat = all(
            should_lower_cat_input(inp) for inp in inputs
        ) and not any(can_fuse_reduction(t) for t in inputs)
        if fuse_pointwise_use or (horizontal_fuse_cat and not fusable_reduction):
            return pointwise_cat(inputs, dim)

    return TensorBox(ir.ConcatKernel.create(inputs, dim))


@register_lowering(aten.select, type_promotion_kind=None)
def select(x, dim, idx):
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    return squeeze(slice_(x, dim, idx, idx + 1), dim)


def fallback_handler(kernel, add_to_fallback_set=True):
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        def wrap_tensors(x):
            return TensorBox.create(x) if isinstance(x, ir.IRNode) else x

        return pytree.tree_map(
            wrap_tensors, ir.FallbackKernel.create(kernel, *args, **kwargs)
        )

    # This lets us detect that a lowering is a fallback handler.
    handler._is_fallback_handler = True  # type: ignore[attr-defined]

    return handler


def make_fallback(op, layout_constraint=None, warn=True, override_decomp=False, get_decomp_fn=None):
    check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
    assert op not in check_decomps or override_decomp, (
        f"both a fallback and a decomp for same op: {op}"
    )
    if (
        warn
        and bool(os.getenv("CI"))
        and get_decompositions([op])
        # if fallback_random, we allow not decomposing random
        and not (
            config.fallback_random
            and op in torch._decomp.decompositions_for_rng.extra_random_decomps
        )
        and not override_decomp
    ):
        # Note: 'warn' is holdover from when this was a warning, but for ops that previously
        # set warn=False we do not want a CI error.
        # Ignore the 'suppress errors' configs in CI, as this particular warning happens on startup anyway and is not
        # likely to be triggered preferentially on one CI config over another.
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )

    def register_fallback(op_overload):
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            op_overload = getattr(op, ol)
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")


make_fallback(aten.randint)


# WIP
make_fallback(aten._adaptive_avg_pool3d)  # @isuruf
# torch >= 2.16 ships decompositions for these two ops; the DVM/MLIR lowering
# keeps the aclnn fallback until the decomposed forms are validated on DVM.
make_fallback(aten.adaptive_max_pool3d, override_decomp=True)  # @isuruf
make_fallback(aten.fractional_max_pool3d)  # @isuruf
make_fallback(aten.max_pool3d_with_indices, override_decomp=True)  # @isuruf (can this one be implemented?)


# 1) Easy
make_fallback(aten.uniform, warn=False)
make_fallback(aten.exponential.default, warn=False)  # (fails accuracy on test_torch.py)
make_fallback(aten._pdist_forward)  # Has decomp. Needs benchmarks
make_fallback(aten.soft_margin_loss_backward, warn=False)  # py_impl?


# 1.5) Easy or Impossible
make_fallback(aten._cdist_forward)  # p=2 should be feasible
make_fallback(aten._cdist_backward)

# 2) Medium
make_fallback(aten._trilinear)


# 3) Difficult
# Scans
make_fallback(aten.segment_reduce.default)
make_fallback(aten._segment_reduce_backward.default)

# Histogram (need to implement Histogram IR)
make_fallback(aten.histc)
make_fallback(aten.histogram.bin_ct)
make_fallback(aten._histogramdd_bin_edges.default)
make_fallback(aten._histogramdd_from_bin_cts.default)

# Need templated kernel
make_fallback(aten.addbmm)
make_fallback(aten._addmm_activation, warn=False)

# Keep matmul_backward as an extern boundary and accept its input layout
# directly. Requiring contiguous creates a layout-copy SchedulerNode whose
# traced graph currently loses the copy semantics during MLIR fusion.
make_fallback(
    aten.matmul_backward.default,
    layout_constraint=None,
    warn=False,
    override_decomp=True,
)
add_layout_constraint(aten.matmul_backward.default, None)

# Need templated kernel. Probably impossible to write efficiently
make_fallback(aten.convolution_backward, constrain_to_fx_strides)
make_fallback(aten._cudnn_rnn, require_dense)
make_fallback(aten._cudnn_rnn_backward, require_contiguous)

# Haven't checked but sound difficult / impossible
make_fallback(aten._embedding_bag, require_contiguous)
make_fallback(aten._embedding_bag_forward_only, require_contiguous)
make_fallback(aten._embedding_bag_backward)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._fused_moving_avg_obs_fq_helper)
make_fallback(aten._fused_moving_avg_obs_fq_helper_functional)


# 4) Backwards (try py_impl'ing them) when fwd is written as a decomp
make_fallback(aten.max_pool3d_with_indices_backward)
make_fallback(aten._adaptive_avg_pool2d_backward, require_dense)
make_fallback(aten._adaptive_avg_pool3d_backward)
make_fallback(aten.adaptive_max_pool2d_backward)
make_fallback(aten.adaptive_max_pool3d_backward)
make_fallback(aten.fractional_max_pool2d_backward)
make_fallback(aten.fractional_max_pool3d_backward)
make_fallback(aten.replication_pad1d_backward)
make_fallback(aten.replication_pad2d_backward)
make_fallback(aten.upsample_linear1d_backward)
make_fallback(aten.upsample_bicubic2d_backward, require_contiguous)
make_fallback(aten.upsample_trilinear3d_backward)
make_fallback(aten.grid_sampler_2d_backward, require_dense)
make_fallback(aten._pdist_backward)


# 5) Impossible (missing triton/CPU features)

# Sorting / Sorting-like
make_fallback(aten.sort)
make_fallback(aten.kthvalue)
make_fallback(aten.topk)
make_fallback(aten.mode)
make_fallback(aten.median)
make_fallback(aten.nanmedian)
make_fallback(aten.randperm)
make_fallback(aten.resize_)
make_fallback(aten.resize_as_)

# Linalg
make_fallback(aten._linalg_det)
make_fallback(aten.linalg_householder_product)
make_fallback(aten.linalg_inv_ex)
make_fallback(aten.linalg_ldl_factor_ex)
make_fallback(aten.linalg_ldl_solve)
make_fallback(aten.linalg_lu)
make_fallback(aten.linalg_lu_factor_ex)
make_fallback(aten.linalg_lu_solve)
make_fallback(aten.linalg_matrix_exp)
make_fallback(aten.linalg_qr)
make_fallback(aten._linalg_slogdet)
make_fallback(aten._linalg_solve_ex)
make_fallback(aten.linalg_solve_triangular)
make_fallback(aten._linalg_svd)
make_fallback(aten.lu_unpack)
make_fallback(aten.ormqr)
make_fallback(aten._linalg_check_errors)
make_fallback(aten.linalg_pinv.atol_rtol_tensor)
make_fallback(aten._linalg_eigh)
make_fallback(aten.triangular_solve)
make_fallback(aten.linalg_cholesky_ex)
make_fallback(aten.cholesky_inverse)
make_fallback(aten.cholesky_solve)
make_fallback(aten.geqrf)
make_fallback(aten._fft_r2c)  # needs complex as well

# Data dependent (are these necessary?)
make_fallback(aten.nonzero.default)

# Misc
make_fallback(aten.gcd.default, warn=False)
make_fallback(aten._thnn_fused_lstm_cell, require_dense)
make_fallback(torch._prims.rng_prims.run_and_save_rng_state)
make_fallback(torch._prims.rng_prims.run_with_rng_state)
make_fallback(torch._prims.rng_prims.graphsafe_run_with_rng_state)


# Implemented / Half implemented
# Scans. Implemented for CUDA, missing CPU
make_fallback(aten.masked_scatter)
make_fallback(aten.masked_scatter_backward)

# Complex number support
make_fallback(aten.view_as_complex, require_contiguous)
make_fallback(aten.angle)  # needs complex

# Needs efficentzerotensor
make_fallback(aten._efficientzerotensor)

# Needs Sparse
make_fallback(aten._sparse_coo_tensor_with_dims_and_tensors)
make_fallback(aten.to_sparse)
make_fallback(aten._to_sparse)

# 6) Pattern-matched
make_fallback(
    aten._scaled_dot_product_efficient_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_efficient_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_cudnn_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_cudnn_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_fused_attention_overrideable.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_fused_attention_overrideable_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(aten._flash_attention_forward.default, sdpa_constraint)
make_fallback(aten._flash_attention_backward.default, sdpa_constraint)
make_fallback(aten._efficient_attention_forward.default, sdpa_constraint)
make_fallback(aten._efficient_attention_backward.default, sdpa_constraint)

# index_reduce requires fallback when use_scatter_fallback(...) returns True
make_fallback(aten.index_reduce)


# Register with type_promotion_kind None.
# For example, fp16.copy_(fp32) should **not** promote the first input's dtype.
@register_lowering(aten.copy, type_promotion_kind=None)
def copy(self, src, non_blocking=False):
    x = src
    if self.get_device() != src.get_device():
        x = to_device(x, self.get_device())
    if self.get_dtype() != src.get_dtype():
        x = to_dtype(x, self.get_dtype())

    if self.get_size() != src.get_size():
        out = expand(x, self.get_size())
        return clone(out)
    return clone(x)


@register_lowering(aten.clone)
def clone(x, *, memory_format=None):
    # TODO(jansel): memory format
    input_graphs = fetch_graphs(x)
    node_name = f'clone_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.clone, node_name)
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
        traced_graph=new_graph,
        node_name=node_name
    )


@register_lowering([torch.tensor, aten.scalar_tensor])
def tensor(data, *, dtype=None, device=None, layout=None, pin_memory=False):
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    assert_nyi(not pin_memory, "pin_memory")
    input_graphs = fetch_graphs([data])
    node_name = f'tensor_{next(node_id)}'
    new_graph = merge_traced_graphs(
        input_graphs, torch.tensor, node_name,
        dtype=dtype,
        device='npu',
        pin_memory=False,
    )
    if isinstance(_unwrap(data), int):
        dtype = dtype or torch.int64
    else:
        dtype = dtype or torch.get_default_dtype()

    ranges: list[sympy.Expr] = []

    if isinstance(data, sympy.Basic):

        def inner_fn(index):
            return ops.index_expr(data, dtype)

    elif isinstance(data, (float, int)):

        def inner_fn(index):
            return ops.constant(data, dtype)

    elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
        # inline small tensors
        ranges.append(sympy.Integer(len(data)))

        def inner_fn(index):
            def binary_search(start, end):
                assert start < end
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(
                    ops.lt(
                        ops.index_expr(index[0], torch.int64),
                        ops.constant(mid, torch.int64),
                    ),
                    binary_search(start, mid),
                    binary_search(mid, end),
                )

            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))

    else:
        return V.graph.add_tensor_constant(
            torch.tensor(data, dtype=dtype, device=device)
        )

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=ranges,
        traced_graph=new_graph,
        node_name=node_name
    )


def _full(fill_value, device, dtype, size):
    value = fill_value
    dtype = torch.int32 if dtype == torch.int64 else dtype
    if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
        value = value.value

    if isinstance(value, (int, float)):

        def inner_fn(index):
            return ops.constant(value, dtype)

    elif isinstance(value, sympy.Basic):

        def inner_fn(index):
            return ops.index_expr(value, dtype)

    else:
        assert len(value.get_size()) == 0
        value_loader = value.make_loader()

        def inner_fn(index):
            return value_loader([])

    node_name = f'full_{next(node_id)}'
    # [wtd#18] Use passed-in device param instead of hardcoded 'npu' to avoid device mismatch
    new_graph = merge_traced_graphs(
        [size, fill_value], aten.full.default, node_name,
        device=device, dtype=dtype, layout=torch.strided, pin_memory=False,
    )

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
        traced_graph=new_graph,
        node_name=node_name
    )


# No direct callers in this module. apply_lowering_patch binds this by name in
# torch._inductor.lowering, where full, full_like, and constant_like resolve it at runtime.
# This implicit dependency is difficult to maintain. Move the NPU policy to
# explicit lowering entry points before removing the override.
def tensor_constructor(fill_value):
    # torch.zeros, torch.ones, etc
    def inner(
        *size,
        names=None,
        dtype=None,
        device=None,
        layout=None,
        pin_memory=False,
        memory_format=None,
    ):
        assert_nyi(names is None, "named tensors")
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        assert_nyi(not pin_memory, "pin_memory")
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        # All sizes at lowering time should be sympy.Symbol, not SymInt!
        for s in size:
            assert not isinstance(s, torch.SymInt)
        size = [sympy.expand(s) for s in size]
        return _full(fill_value, device, dtype, size)

    return inner


@register_lowering([torch.empty, aten.empty])
def empty(
    *size,
    names=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    assert_nyi(names is None, "named tensors")
    device = decode_device(device)
    if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
        size = tuple(size[0])
    return empty_strided(
        size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


empty_like = register_lowering(aten.empty_like)(create_tensor_like(empty))


@register_lowering(aten.empty_strided)
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    assert isinstance(size, (list, tuple))
    assert isinstance(stride, (list, tuple, type(None)))
    assert_nyi(not pin_memory, "pin_memory")
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    device = device or torch.tensor(0.0).device
    device = decode_device(device)
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    pointwise.realize()
    buffer = pointwise.data.data
    # explicitly set ranges to zeros in order to make a NopKernelSchedulerNode
    buffer.data = dataclasses.replace(buffer.data, ranges=[0] * len(size))
    assert isinstance(buffer, ir.ComputedBuffer)
    size = [sympy.expand(s) for s in size]
    stride = (
        [sympy.expand(s) for s in stride]
        if stride
        else ir.FlexibleLayout.contiguous_strides(size)
    )
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=size,
        stride=stride,
    )
    return pointwise


# Keep the explicit registration while using PyTorch's constructor.
register_lowering(aten.full_like, type_promotion_kind=None)(full_like)


# The installer binds this helper for upstream _unsafe_masked_index_put_accumulate.
@make_pointwise
def clamp(a, min, max):
    return ops.maximum(min, ops.minimum(max, a))


def should_not_sum(a, b, keepdims):
    if not b:
        return False
    for i in b:
        if i < 0 or i >= len(a):
            return False
        if a[i] != 1:
            return False

    unique_indices = reversed(b)
    if not keepdims:
        for i in unique_indices:
            del a[i]
        return a
    return a

def make_reduction(reduction_type: ReductionType, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        if dtype is not None:
            x = to_dtype(x, dtype)
        if axis and axis[-1] < 0:
            offset = len(x.get_size())
            axis = [ax + offset for ax in axis]
            axis = sorted(axis)
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        new_size = should_not_sum(x.get_size().copy(), axis, keepdims)
        if new_size:
            node_name = f'reshape_{next(node_id)}'
            input_graphs = fetch_graphs([x, new_size])
            new_graph = merge_traced_graphs(input_graphs, aten.reshape, node_name)
        else:
            node_name = f'reduction_{next(node_id)}'
            input_graphs = fetch_graphs([x, axis if axis is not None else list(range(len(x.get_size())))])
            reduction_kwargs = {"keepdim": keepdims}
            if dtype is not None:
                reduction_kwargs["dtype"] = dtype
            new_graph = merge_traced_graphs(input_graphs, reduction_type_to_aten_fn[reduction_type],
                                            node_name, **reduction_kwargs)
        result = Reduction.create(reduction_type=reduction_type,
                                  input_node=x,
                                  node_name=node_name,
                                  traced_graph=new_graph,
                                  **kwargs)
        if isinstance(
            result.data.data,  # type: ignore[attr-defined]
            Reduction,
        ):  # Only realize if reduction isn't unrolled
            result.realize()
        return result

    return inner


def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    if correction is None:
        correction = 1

    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()

    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)

    denom = sympy_product(size[i] for i in axis)
    if correction:
        denom = sympy.Max(denom - correction, 0)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    x_var = div(sum_result, denom)
    if not return_mean:
        return (x_var,)

    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    return x_var, x_mean


# Upstream var and var_mean reach this two-step reduction through the same-name patch.
def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)
    kwargs = dict(
        x=x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=return_mean,
    )
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    return output[0] if not return_mean else output


@make_pointwise
@register_to_aten(aten_fn=aten.pow)
def pow_native(a, b):
    return ops.pow(a, b)


fallback_pow_tensor_tensor = fallback_handler(
    aten.pow.Tensor_Tensor, add_to_fallback_set=False
)
fallback_pow_scalar = fallback_handler(aten.pow.Scalar, add_to_fallback_set=False)
fallback_pow_tensor_scalar = fallback_handler(
    aten.pow.Tensor_Scalar, add_to_fallback_set=False
)


@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    # The native DVM scalar pow path does not handle non-finite exponents.
    if isinstance(a, TensorBox) and isinstance(b, float) and not math.isfinite(b):
        return fallback_pow_tensor_scalar(a, b)
    if isinstance(b, float) and b.is_integer():
        return pow(a, int(b))
    elif isinstance(b, float) and b == 0.5:
        return sqrt(a)
    elif isinstance(b, int) and b == 1:
        return clone(a)

    # Type promotion ensures all tensor arguments have the same type
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
    is_integer_pow = is_integer_dtype(dtype)

    # Optimize away small fixed powers, or for integers avoid falling back to ATen
    embed_exponent = isinstance(b, int) and (
        -32 < b < 32 or (is_integer_pow and b >= 0)
    )
    if embed_exponent:
        loader = a.make_loader()

        def fn(idx):
            return pow_recursive(loader(idx), b, a.get_dtype())

        input_graphs = fetch_graphs([a, b])
        node_name = f'pointwise_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.pow, node_name)

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
            node_name=node_name,
            traced_graph=new_graph,
        )

    if isinstance(a, Number):
        if a == 1:
            return full_like(b, 1)
        if a == 2 and is_float_dtype(b.get_dtype()):
            return exp2(b)

    if is_integer_pow:
        # ops.pow doesn't work for integers
        if isinstance(a, Number):
            return fallback_pow_scalar(a, b)
        elif isinstance(b, Number):
            return fallback_pow_tensor_scalar(a, b)
        else:
            return fallback_pow_tensor_tensor(a, b)

    return pow_native(a, b)


def mutate_to(changed, val, unsafe_alias=False):
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data

    if not isinstance(val, ir.StorageBox):
        # introduce a copy to handle views
        input_graphs = fetch_graphs([changed, val])
        node_name = f'copy__{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.copy_, node_name)
        val = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
            traced_graph=new_graph,
            node_name=node_name
        ).data
        assert isinstance(val, ir.StorageBox)

    if isinstance(changed_data, ir.StorageBox) and not (
        changed_data.is_input_buffer()
        # In AOTI, module parameters and buffers are not lifted as graph inputs
        or changed_data.is_module_buffer()
        or isinstance(changed_data.data, ir.NopKernel)
    ):
        # Fast path, just swing the data pointer
        val.realize()
        changed_data.data = val.data
        return changed

    ir.MutationLayoutSHOULDREMOVE.realize_into(
        val, changed_data, unsafe_alias=unsafe_alias
    )
    return changed


def _has_slice_op(tensor_box):
    """
    Check whether TensorBox is derived from aten.slice.
    Only check the first origin node.
    """
    if not isinstance(tensor_box, TensorBox):
        return False
    visited = set()

    def visit(node):
        if node is None:
            return False
        obj_id = id(node)
        if obj_id in visited:
            return False
        visited.add(obj_id)
        origins = getattr(node, "origins", None)
        if origins:
            first_origin = next(iter(origins))
            if isinstance(first_origin, torch.fx.Node):
                return first_origin.target == aten.slice.Tensor
        origin_node = getattr(node, "origin_node", None)
        if origin_node is not None:
            return origin_node.target == aten.slice.Tensor

    return visit(tensor_box.data)


@register_lowering(aten.copy_, type_promotion_kind=None)
def copy_(dst, src, non_blocking=False):
    if dst is src:
        # dst.copy_(dst) can happen from the reinplacing pass
        return dst
    has_slice_dst = _has_slice_op(dst)
    if has_slice_dst:
        return fallback_handler(aten.copy_.default)(
            dst,
            src,
            non_blocking=non_blocking,
        )

    src = to_device(src, dst.get_device())
    src = to_dtype(src, dst.get_dtype())
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)


@make_pointwise
def truncdiv(a, b):
    return ops.truncdiv(a, b)


@register_lowering(aten.div, broadcast=True)
def div_mode(a, b, rounding_mode=None):
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    if rounding_mode == "floor":
        assert not both_boolean, "floordiv operands can not be boolean at the same time"
        return fallback_handler(aten.div.Tensor_mode)(a, b, rounding_mode=rounding_mode)
    if rounding_mode == "trunc":
        assert not both_boolean, "truncdiv operands can not be boolean at the same time"
        return fallback_handler(aten.div.Tensor_mode)(a, b, rounding_mode=rounding_mode)
    return div(a, b)


@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)
        fn = register_fn_to_aten_fn(fn, aten.mul)
        return make_pointwise(fn)(a, b)


# NOTE: prims.div maps to a / b in C, so performs truncation division on
#   integer inputs and true division for floating and complex inputs.
@register_lowering([prims.div], broadcast=True)
def div_prim(a, b):
    is_integral = all(is_boolean_type(x) or is_integer_type(x) for x in [a, b])

    if is_integral:
        return truncdiv(a, b)

    def fn(*args):
        return ops.truediv(*args)

    fn = register_fn_to_aten_fn(fn, aten.div)
    return make_pointwise(fn)(a, b)


# Select true division for this overload after the local div_mode packet registration.
lowerings[aten.div.Tensor] = _upstream_lowerings[aten.div.Tensor]


@register_lowering([aten.reciprocal], broadcast=True,)
def reciprocal(a):
    return div(1.0, a)


@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if isinstance(axis, int):
        axis = (axis,)
    if axis and any(ax < 0 for ax in axis):
        offset = len(x.get_size())
        axis = tuple(ax + offset if ax < 0 else ax for ax in axis)
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    if sum_fallback_predicate is not None and sum_fallback_predicate(x, axis):
        fn = fallback_handler(aten.sum.dim_IntList, add_to_fallback_set=False)
    else:
        fn = make_reduction("sum", override_return_dtype=dtype)
    r = fn(x, axis, keepdims, dtype=dtype)
    return r


@register_lowering(aten.max, type_promotion_kind=None)
def reduce_max(x, dim=None, keepdim=False):
    if dim is not None:
        return (
            reduce_amax(x, axis=dim, keepdims=keepdim),
            reduce_argmax(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amax(x, axis=None, keepdims=keepdim)


@register_lowering(aten.min, type_promotion_kind=None)
def reduce_min(x, dim=None, keepdim=False):
    if dim is not None:
        return (
            reduce_amin(x, axis=dim, keepdims=keepdim),
            reduce_argmin(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amin(x, axis=None, keepdims=keepdim)


register_lowering(prims.xor_sum)(make_reduction("xor_sum"))
reduce_amax = register_lowering(aten.amax)(make_reduction("max"))
reduce_amin = register_lowering(aten.amin)(make_reduction("min"))
reduce_argmax = register_lowering(aten.argmax)(
    make_reduction("argmax", override_return_dtype=torch.int64)
)
reduce_argmin = register_lowering(aten.argmin)(
    make_reduction("argmin", override_return_dtype=torch.int64)
)

add = register_pointwise(
    aten.add, allow_alpha=True, override_fn_when_input_bool="logical_or"
)

def register_pointwise_numeric(op, name=None, triton_fallback=None):
    return register_pointwise(
        op,
        name=name,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        triton_fallback=triton_fallback,
    )


def register_pointwise_numeric_ldf64(op):
    return register_pointwise(
        op,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        use_libdevice_for_f64=True,
    )

@register_lowering([aten.neg])
def neg(a):
    if a.get_dtype() in (torch.int32, torch.int64):
        return mul(a, -1)
    fn = ops_wrapper(aten.neg.__name__)
    fn = register_fn_to_aten_fn(fn, aten.neg)
    return make_pointwise(fn)(a)


rsqrt = register_pointwise_numeric(aten.rsqrt)  # noqa: F811
exp = register_pointwise_numeric_ldf64(aten.exp)
exp2 = register_pointwise_numeric(aten.exp2)
expm1 = register_pointwise_numeric(aten.expm1)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
sqrt = register_pointwise_numeric_ldf64(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub, allow_alpha=True)
register_pointwise_numeric_ldf64(aten.cos)
register_pointwise_numeric_ldf64(aten.sin)
bitwise_and = register_pointwise(aten.bitwise_and)
bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)
bitwise_not = register_pointwise(
    aten.bitwise_not, override_fn_when_input_bool="logical_not"
)
bitwise_or = register_pointwise(aten.bitwise_or)
bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)
bitwise_xor = register_pointwise(aten.bitwise_xor)
register_pointwise_numeric(aten.lgamma)
erf = register_pointwise_numeric(aten.erf)
register_lowering(
    aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)(erf)

register_pointwise_numeric(aten.log1p)
register_pointwise_numeric(aten.tan)
register_pointwise_numeric(aten.tanh)
register_pointwise_numeric_ldf64(aten.log)
logical_and = register_pointwise(
    aten.logical_and,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_not = register_pointwise(
    aten.logical_not,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_or = register_pointwise(
    aten.logical_or,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_xor = register_pointwise(
    aten.logical_xor,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
maximum = register_pointwise(aten.maximum)
minimum = register_pointwise(aten.minimum)
clamp_min = register_pointwise(aten.clamp_min, name='maximum')
clamp_max = register_pointwise(aten.clamp_max, name='minimum')
register_lowering(aten.clamp_min)(clamp_min)
register_lowering(aten.clamp_max)(clamp_max)
abs = register_pointwise(aten.abs)
# reciprocal = register_pointwise_numeric(aten.reciprocal)
register_pointwise(aten.remainder)
sign = register_pointwise(aten.sign, override_fn_when_input_bool="identity")
register_pointwise(aten.signbit, override_return_dtype=torch.bool)

register_lowering(aten._neg_view)(neg)

register_pointwise(aten.le, override_return_dtype=torch.bool)
register_pointwise(aten.lt, override_return_dtype=torch.bool)
register_pointwise(aten.ge, override_return_dtype=torch.bool)
gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
register_pointwise(aten.eq, override_return_dtype=torch.bool)
register_pointwise(aten.ne, override_return_dtype=torch.bool)

register_pointwise_numeric(aten.cosh)
register_pointwise_numeric(aten.sinh)
register_pointwise_numeric(aten.acos)
register_pointwise_numeric(aten.acosh)
register_pointwise_numeric(aten.asin)
register_pointwise_numeric(aten.asinh)
register_pointwise_numeric(aten.atan2)
register_pointwise_numeric(aten.atan)
register_pointwise_numeric(aten.atanh)
register_pointwise_numeric(aten.copysign)
register_pointwise_numeric(aten.erfc)
register_pointwise_numeric(aten.erfinv)
register_pointwise_numeric(aten.hypot)
register_pointwise_numeric(aten.log10)
register_pointwise_numeric(aten.log2)
register_pointwise_numeric(aten.nextafter)

from torch._inductor.codegen.common import pointwise_overrides_data


def _get_pointwise_overrides(ns, name):
    data = pointwise_overrides_data[name]
    op = getattr(ns, data.name, None)
    if op is None:
        return

    def make_triton_fallback(op):
        if data.triton is None:
            return fallback_handler(op)

    if isinstance(op, torch._ops.OpOverloadPacket):
        for olname in op.overloads():
            ol = getattr(op, olname)
            yield ol, data.type_promotion_kind, make_triton_fallback(ol)
    else:
        yield op, data.type_promotion_kind, make_triton_fallback(op)


for name in pointwise_overrides_data:
    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        aten, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )

    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        prims, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )


foreach_add_list = register_foreach_pointwise(
    aten._foreach_add.List, add, allow_alpha=True
)
foreach_add_scalar = register_foreach_pointwise(
    aten._foreach_add.Scalar, add, allow_alpha=True
)
register_foreach_pointwise(aten._foreach_add.Tensor, add, allow_alpha=True)
foreach_mul_list = register_foreach_pointwise(aten._foreach_mul.List, mul)
register_foreach_pointwise(aten._foreach_mul.Tensor, mul)
foreach_mul_scalar = register_foreach_pointwise(aten._foreach_mul.Scalar, mul)
register_foreach_pointwise(aten._foreach_sub.List, sub)
register_foreach_pointwise(aten._foreach_sub.Scalar, sub)
register_foreach_pointwise(aten._foreach_neg.default, neg)
register_foreach_pointwise(aten._foreach_abs.default, abs)
register_foreach_pointwise(aten._foreach_pow.Scalar, pow)
register_foreach_pointwise(aten._foreach_pow.List, pow)
register_foreach_pointwise(aten._foreach_pow.ScalarAndTensor, pow)
foreach_div_list = register_foreach_pointwise(aten._foreach_div.List, div)
register_foreach_pointwise(aten._foreach_div.Tensor, div)
foreach_div_scalar = register_foreach_pointwise(aten._foreach_div.Scalar, div)
register_foreach_pointwise(aten._foreach_sqrt, sqrt)
register_foreach_pointwise(aten._foreach_rsqrt, rsqrt)
register_foreach_pointwise(aten._foreach_maximum.List, maximum)
register_foreach_pointwise(aten._foreach_maximum.Scalar, maximum)
register_foreach_pointwise(aten._foreach_minimum.List, minimum)
register_foreach_pointwise(aten._foreach_minimum.Scalar, minimum)
register_foreach_pointwise(aten._foreach_clamp_min.List, maximum)
register_foreach_pointwise(aten._foreach_clamp_min.Scalar, maximum)
register_foreach_pointwise(aten._foreach_clamp_max.List, minimum)
register_foreach_pointwise(aten._foreach_clamp_max.Scalar, minimum)
register_foreach_pointwise(aten._foreach_reciprocal, reciprocal)
register_foreach_pointwise(aten._foreach_sign, sign)
register_foreach_pointwise(aten._foreach_copy, copy)


# these are only encountered as outputs of the graph
# reinplacing epilogue copies improves compile time
# by removing extra buffers sent to the scheduler.
def register_foreach_inplace(aten_op, outplace_aten_op, outplace_op):
    inplaceable_foreach_ops[outplace_aten_op] = aten_op
    inplace_foreach_ops.add(aten_op)

    def fn(*args, **kwargs):
        results = outplace_op(*args, **kwargs)
        mut_results = []
        for arg, result in zip(args[0], results):
            mut_results.append(mutate_to(arg, result, unsafe_alias=True))

        return mut_results

    _register_foreach_lowering(aten_op, fn)


register_foreach_inplace(
    aten._foreach_add_.List, aten._foreach_add.List, foreach_add_list
)
register_foreach_inplace(
    aten._foreach_add_.Scalar, aten._foreach_add.Scalar, foreach_add_scalar
)
register_foreach_inplace(
    aten._foreach_mul_.List, aten._foreach_mul.List, foreach_mul_list
)
register_foreach_inplace(
    aten._foreach_mul_.Scalar, aten._foreach_mul.Scalar, foreach_mul_scalar
)
register_foreach_inplace(
    aten._foreach_div_.List, aten._foreach_div.List, foreach_div_list
)
register_foreach_inplace(
    aten._foreach_div_.Scalar, aten._foreach_div.Scalar, foreach_div_scalar
)


def register_inplace(aten_op, outplace_op):
    @register_lowering(aten_op, type_promotion_kind=None)
    def fn(*args, **kwargs):
        result = outplace_op(*args, **kwargs)
        result = to_dtype(result, args[0].get_dtype())
        return mutate_to(args[0], result)

    return fn


register_inplace(aten.add_, add)
register_inplace(aten.bitwise_and_, bitwise_and)
register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
register_inplace(aten.bitwise_not_, bitwise_not)
register_inplace(aten.bitwise_or_, bitwise_or)
register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
register_inplace(aten.bitwise_xor_, bitwise_xor)
register_inplace(aten.mul_, mul)
register_inplace(aten.div_.Tensor, div)
register_inplace(aten.div_.Tensor_mode, div_mode)
register_inplace(aten.logical_and_, logical_and)
register_inplace(aten.logical_not_, logical_not)
register_inplace(aten.logical_or_, logical_or)
register_inplace(aten.logical_xor_, logical_xor)
register_inplace(aten.sub_, sub)
register_inplace(aten.relu_, relu)
register_inplace(aten.sigmoid_, sigmoid)


register_lowering(aten.__and__)(bitwise_and)
register_lowering(aten.__lshift__)(bitwise_left_shift)
register_lowering(aten.__or__)(bitwise_or)
register_lowering(aten.__rshift__)(bitwise_right_shift)
register_lowering(aten.__xor__)(bitwise_xor)

register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)


if hasattr(torch.ops.fsdp, "copy_"):

    @register_lowering(torch.ops.fsdp.copy_.default)
    def fsdp_copy_(dst, src):
        if dst is src:
            # dst.copy_(dst) can happen from the reinplacing pass
            return dst
        src = to_device(src, dst.get_device())
        src = to_dtype(src, dst.get_dtype())
        src = expand(src, dst.get_size())
        return mutate_to(dst, src)


from torch._higher_order_ops.auto_functionalize import auto_functionalized


make_fallback(auto_functionalized)
