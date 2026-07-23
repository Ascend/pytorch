# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Copyright (c) 2013 the respective contributors
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sympy
from . import config as ncfg
from . import device_props as _device_props
from .ir import NPUCatLoopKernel
from .lowering_override_list import GENERATE_LIST, KEEP_UPSTREAM_LOWERING
import torch
from torch._inductor.lowering import (
    lowerings,
    make_fallback,
    register_lowering,
    _validate_dim,
    to_dtype,
    var_mean_sum_,
    fallback_handler,
    Pointwise,
    _convert_element_type,
)
from torch.utils._sympy.functions import Identity
from torch._inductor import ir

from torch._inductor.decomposition import decompositions
from torch._prims_common import get_computation_dtype, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._inductor.virtualized import ops, V

import torch._ops

log = logging.getLogger("torch._inductor")

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu

FALLBACK_LIST = []


def _register_npu_inductor_fallbacks():
    gen_set = set()
    for fn in GENERATE_LIST:
        gen_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                gen_set.add(other_fn)

    # Ops whose upstream lowering we deliberately preserve (see above).
    keep_set = set()
    for fn in KEEP_UPSTREAM_LOWERING:
        keep_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                keep_set.add(getattr(fn, overload))

    for op in lowerings:
        if op in keep_set:
            continue
        if op not in decompositions and op not in gen_set and isinstance(
            op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload,
                 torch._ops.HigherOrderOperator)):
            make_fallback(op)
            FALLBACK_LIST.append(op)

def overwrite_lowering(aten_fn, decomp_fn, *args, **kwargs):
    aten_fn = [aten_fn] if not isinstance(aten_fn, (list, tuple)) else list(aten_fn)

    for fn in list(aten_fn):
        register_lowering(fn, *args, **kwargs)(decomp_fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                overwrite_lowering(other_fn, decomp_fn, *args, **kwargs)

overwrite_lowering(
    [
        prims.convert_element_type,
        npu.npu_dtype_cast,
        npu.npu_dtype_cast_backward
    ],
    _convert_element_type, type_promotion_kind=None)


def npu_var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
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

torch._inductor.lowering.var_mean_helper_ = npu_var_mean_helper_


def npu_lowering_index_select(x, select_dim, indices):
    """A5: lower a row gather so inner_fn emits ops.index_select (CANN register
    gather) instead of an indirect load. Falls back to indirect load on a nested
    (indirect output coord) gather the register op can't express."""
    assert isinstance(x, ir.TensorBox)
    assert isinstance(indices, ir.TensorBox)
    assert "int" in str(indices.get_dtype())
    weight_loader = x.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    x_size = x.get_size()
    new_size = [*x_size[:select_dim], *indices.get_size(), *x_size[select_dim + 1:]]

    def inner_fn(idx):
        assert len(idx) == len(new_size), f"{idx} != {new_size}"
        nested_indirect = any("tmp" in str(v) or "indirect" in str(v) for v in idx)
        var_index = indices_loader(idx[select_dim:select_dim + indices_ndim])
        set_indirect = ops.indirect_indexing(var_index, x_size[select_dim])
        x_idx = [*idx[:select_dim]] + [set_indirect] + [*idx[select_dim + indices_ndim:]]
        if nested_indirect:
            return weight_loader(x_idx)
        try:
            weight_indexer = x.data.make_indexer()
            weight_name = x.data.get_name()
            return ops.index_select(
                weight_name,
                weight_indexer(x_idx),
                var_index,
                str(set_indirect),
                int(x_size[select_dim]),
            )
        except Exception:
            return weight_loader(x_idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
    )


_upstream_embedding = torch._inductor.lowering.lowerings.get(aten.embedding.default)


def npu_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """A5 CANN embedding via ops.index_select. Keeps the upstream indirect gather
    unless: A5 + cann_index_select on, weight a real 2-D [V,H] buffer, node not
    skip_lowering."""
    def _upstream():
        if _upstream_embedding is not None:
            return _upstream_embedding(weight, indices)
        return fallback_handler(aten.embedding.default)(
            weight, indices, padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
        )

    if not (_device_props.is_a5() and ncfg.cann_index_select):
        return _upstream()
    node = getattr(V, "current_node", None)
    if node is not None and node.meta.get("skip_lowering", False):
        return _upstream()
    wsize = weight.get_size()
    if len(wsize) != 2 or V.graph.sizevars.statically_known_equals(wsize[1], 1):
        return _upstream()
    if isinstance(weight, ir.TensorBox) and isinstance(weight.data, ir.BaseView):
        return _upstream()
    return npu_lowering_index_select(weight, 0, indices)


overwrite_lowering(aten.embedding, npu_embedding, type_promotion_kind=None)


def npu_cat_loop(inputs, dim):
    """Build the concat (see NPUCatLoopKernel)."""
    inputs_ranges = []
    prev_end = 0
    for inp in inputs:
        size = inp.get_size()[dim]
        inputs_ranges.append((prev_end, prev_end + size))
        prev_end = inputs_ranges[-1][1]

    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        # Representative value fn for dtype/read-dep tracking; the real per-input
        # stores are emitted by store_output. Mirror the in-bounds loads.
        acc = None
        for i, inp in enumerate(inputs):
            lo, _ = inputs_ranges[i]
            idx_load = list(idx)
            local = sympy.Mod(idx_load[dim] - lo, inp.get_size()[dim])
            idx_load[dim] = Identity(local)
            v = inputs_loaders[i](idx_load)
            acc = v if acc is None else ops.add(acc, v)
        return acc

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][1]

    box = NPUCatLoopKernel.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
        cat_inputs=tuple(inputs),
        cat_dim=dim,
        cat_ranges=tuple(inputs_ranges),
    )
    # Pin against fusion: output is written by cat_store side effects, not a
    # returned value; a fused consumer would re-load before the stores run.
    name = box.realize()
    if name:
        V.graph.no_fuse_buffer_names.add(name)
    return box


def npu_cat(inputs, dim=0):
    """NPU cat: every concat routes to cat_loop (see NPUCatLoopKernel) so the
    concat axis tiles freely and scales to any axis/extent and dynamic shapes."""
    if len(inputs) == 1:
        from torch._inductor.lowering import clone
        return clone(inputs[0])

    dim = _validate_dim(inputs[0], dim, 0)
    from torch._inductor.lowering import get_promoted_dtype
    dtype = get_promoted_dtype(
        *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    return npu_cat_loop(inputs, dim)


overwrite_lowering(aten.cat, npu_cat, type_promotion_kind=None)


_upstream_slice_ = torch._inductor.lowering.slice_


def npu_slice(x, dim=0, start=0, end=2**63, step=1, clamp=True):
    """NPU slice. On the unbacked path, torch 2.13's slice_ reads get_stride()[dim]
    eagerly, which a gather View can't answer (raises NotImplementedError). Clone
    the View into a contiguous buffer there so the stride resolves; static slices
    are untouched. The unbacked condition is read from the node's unbacked_bindings."""
    if (
        isinstance(x, ir.TensorBox)
        and isinstance(x.data, ir.BaseView)
        and not ir.is_storage_and_layout(x)
    ):
        current_node = V.graph.current_node
        unbacked_bindings = (
            current_node.meta.get("unbacked_bindings", {})
            if current_node is not None
            else {}
        )
        if unbacked_bindings:
            from torch._inductor.lowering import clone
            x = clone(x)

    # Bisheng strided-broadcast-load miscompile workaround (realize_strided_slice_input).
    # A step>1 slice over an unrealized chain fuses into operand loads; if one is a
    # low-rank broadcast the fused load is a strided broadcast, which ascend
    # miscompiles to all-ZEROS. Realizing the input makes the slice read it full-rank
    # (an FX clone gets DCE'd back, so force IR-level realize). Step>1 producers only.
    if (
        step != 1
        and isinstance(x, ir.TensorBox)
        and ncfg.realize_strided_slice_input
        and not ir.is_storage_and_layout(x)
    ):
        data = x.data
        is_pointwise_producer = isinstance(data, ir.StorageBox) and isinstance(
            getattr(data, "data", None), (ir.Pointwise, ir.Reduction)
        )
        if is_pointwise_producer:
            x.realize()

    return _upstream_slice_(x, dim=dim, start=start, end=end, step=step, clamp=clamp)


overwrite_lowering(aten.slice, npu_slice, type_promotion_kind=None)


_upstream_slice_scatter = torch._inductor.lowering.slice_scatter


def npu_slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    """NPU slice_scatter, two structural departures:

    1. Unbacked bounds -> eager fallback (upstream's ``u0 < 0`` is an unguardable
       data-dependent branch; select_scatter guards it, slice_scatter doesn't).
    2. step==1, start>0 -> route as cat([x[:A], src, x[B:]]) via npu_cat_loop:
       upstream's blend at ``FloorDiv(idx-start, step)`` has a negative base
       ``(-start)+x0`` that miscompiles once the axis splits across blocks (wrong
       numerics, or fails to compile). Gated on slice_scatter_via_cat_loop.
    """
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    def _has_unbacked(v):
        if v is None:
            return False
        return bool(free_unbacked_symbols(sympy.expand(v)))

    if _has_unbacked(start) or _has_unbacked(end):
        return fallback_handler(aten.slice_scatter.default)(
            x, src, dim, start, end, step
        )

    if ncfg.slice_scatter_via_cat_loop and step == 1:
        maybe = _npu_slice_scatter_via_cat(x, src, dim, start, end)
        if maybe is not None:
            return maybe

    return _upstream_slice_scatter(x, src, dim=dim, start=start, end=end, step=step)


def _npu_slice_scatter_via_cat(x, src, dim, start, end):
    """Route a step-1 slice_scatter to npu_cat_loop as cat([x[:A], src, x[B:]]).
    Returns None (use upstream) for start==0 / full replace / non-static bounds;
    only start>0 with statically-known bounds is rerouted."""
    from torch._inductor.lowering import slice_ as _slice, expand

    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    start, end = ir.SliceView.normalize_start_end(x, dim, start, end)

    sv = V.graph.sizevars
    if sv.statically_known_equals(start, 0):
        return None
    if not sv.statically_known_lt(start, end):
        return None
    band_len = end - start
    if not sv.statically_known_lt(0, band_len):
        return None

    # Pieces: x[:start], src, x[end:] (dropped if band reaches the end); total
    # extent == dim_size, reproducing x with the band replaced by src.
    src = to_dtype(src, x.get_dtype())
    band = list(x.get_size())
    band[dim] = band_len
    src = expand(src, band)

    pieces = [_slice(x, dim, 0, start), src]
    if not sv.statically_known_equals(end, dim_size):
        pieces.append(_slice(x, dim, end, dim_size))

    return npu_cat_loop(pieces, dim)



overwrite_lowering(aten.slice_scatter, npu_slice_scatter, type_promotion_kind=None)


# bishengir degrades a tail-axis-broadcast fused pointwise to SCALAR when the
# broadcast inner axis is below the vector width (910B3: len 8 -> 4us, len 4 ->
# 60+us). Realize into its own buffer only when strictly below threshold.
NPU_TAIL_BCAST_MIN_VEC = ncfg.tail_bcast_min_vec


def _is_tail_axis_broadcast(x, sizes):
    """True iff expanding x to sizes broadcasts the inner dim and the target inner
    length is in the slow scalar-degenerate range (1 < len < NPU_TAIL_BCAST_MIN_VEC).
    ExpandView has no get_stride(), so inspect x's own size vs the target sizes."""
    try:
        in_size = x.get_size()
    except Exception:
        return False
    if len(in_size) == 0 or len(sizes) < 2 or len(in_size) != len(sizes):
        return False
    in_inner = in_size[-1]
    tgt_inner = sizes[-1]
    if not V.graph.sizevars.statically_known_equals(in_inner, sympy.Integer(1)):
        return False
    if not isinstance(tgt_inner, (int, sympy.Integer)):
        return False
    tgt_inner = int(tgt_inner)
    return 1 < tgt_inner < NPU_TAIL_BCAST_MIN_VEC


from torch._inductor.lowering import expand as _orig_expand


def npu_expand(x, sizes):
    """NPU expand: realize a SHORT tail-axis broadcast into its own contiguous
    buffer so the consumer reads a full [s,c] operand and bishengir keeps the vector
    path. Pinned no-fuse. Gated by realize_tail_bcast (see _is_tail_axis_broadcast)."""
    result = _orig_expand(x, sizes)
    if not ncfg.realize_tail_bcast:
        return result
    if not _is_tail_axis_broadcast(x, sizes):
        return result
    # realize() on the ExpandView realizes its underlying storage, not the [s,c]
    # result; wrap it in a Pointwise copy and realize THAT to materialize the
    # broadcast as its own buffer.
    try:
        copied = Pointwise.create(
            device=result.get_device(),
            dtype=result.get_dtype(),
            inner_fn=result.make_loader(),
            ranges=list(result.get_size()),
        )
        name = copied.realize()
    except Exception as e:
        log.debug("[NPU] realize tail-bcast expand skipped: %r", e)  # noqa: G200
        return result
    if name:
        V.graph.no_fuse_buffer_names.add(name)
        log.debug("[NPU] realize tail-axis broadcast expand: buf=%s in_size=%s target=%s",
                  name, x.get_size(), list(sizes))
    return copied


overwrite_lowering(aten.expand, npu_expand, type_promotion_kind=None)
# broadcast_tensors calls the module-level ``expand`` directly (not the FX table),
# so rebind it too.
torch._inductor.lowering.expand = npu_expand


_upstream_permute = torch._inductor.lowering.permute


def _permuted_inner_stride(x, dims):
    """Innermost stride the permuted view would carry (input_stride[dims[-1]]), or
    None if not statically determinable. Read from the input, not the view (a fresh
    PermuteView over an unrealized producer may not answer get_stride())."""
    try:
        in_stride = x.get_stride()
    except Exception:
        return None
    if not dims or len(dims) != len(in_stride):
        return None
    last_src = dims[-1]
    if last_src < 0 or last_src >= len(in_stride):
        return None
    return in_stride[last_src]


def npu_permute(x, dims):
    """NPU permute: realize a permute that pushes a NON-UNIT stride onto the inner
    axis into a contiguous buffer, instead of folding it into the consumer's loads
    as an inner-axis strided gather (which ascend degrades to a scalar gather; T5
    fwd softmax pos_bias 279us->30us). OFF by default (realize_permute_gather): a
    non-unit inner stride is often fine (transpose-for-matmul). When on: inner
    stride statically != 1, inner length statically > 1, input an unrealized
    producer or plain buffer; else delegate to upstream permute."""
    result = _upstream_permute(x, dims)
    if not ncfg.realize_permute_gather:
        return result
    if not isinstance(x, ir.TensorBox) or not isinstance(result, ir.TensorBox):
        return result

    inner_stride = _permuted_inner_stride(x, list(dims))
    if inner_stride is None:
        return result
    # Realize unless the inner stride is PROVABLY unit (the no-gather fast path).
    if isinstance(inner_stride, (int, sympy.Integer)):
        if int(inner_stride) == 1:
            return result
    else:
        try:
            if V.graph.sizevars.statically_known_equals(
                inner_stride, sympy.Integer(1)
            ):
                return result
        except Exception:
            pass

    # Inner axis must not be a provable singleton (size-1 inner is not a gather).
    try:
        res_size = result.get_size()
    except Exception:
        return result
    if len(res_size) == 0:
        return result
    inner_len = res_size[-1]
    if isinstance(inner_len, (int, sympy.Integer)):
        if int(inner_len) <= 1:
            return result
    else:
        try:
            if V.graph.sizevars.statically_known_leq(inner_len, sympy.Integer(1)):
                return result
        except Exception:
            pass

    # Only realize an unrealized producer or plain buffer, never a reinterpret view.
    data = x.data
    is_realizable = isinstance(data, ir.StorageBox) and isinstance(
        getattr(data, "data", None), (ir.Pointwise, ir.Reduction, ir.Buffer)
    )
    if not is_realizable:
        return result

    try:
        copied = Pointwise.create(
            device=result.get_device(),
            dtype=result.get_dtype(),
            inner_fn=result.make_loader(),
            ranges=list(result.get_size()),
        )
        name = copied.realize()
    except Exception as e:
        log.debug("[NPU] realize permute-gather skipped: %r", e)  # noqa: G200
        return result
    if name:
        V.graph.no_fuse_buffer_names.add(name)
        log.debug("[NPU] realize permute inner-stride gather: buf=%s dims=%s inner_stride=%s out_size=%s",
                  name, list(dims), inner_stride, list(result.get_size()))
    return copied


overwrite_lowering(aten.permute, npu_permute, type_promotion_kind=None)
