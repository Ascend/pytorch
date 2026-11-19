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

import os
import sympy

import torch._ops
from torch._inductor import config as inductor_config
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.ir import ExpandView, TensorBox, ops_wrapper, StorageBox, View
from torch._inductor.ir import Reduction, Pointwise
from torch._inductor.lowering import (
    lowerings,
    make_fallback,
    register_lowering,
    to_dtype,
    fallback_cumsum,
    div as div_pt,
    squeeze as squeeze_pt,
    square as square_pt,
    sub as sub_pt,
    sum_,
    fallback_handler,
    _make_reduction_inner,
    _validate_reduction_axis,
    add_needs_realized_inputs,
    add_layout_constraint,
    require_channels_last,
    _validate_dim as _validate_dim_pt,
    get_promoted_dtype,
    add as add_pt,
    rsqrt as rsqrt_pt,
    mul as mul_pt,
    sqrt as sqrt_pt,
    clone as clone_pt,
    pow_recursive,
    exp2 as exp2_pt, make_pointwise, pointwise_cat
)
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import sympy_product
from torch._prims_common import (
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    get_computation_dtype,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number
)

from torch._inductor.lowering import clone, full_like as full_like_pt
from torch._inductor.virtualized import V, ops
from torch._inductor import scheduler
from torch._inductor.scheduler import Scheduler
from .. import npu_dtype_cast, _npu_dtype_cast
from . import ir as npu_ir
from .lowering_common import (
    add_overload as _add_overload,
    enable_full_lowering_fallback as enable_full_lowering_fallback_common,
    resolve_op_from_name,
)
from .lowering_override_list import LOWERING_OVERRIDE_OP
from .config import log, is_ascend950, enable_full_lowering_fallback

from .lowering_fallback_list import FALLBACK_LIST, NPU_EXTRA_FALLBACK_LIST

from . import config as npu_config
from .lowering_fx import (
    fetch_graphs,
    merge_traced_graphs,
    node_id,
    create_fake_input,
    subtract_graph,
    create_fx_from_snodes_by_traced_graph,
    create_compile_kwargs,
    generate_fx_graph_code,
    dump_fx_graph_code,
    snodes_to_fx,
    )


def npu_make_fallback(op, layout_constraint=None, warn=True, override_decomp=False):
    if op in decompositions and not override_decomp:
        raise RuntimeError(f"both a fallback and a decomp for same op: {op}")

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

make_fallback = npu_make_fallback

if npu_config.dump_fx_graph:
    from .lowering_fx import (
        _make_reduction_inner,
        reduction_type_to_aten_fn,
        npu_compute_ancestors,
        _npu_prune_redundant_deps,
        _npu_get_unmet_dep_nodes,
        clone,
        to_dtype,
        pointwise_cat,
        DUMP_FX_GRAPH_LOWERING_OPS
    )
    LOWERING_OVERRIDE_OP = list((set(LOWERING_OVERRIDE_OP) | set(DUMP_FX_GRAPH_LOWERING_OPS)) - set(FALLBACK_LIST))

    Scheduler.compute_ancestors = npu_compute_ancestors
    scheduler._prune_redundant_deps = _npu_prune_redundant_deps
    Scheduler._get_unmet_dep_nodes = _npu_get_unmet_dep_nodes


def make_reduction(reduction_type: ReductionType, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        if npu_config.dump_fx_graph:
            node_name = f'reduction_{next(node_id)}'
            input_graphs = fetch_graphs([x, axis if axis is not None else list(range(len(x.get_size())))])
            new_graph = merge_traced_graphs(input_graphs, reduction_type_to_aten_fn[reduction_type],
                                            node_name, keepdim=keepdims)
            result = Reduction.create(reduction_type=reduction_type,
                                    input_node=x,
                                    node_name=node_name,
                                    traced_graph=new_graph,
                                    **kwargs)
        else:
            result = Reduction.create(reduction_type=reduction_type,
                                        input_node=x,
                                        **kwargs)
        if isinstance(
            result.data.data,  # type: ignore[attr-defined]
            Reduction,
        ):  # Only realize if reduction isn't unrolled
            result.realize()
        return result

    return inner

lowering.make_reduction = make_reduction

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu

def _npu_fractional_pooling_offsets(
    samples,
    in_sz,
    out_sz,
    kernel_sz,
    dim,
    ndims,
):
    out_sz = out_sz[dim]
    in_sz = in_sz[dim]
    kernel_sz = kernel_sz[dim]
    samples_loader = samples.make_loader()
    sample_dim = dim if ndims == 3 else ndims - 1 - dim
    dtype = samples.get_dtype()
    compute_dtype = get_computation_dtype(dtype)

    def round_to_input_dtype(value):
        # Preserve CPU eager scalar_t rounding before integer truncation.
        value = ops.to_dtype(value, dtype, use_compute_types=False)
        return ops.to_dtype(value, compute_dtype)

    def load(prefix, i):
        samples_shape = samples.get_size()

        if len(samples_shape) == 3:
            if len(prefix) == 1:
                sample = samples_loader([0, prefix[0], sample_dim])
            elif len(prefix) >= 2:
                sample = samples_loader(
                    [prefix[0], prefix[1], sample_dim]
                )
            else:
                sample = samples_loader([0, 0, sample_dim])
        else:
            sample = samples_loader([*prefix, sample_dim])

        i_index = ops.index_expr(i, torch.int64)
        i_expr = round_to_input_dtype(i_index)
        diff = ops.index_expr(in_sz - kernel_sz, torch.int64)
        out_sz_expr = ops.index_expr(out_sz - 1, torch.int64)
        safe_den = ops.where(ops.eq(out_sz_expr, 0), 1, out_sz_expr)
        alpha = round_to_input_dtype(
            ops.truediv(
                round_to_input_dtype(diff),
                round_to_input_dtype(safe_den),
            )
        )
        alpha = ops.where(ops.eq(out_sz_expr, 0), 0, alpha)
        shifted = round_to_input_dtype(i_expr + sample)
        left = round_to_input_dtype(shifted * alpha)
        right = round_to_input_dtype(sample * alpha)
        left = ops.to_dtype(ops.trunc(left), torch.int64)
        right = ops.to_dtype(ops.trunc(right), torch.int64)
        seq_i = left - right
        mask = ops.lt(i_index, out_sz_expr)
        return ops.indirect_indexing(
            ops.where(mask, seq_i, diff), sympy.sympify(in_sz)
        )

    return load


def _npu_fractional_max_pool(
    x,
    kernel_size,
    output_size,
    random_samples,
    n_dim,
):
    x.realize_hint()
    batch, inp_dhw = x.shape[:-n_dim], x.shape[-n_dim:]

    with inductor_config.patch(unroll_reductions_threshold=25):
        dhw_index_fn = [
            _npu_fractional_pooling_offsets(
                samples=random_samples,
                in_sz=inp_dhw,
                out_sz=output_size,
                kernel_sz=kernel_size,
                ndims=n_dim,
                dim=d,
            )
            for d in range(n_dim)
        ]

        x_loader = x.make_loader()

        def increments_to_index(idx, reduction_idx):
            prefix = idx[:-n_dim]
            bdhw = idx[-n_dim:]
            return [
                dhw_index_fn[d](prefix, bdhw[d]) + reduction_idx[d]
                for d in range(n_dim)
            ]

        def fn_inner(idx, reduction_idx):
            prefix = idx[:-n_dim]
            return x_loader(
                [*prefix, *increments_to_index(idx, reduction_idx)]
            )

        new_size = list(batch) + list(output_size)
        dtype = x.get_dtype()
        result = Reduction.create(
            reduction_type="max",
            input_node=x,
            device=x.get_device(),
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        offsets = Reduction.create(
            reduction_type="argmax",
            input_node=x,
            device=x.get_device(),
            dst_dtype=torch.int64,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        if not isinstance(result, TensorBox):
            raise RuntimeError(
                f"fractional_max_pool3d result must be a TensorBox, got {type(result)}"
            )
        if isinstance(result.data.data, Reduction):
            result.realize()
        if not isinstance(offsets, TensorBox):
            raise RuntimeError(
                f"fractional_max_pool3d offsets must be a TensorBox, got {type(offsets)}"
            )
        if isinstance(offsets.data.data, Reduction):
            offsets.realize()

        indices = lowering._pool_offsets_to_indices(
            offsets, kernel_size, x.shape, increments_to_index
        )
        return result, indices

def _register_npu_inductor_fallbacks():
    _torch_searchsorted = lowering.searchsorted

    @register_lowering(aten.searchsorted.Tensor, type_promotion_kind=None)
    def searchsorted(
        sorted_sequence: TensorBox,
        self: TensorBox,
        *,
        out_int32: bool = False,
        right: bool = False,
        side=None,
        sorter=None,
    ):
        # The native Inductor lowering queries physical strides for boundaries and
        # an optional sorter. Generic views such as SliceView only carry a reindex
        # function, so realize() materializes the backing storage but does not make
        # get_stride() available on the existing view object. Build and use a new
        # contiguous IR value whenever the view has no physical stride.
        if sorted_sequence.maybe_get_stride() is None:
            sorted_sequence = ir.ExternKernel.require_contiguous(sorted_sequence)
        if sorter is not None and sorter.maybe_get_stride() is None:
            sorter = ir.ExternKernel.require_contiguous(sorter)
        return _torch_searchsorted(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    env_fallback_list = enable_full_lowering_fallback
    if env_fallback_list:
        for op_name in env_fallback_list.split(','):
            op_name = op_name.strip()
            op = resolve_op_from_name(op_name, log)
            if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                FALLBACK_LIST.append(op)
                if op in LOWERING_OVERRIDE_OP:
                    LOWERING_OVERRIDE_OP.remove(op)
                log.info(f"[npu|inductor|lowering|fallback] User specified fallback: {op_name}")
            else:
                log.warning(f"[npu|inductor|lowering|fallback] Cannot resolve operator: {op_name}")

    # 把需要overload的op在lowering里删除
    overload_op_set = set()
    _add_overload(LOWERING_OVERRIDE_OP, overload_op_set)

    for op in overload_op_set:
        if op in lowerings:
            del lowerings[op]

    @register_lowering(aten.fractional_max_pool3d)
    def fractional_max_pool3d(x, kernel_size, output_size, random_samples):
        return _npu_fractional_max_pool(
            x,
            kernel_size,
            output_size,
            random_samples,
            n_dim=3,
        )

    if npu_config.dump_fx_graph:
        from .lowering_fx import _register_npu_inductor_fallbacks_fx
        (squeeze, _, _, _, _validate_dim, full_like, mul, div, rsqrt, add, square, sub) = _register_npu_inductor_fallbacks_fx(make_reduction)
    else:
        (squeeze, _validate_dim, full_like, mul, div, rsqrt, add, square, sub) = (squeeze_pt, _validate_dim_pt, full_like_pt, mul_pt, div_pt, rsqrt_pt, add_pt, square_pt, sub_pt)

    # fallback
    for op in lowering.lowerings:
        if op in FALLBACK_LIST and op not in decompositions \
            and isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            make_fallback(op)
    log.info(f"[npu|inductor|lowering|fallback] with FALLBACK_LIST, len(lowerings): {len(lowerings)}, "
                f"len(FALLBACK_LIST): {len(FALLBACK_LIST)}, make_fallback finished.")
    log.info(f"[npu|inductor|lowering|fallback] len(NPU_EXTRA_FALLBACK_LIST): {len(NPU_EXTRA_FALLBACK_LIST)}")

    _add_fallback_ops_for_torchgen()
    # register the reductions useing custom make_reduction
    reduce_amax = register_lowering(aten.amax)(make_reduction("max"))
    reduce_amin = register_lowering(aten.amin)(make_reduction("min"))
    reduce_argmax = register_lowering(aten.argmax)(
        make_reduction("argmax", override_return_dtype=torch.int64)
    )
    reduce_argmin = register_lowering(aten.argmin)(
        make_reduction("argmin", override_return_dtype=torch.int64)
    )

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

    @register_lowering(aten.mean)
    def mean(x, axis=None, keepdim=False, *, dtype=None):
        if dtype is not None:
            x = to_dtype(x, dtype)
        size = x.get_size()
        axis = _validate_reduction_axis(x, axis)
        # compute in higher-precision until end of mean lowering
        output_dtype = x.get_dtype()
        if output_dtype in (torch.float16, torch.bfloat16):
            x = to_dtype(x, torch.float)
        sum_result = sum_(x, axis, keepdim)
        denom = sympy_product(size[i] for i in axis)
        denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
        denom = ExpandView.create(denom, list(sum_result.get_size()))
        return to_dtype(div(sum_result, denom), output_dtype)

    @register_lowering(npu.npu_dtype_cast, type_promotion_kind=None)
    def _convert_npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    @register_lowering(npu._npu_dtype_cast, type_promotion_kind=None)
    def _convert__npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    @register_lowering(prims.convert_element_type, type_promotion_kind=None)
    def _convert_element_type(x: TensorBox, dtype: torch.dtype):
        if dtype == torch.float64 or x.get_dtype() == torch.float64:
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)

        if dtype.is_complex or x.get_dtype().is_complex:
            if x.get_size():
                dst = full_like(x, 0, dtype=dtype)
                ir.InplaceCopyFallback.create(dst, x)
                return dst
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)

        return to_dtype(x, dtype, copy=True)

    @make_pointwise
    def pow_native(a,b):
        return ops.pow(a,b)

    fallback_pow_tensor_tensor = fallback_handler(
        aten.pow.Tensor_Tensor, add_to_fallback_set=False
    )
    fallback_pow_scalar = fallback_handler(aten.pow.Scalar, add_to_fallback_set=False)
    fallback_pow_tensor_scalar = fallback_handler(
        aten.pow.Tensor_Scalar, add_to_fallback_set=False
    )

    @register_lowering(aten.pow, broadcast=True)
    def pow(a, b):
        if isinstance(b, float) and b == int(b):
            return pow(a, int(b))
        elif isinstance(b, float) and b == 0.5:
            return sqrt_pt(a)
        elif isinstance(b, int) and b == 1:
            return clone_pt(a)

        # Type promotion ensures all tensor arguments have the same Type
        dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
        is_integer_pow = is_integer_dtype(dtype)
        is_fp64_pow = (dtype == torch.float64)

        # Optimize away small fixed powers, or for integers avoid falling back to Aten
        embed_exponent = isinstance(b, int) and (
            -32 < b < 32 or (is_integer_pow and b >= 0)
        )
        if embed_exponent:
            loader = a.make_loader()

            def fn(idx):
                return pow_recursive(loader(idx), b, a.get_dtype())

            return Pointwise.create(
                device=a.get_device(),
                dtype=a.get_dtype(),
                inner_fn=fn,
                ranges=a.get_size(),
            )

        if isinstance(a, Number):
            if a == 1:
                return full_like(b, 1)
            if a == 2 and is_float_dtype(b.get_dtype()):
                return exp2_pt(b)

        if is_integer_pow or is_fp64_pow:
            # ops.pow doesn't work for integers
            if isinstance(a, Number):
                return fallback_pow_scalar(a, b)
            elif isinstance(b, Number):
                return fallback_pow_tensor_scalar(a, b)
            else:
                return fallback_pow_tensor_tensor(a, b)

        return pow_native(a, b)



    # All the indexing decompositions are written in terms of index, index_put, and index_put_
    # We cannot have this lowering as a decomposition as it introduces
    # mutation in the graph, which is bad for Aot Autograd. Aot Autograd runs dead
    # code elimination and common subexpression elimination optimizations, which
    # assume graphs to be side-effect free. More details at

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

    def use_two_step_variance(x, axis, keepdim):
        # Instead of unrolling welford, just unroll the simpler two-step var
        axis = _validate_reduction_axis(x, axis)
        kwargs = _make_reduction_inner(
            x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
        )

        ranges = kwargs["ranges"]
        reduction_numel = sympy_product(kwargs["reduction_ranges"])
        return (
            isinstance(reduction_numel, sympy.Integer)
            and int(reduction_numel) < torch._inductor.config.unroll_reductions_threshold
            and sympy_product(ranges) != 1
        )

    def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
        if correction is None:
            correction = 1

        kwargs = _make_reduction_inner(
            x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
        )
        loader = kwargs.pop("inner_fn")
        kwargs.pop("dst_dtype")
        kwargs.pop("src_dtype")

        mean, m2, count = ir.WelfordReduction.create(
            inner_fns=(loader,),
            reduction_type="welford_reduce",
            dtype=x.get_dtype(),
            **kwargs,
        )
        # Welford is represented by three sibling IR nodes.  Keep their
        # identity so the NPU scheduler can fuse them before their consumers.
        from .choices import mark_welford_output_group

        m2.realize()

        dtype = x.get_dtype()
        size = x.get_size()
        axis = _validate_reduction_axis(x, axis)
        rnumel = sympy_product(size[i] for i in axis)

        def get_constant_or_index_expr(x, dtype):
            if isinstance(x, sympy.Expr) and not x.is_number:
                return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
            return ops.constant(x, dtype)

        def scale_fn(data):
            c = get_constant_or_index_expr(correction, dtype)
            N = get_constant_or_index_expr(rnumel, dtype)
            zero = ops.constant(0, dtype)
            return data / ops.maximum(zero, N - c)

        var = make_pointwise(scale_fn)(m2)

        if return_mean:
            mean.realize()
        mark_welford_output_group((mean, m2))
        if return_mean:
            return var, mean
        return (var,)

    def var_mean_helper_(
        x, *, axis, correction, keepdim, return_mean, allow_welford
    ):
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
        output = (
            var_mean_sum_(**kwargs)
            if not allow_welford
            or not npu_config.enable_welford
            or use_two_step_variance(x, axis=axis, keepdim=keepdim)
            else var_mean_welford_(**kwargs)
        )
        output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
        return output[0] if not return_mean else output

    @register_lowering(aten.var_mean)
    def var_mean(x, axis=None, *, correction=None, keepdim=False):
        return var_mean_helper_(
            x,
            axis=axis,
            correction=correction,
            keepdim=keepdim,
            return_mean=True,
            allow_welford=True,
        )

    @register_lowering([aten.var, prims.var])
    def var_(x, axis=None, *, correction=None, keepdim=False):
        # A standalone var can fuse with a separate mean reduction. Welford then
        # creates a dual-reduction kernel that falls back from the NPU linear path.
        return var_mean_helper_(
            x,
            axis=axis,
            correction=correction,
            keepdim=keepdim,
            return_mean=False,
            allow_welford=False,
        )

    @register_lowering(aten.cat)
    def cat(inputs, dim=0):
        cpu_device = inputs[0].get_device().type == "cpu"
        if cpu_device and all(
            input.get_dtype() in [torch.int8, torch.uint8] for input in inputs
        ):
            return lowering.cat(inputs)

        if len(inputs) == 1:
            return clone(inputs[0])

        if not is_ascend950 or len(inputs) > torch._inductor.config.max_pointwise_cat_inputs:
            return fallback_handler(aten.cat.default)(inputs, dim)

        dim = _validate_dim(inputs[0], dim, 0)
        dtype = get_promoted_dtype(
            *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        )
        inputs = [to_dtype(inp, dtype) for inp in inputs]

        # for NPU, we never cat tensor with mask+where, except config.force_pointwise_cat = True
        if torch._inductor.config.force_pointwise_cat:
            return pointwise_cat(inputs, dim)

        return TensorBox(ir.ConcatKernel.create(inputs, dim))

    @register_lowering(aten.native_layer_norm)
    def native_layer_norm(
        x,
        normalized_shape,
        weight=None,
        bias=None,
        eps=1e-5
    ):
        def is_transposed_layer_norm_consumer():
            """Identify the LN2 topology that benefits from LayerNormV4."""
            node = getattr(V.graph, "current_node", None)
            if node is None or node.target != torch.ops.aten.native_layer_norm.default:
                return False
            value = node.args[0] if node.args else None
            view_ops = {
                torch.ops.aten.view.default,
                torch.ops.aten._unsafe_view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.permute.default,
                torch.ops.aten.clone.default,
                torch.ops.aten.contiguous.default,
            }
            saw_permute = False
            saw_reshape = False
            for _ in range(10):
                if not hasattr(value, "target"):
                    return False
                if value.target == torch.ops.aten.native_layer_norm.default:
                    return saw_permute and saw_reshape
                if value.target not in view_ops or not value.args:
                    return False
                saw_permute |= value.target == torch.ops.aten.permute.default
                saw_reshape |= value.target in {
                    torch.ops.aten.view.default,
                    torch.ops.aten._unsafe_view.default,
                    torch.ops.aten.reshape.default,
                }
                value = value.args[0]
            return False

        def should_use_layer_norm_v4():
            """Select LayerNormV4 only for its known A5 regression cases."""
            if (
                not is_ascend950
                or not npu_config.enable_welford
                or not npu_config.enable_layernorm_v4
            ):
                return False
            if x.dtype not in (torch.float16, torch.bfloat16):
                return False

            from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

            if not isinstance(normalized_shape, (list, tuple)):
                shape = (normalized_shape,)
            else:
                shape = tuple(normalized_shape)

            input_shape = x.get_size()
            if free_unbacked_symbols(input_shape) or free_unbacked_symbols(shape):
                return True

            if len(shape) != 1 or shape[0] not in (512, 640):
                return False

            if len(input_shape) < 2:
                return False
            row_numel = sympy_product(input_shape[:-1])
            if shape[0] == 640:
                return (
                    len(input_shape) == 3
                    and input_shape[-2] == 80
                    and is_transposed_layer_norm_consumer()
                )
            return not V.graph.sizevars.statically_known_lt(row_numel, 512)

        # Keep the low-precision fallback when Welford is disabled.  The
        # custom lowering is used only when FP16/BF16 Welford has been
        # explicitly enabled on Ascend950.
        if (
            is_ascend950
            and x.dtype in (torch.float16, torch.bfloat16)
            and (
                not npu_config.enable_welford
                or should_use_layer_norm_v4()
            )
        ):
            return fallback_handler(aten.native_layer_norm.default)(
                x, normalized_shape, weight, bias, eps
            )

        # Validate input
        if not isinstance(normalized_shape, (list, tuple)):
            normalized_shape = (normalized_shape,)

        normalized_ndim = len(normalized_shape)
        input_shape = x.get_size()

        # Calculate reduction dimension indices
        reduce_dims = list(range(len(input_shape) - normalized_ndim, len(input_shape)))

        # Compute mean and variance
        var, mean = var_mean_helper_(
            x=x,
            axis=reduce_dims,
            correction=0,  # Layer normalization uses 0 correction (population variance)
            keepdim=True,  # Keep dimensions for broadcasting
            return_mean=True,
            allow_welford=True,
        )

        # Calculate normalized result (x - mean) / sqrt(var + eps)
        x_normalized = sub(x, mean)

        # Add eps to variance
        eps_tensor = ir.IndexingConstant(index=eps, dtype=var.get_dtype(), device=var.get_device())
        eps_tensor = ExpandView.create(eps_tensor, var.get_size())
        var_eps = add(var, eps_tensor)

        # Calculate reciprocal of sqrt(var + eps)
        inv_std = rsqrt(var_eps)  # 1 / sqrt(var + eps)

        # Normalization
        normalized = mul(x_normalized, inv_std)

        # Apply optional affine transformation (gamma * normalized + beta)
        if weight is not None:
            # weight will be broadcast automatically, mul function in lowering supports broadcasting
            normalized = mul(normalized, weight)

        if bias is not None:
            # add will be broadcast automatically
            normalized = add(normalized, bias)

        # native_layer_norm returns three values: output, mean, reciprocal of standard deviation
        return normalized, mean, inv_std


def _enable_full_lowering_fallback():
    enable_full_lowering_fallback_common(
        lowerings,
        decompositions,
        make_fallback,
        FALLBACK_LIST,
        excluded_ops=(
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
        ),
    )

def _add_fallback_ops_for_torchgen():
    import torchgen.aoti.fallback_ops as fallback_ops
    from torchnpugen.aoti.fallback_ops import inductor_fallback_ops_npu, inductor_fallback_ops_npu_not_support

    if getattr(ir.FallbackKernel.codegen, "_npu_device_aware", False):
        return

    # torchgen builds the cpu/cuda shims from the pristine table, while
    # gen_npu_c_shim builds the npu shims from the merged table below; keep a
    # pristine copy so non-NPU nodes can tell the difference.
    upstream = frozenset(fallback_ops.inductor_fallback_ops)

    fallback_ops.inductor_fallback_ops = {
        k: v for k, v in (fallback_ops.inductor_fallback_ops | inductor_fallback_ops_npu).items()
        if k not in inductor_fallback_ops_npu_not_support
    }

    # A non-NPU node of an NPU-only op would emit a symbol (e.g.
    # aoti_torch_cpu_scatter_add) that the build never has; route it through
    # the proxy executor instead.
    orig_codegen = ir.FallbackKernel.codegen

    def codegen(self, wrapper):
        kernel = self.op_overload
        if (
            V.graph.cpp_wrapper
            and not self.use_runtime_dispatch
            and isinstance(kernel, torch._ops.OpOverload)
            and kernel.namespace == "aten"
        ):
            device = self.get_device()
            if (device is None or device.type != "npu") and str(kernel) not in upstream:
                self.use_runtime_dispatch = True
        return orig_codegen(self, wrapper)

    codegen._npu_device_aware = True
    ir.FallbackKernel.codegen = codegen
