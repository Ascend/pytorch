import os
import sympy
from functools import reduce

import torch._ops
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.ir import ExpandView, TensorBox, ops_wrapper
from torch._inductor.ir import Reduction
from torch._inductor.lowering import sum_ as sum__pt, clone
from torch._inductor.utils import sympy_product
from torch._prims_common import (
    is_boolean_dtype,
    is_integer_dtype,
    get_computation_dtype,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
)
from torch._inductor.lowering import (
    lowerings,
    make_fallback,
    register_lowering,
    to_dtype,
    fallback_cumsum,
    _validate_reduction_axis,
    div as div_pt,
    squeeze as squeeze_pt,
    square as square_pt,
    sub as sub_pt,
    fallback_handler,
    logical_and,
    make_pointwise,
    _make_reduction_inner,
    _validate_reduction_axis,
    add_needs_realized_inputs,
    add_layout_constraint,
    require_channels_last,
    _validate_dim as _validate_dim_pt,
    get_promoted_dtype,
)
from .. import npu_dtype_cast, _npu_dtype_cast
from .config import log, enable_full_lowering_fallback
from .lowering_op_list import GENERATE_LIST, GENERATE_LIST2, FALLBACK_LIST, LOWERING_OVERLOAD_OP
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
        clone,
        to_dtype
    )

    LOWERING_OVERLOAD_OP = list(set(GENERATE_LIST) | set(LOWERING_OVERLOAD_OP))


def make_reduction(reduction_type: str, override_return_dtype=None):
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
                result.data.data, Reduction
        ):  # Only realize if reduction isn't unrolled
            size = x.get_size()
            axis = set(_validate_reduction_axis(x, axis))
            kept_idx = []
            reduced_idx = []
            for i in range(len(size)):
                if i in axis:
                    reduced_idx.append(i)
                else:
                    kept_idx.append(i)

            object.__setattr__(result.data.data, "kept_idx", kept_idx)
            object.__setattr__(result.data.data, "reduced_idx", reduced_idx)

            result.realize()
        return result

    return inner

lowering.make_reduction = make_reduction

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims


def _init_set(input_list, output_set):
    for fn in input_list:
        output_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                output_set.add(other_fn)


def _resolve_op_from_name(op_name: str):
    try:
        obj = torch.ops
        for part in op_name.split('.'):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        log.warning(f"[npu|inductor|lowering|fallback] invalid identifier name: {op_name}")
        return None


def _register_npu_inductor_fallbacks():
    gen_set = set()
    _init_set(GENERATE_LIST, gen_set)
    overload_op_set = set()
    _init_set(LOWERING_OVERLOAD_OP, overload_op_set)
    env_fallback_list = enable_full_lowering_fallback
    if env_fallback_list:
        for op_name in env_fallback_list.split(','):
            op_name = op_name.strip()
            op = _resolve_op_from_name(op_name)
            if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                FALLBACK_LIST.append(op)
                log.info(f"[npu|inductor|lowering|fallback] User specified fallback: {op_name}")
            else:
                log.warning(f"[npu|inductor|lowering|fallback] Cannot resolve operator: {op_name}")
    # 算子fallback
    for op in lowering.lowerings:
        if op in FALLBACK_LIST and op not in decompositions \
            and isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            make_fallback(op)
    # 把不在白名单的op fallback
    for op in lowerings:
        if op not in decompositions and op not in gen_set:
            if isinstance(op, torch._ops.OpOverloadPacket) or \
                    isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                flag = False
                for gens in GENERATE_LIST2:
                    if str(op).find(gens) != -1:
                        flag = True
                if flag:
                    continue
                else:
                    make_fallback(op)
                    FALLBACK_LIST.append(op)
    # 把需要overload的op在lowering里删除
    for op in overload_op_set:
        if op in lowerings:
            del lowerings[op]
            
    if npu_config.dump_fx_graph:
        from .lowering_fx import _register_npu_inductor_fallbacks_fx
        (squeeze, _validate_dim, div, square, sub, sum_) = _register_npu_inductor_fallbacks_fx(make_reduction)
    else:
        (squeeze, _validate_dim, div, square, sub, sum_) = (squeeze_pt, _validate_dim_pt, div_pt, square_pt, sub_pt, sum__pt)
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

    @register_lowering(aten.cumsum)
    def cumsum(x, axis=None, dtype=None):
        if (
                is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            # torch.int64->torch.int32
            dtype = torch.int32
        if len(x.get_size()) == 0:
            if axis not in [0, -1]:
                raise ValueError("axis must be 0 or -1")
            dtype = dtype or x.get_dtype()
            return to_dtype(x, dtype, copy=True)
        return fallback_cumsum(x, dim=axis, dtype=dtype)

    @register_lowering(npu_dtype_cast, type_promotion_kind=None)
    def _convert_npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    @register_lowering(_npu_dtype_cast, type_promotion_kind=None)
    def _convert__npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

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
        output = (
            var_mean_sum_(**kwargs)
        )
        output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
        return output[0] if not return_mean else output

    @register_lowering(aten.var_mean)
    def var_mean(x, axis=None, *, correction=None, keepdim=False):
        return var_mean_helper_(
            x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
        )

    @register_lowering([aten.var, prims.var])
    def var_(x, axis=None, *, correction=None, keepdim=False):
        return var_mean_helper_(
            x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False
        )

    @register_lowering(aten.embedding, type_promotion_kind=None)
    def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
        return fallback_handler(aten.embedding.default)(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
                                                        sparse=False)

    @register_lowering(aten.cat)
    def cat(inputs, dim=0):
        if len(inputs) == 1:
            return clone(inputs[0])
        dim = _validate_dim(inputs[0], dim, 0)
        dtype = get_promoted_dtype(
            *inputs,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

        )
        inputs = [to_dtype(inp, dtype) for inp in inputs]
        return TensorBox(ir.ConcatKernel.create(inputs, dim))

    make_fallback(aten._log_softmax)
    make_fallback(aten.gather)
    make_fallback(aten.nll_loss_forward)


def get_nested_attr(obj, attr_path, default=None):
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default
    

def _fallback_ops_with_meta():
    """
    Fallback all ops that have a Meta implementation but are not yet in lowerings
    """
    all_ops = torch._C._dispatch_get_all_op_names()

    for op_name in all_ops:
        has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta")
        has_comp = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "CompositeImplicitAutograd")

        if not (has_meta or has_comp):
            continue

        try:
            namespace, name_with_overload = op_name.split(".", 1)
        except ValueError:
            continue

        if "." in name_with_overload:
            name, overload = name_with_overload.split(".", 1)
        else:
            name, overload = name_with_overload, "default"
            
        normalized_path = f"{namespace}.{name}.{overload}"
        op_overload = get_nested_attr(torch.ops, normalized_path)
        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        if op_overload in lowerings or op_overload in decompositions:
            continue

        make_fallback(op_overload)
        FALLBACK_LIST.append(op_overload)


def _enable_full_lowering_fallback():
    ops_to_fallback = list(filter(
        lambda op: op not in decompositions and
            isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)) and
            op not in (torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
                       torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional),
        lowerings
    ))
    for op in ops_to_fallback:
        make_fallback(op)
        FALLBACK_LIST.append(op)
    _fallback_ops_with_meta()
