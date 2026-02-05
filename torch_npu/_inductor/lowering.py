import sympy
import torch._ops
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.ir import ExpandView, TensorBox, ops_wrapper
from torch._inductor.ir import Reduction
from torch._inductor.lowering import sum_, _validate_dim, get_promoted_dtype
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
    div,
    squeeze,
    square,
    sub,
    fallback_handler,
    logical_and,
    make_pointwise,
    _make_reduction_inner,
    _validate_reduction_axis,
    add_needs_realized_inputs,
    add_layout_constraint
)
import torch_npu
from torch_npu import npu_dtype_cast, _npu_dtype_cast
from .lowering_op_list import GENERATE_LIST, GENERATE_LIST2, FALLBACK_LIST, LOWERING_OVERLOAD_OP


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


def make_reduction(reduction_type: str, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
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


def _register_npu_inductor_fallbacks():
    gen_set = set()
    _init_set(GENERATE_LIST, gen_set)
    overload_op_set = set()
    _init_set(LOWERING_OVERLOAD_OP, overload_op_set)

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
                    FALLBACK_LIST.append(op)
    for op in FALLBACK_LIST:
        make_fallback(op)

    # 把需要overload的op在lowering里删除
    for op in overload_op_set:
        if op in lowerings:
            del lowerings[op]

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
        cpu_device = inputs[0].get_device().type == "cpu"
        if cpu_device and all(
            inp.get_dtype() in [torch.int8, torch.uint8] for inp in inputs
        ):
            for inp in inputs:
                inp.realize()
            if all(len(inp.get_size()) == 4 for inp in inputs):
                inputs, _ = require_channels_last(aten.cat, *inputs)
            return fallback_handler(aten.cat.deault)(inputs, dim)
        if len(inputs) == 1:
            return clone(inputs[0])
        dim = _validate_dim(inputs[0], dim, 0)
        dtype = get_promoted_dtype(
            *inputs,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

        )
        inputs = [to_dtype(inp, dtype) for inp in inputs]
        return TensorBox(ir.ConcatKernel.create(inputs, dim))