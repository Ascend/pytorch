import sympy
import torch._ops
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.ir import ExpandView, TensorBox, ops_wrapper, StorageBox, View
from torch._inductor.ir import Reduction, Pointwise
from torch._inductor.lowering import sum_
from torch._inductor.utils import sympy_product
from torch._prims_common import (
    is_boolean_dtype,
    is_integer_dtype,
    get_computation_dtype,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number
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
    is_boolean_type,
    logical_and,
    make_pointwise,
    _make_reduction_inner,
    _validate_reduction_axis,
    add_needs_realized_inputs,
    add_layout_constraint,
    require_channels_last,
    _validate_dim,
    get_promoted_dtype,
    add,
    rsqrt,
    mul
)

from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor.lowering import (unsqueeze, index_put_as_masked_fill, index_put_fallback, needs_fallback_due_to_atomic_add_limitations, view, check_and_broadcast_indices, index_output_size_and_inner_fn, expand, clone, new_empty, scatter_fallback, full_like)
from torch._inductor.virtualized import V, ops
from torch_npu.npu._backends import get_soc_version
from torch_npu import npu_dtype_cast, _npu_dtype_cast
from torch_npu.npu._backends import get_soc_version
from torch_npu._inductor import ir as npu_ir
from torch_npu._inductor.codegen.triton_utils import NPUKernelType
from .ir import IndexputTemplate, ScatterTemplate
from .lowering_op_list import GENERATE_LIST, GENERATE_LIST2, FALLBACK_LIST, LOWERING_OVERLOAD_OP
from .config import inductor_indirect_memory_mode, lowering_cat_with_concat_kernel


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
npu = torch.ops.npu


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
                    make_fallback(op)
                    FALLBACK_LIST.append(op)
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
        if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
            # torch.int64->torch.int32
            dtype = torch.int64 if get_soc_version() >= 250 else torch.int32
        if len(x.get_size()) == 0:
            if axis not in [0, -1]:
                raise ValueError("axis must be 0 or -1")
            dtype = dtype or x.get_dtype()
            return to_dtype(x, dtype, copy=True)
        return fallback_cumsum(x, dim=axis, dtype=dtype)

    @register_lowering(npu.npu_dtype_cast, type_promotion_kind=None)
    def _convert_npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    @register_lowering(npu._npu_dtype_cast, type_promotion_kind=None)
    def _convert__npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    def lowering_index_select(x, select_dim, indices, index_select_type):
        assert isinstance(x, TensorBox)
        assert isinstance(indices, TensorBox)
        assert "int" in str(indices.get_dtype())
        weight_loader = x.make_loader()
        indices_loader = indices.make_loader()
        indices_ndim = len(indices.get_size())
        x_size = x.get_size()
        new_size = [*x_size[:select_dim], *indices.get_size(), *x_size[select_dim + 1:]]

        def fn(idx):
            assert len(idx) == len(new_size), f"{idx} != {new_size}"
            is_indirect_idx = any(['tmp' in str(var) or 'indirect' in str(var) for var in idx])

            var_index = indices_loader(idx[select_dim:select_dim + indices_ndim])
            set_indirect = ops.indirect_indexing(var_index, x_size[select_dim])
            x_idx = [*idx[:select_dim]] + [set_indirect] + [*idx[select_dim + indices_ndim:]]
            if is_indirect_idx:
                return weight_loader(x_idx)
            try:
                index_loader = x.data.make_indexer()
                loader_name = x.data.get_name()
                return ops.index_select(loader_name, index_loader(x_idx), var_index, set_indirect, int(x_size[select_dim]), index_select_type)
            except Exception as e:
                return weight_loader(x_idx)

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=new_size,
        )

    @register_lowering(aten.embedding, type_promotion_kind=None)
    def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
        node = V.current_node
        if node.meta.get("skip_lowering", False):
            return fallback_handler(aten.embedding.default)(weight, indices, padding_idx=padding_idx, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)

        if inductor_indirect_memory_mode != str(NPUKernelType.SIMT_TEMPLATE):
            return lowering.embedding(weight, indices)

        def should_use_template():
            weight_size = weight.get_size()
            if 1 in weight_size:
                return False
            if isinstance(weight, TensorBox) and isinstance(weight.data, ir.BaseView):
                return False
            return True

        if should_use_template():
            return lowering_index_select(weight, 0, indices, 'embedding')
        return lowering.embedding(weight, indices)

    @register_lowering(aten.cat)
    def cat(inputs, dim=0):
        if len(inputs) == 1:
            return clone(inputs[0])

        if lowering_cat_with_concat_kernel:
            def is_reindex_view(x) -> bool:
                if isinstance(x, (TensorBox, ir.StorageBox)):
                    return is_reindex_view(x.data)
                if isinstance(x, ir.View) and "ModularIndexing" in x.reindex_str():
                    return True
                return False

            for inp in inputs:
                if is_reindex_view(inp):
                    return TensorBox(npu_ir.ConcatKernel.create(inputs, dim, True))

            input_dims = len(inputs[0].get_size())
            if input_dims > 1 and (dim == -1 or dim == input_dims - 1):
                return TensorBox(npu_ir.ConcatKernel.create(inputs, dim, False))
            else:
                return fallback_handler(aten.cat.default)(inputs, dim)
        else:
            dim = _validate_dim(inputs[0], dim, 0)
            dtype = get_promoted_dtype(
                *inputs,
                type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

            )
            inputs = [to_dtype(inp, dtype) for inp in inputs]
            return TensorBox(ir.ConcatKernel.create(inputs, dim))

    @register_lowering(aten.gather, type_promotion_kind=None)
    def gather(x, dim, index, sparse_grad=False):
        # sparse_grad doesn't affect forward computation,
        # and backward tracing is taken care of by AOT Autograd
        assert isinstance(x, TensorBox)
        if index.get_numel() == 0:
            # Empty index case. Return an empty array with the same shape
            return new_empty(x, index.get_size())

        assert index.get_dtype() == torch.int64
        size = x.get_size()
        offset = len(size) == 0
        dim = _validate_dim(x, dim, offset)

        if offset:
            x = expand(x, [1])
            size = [1]

        def should_use_template():
            template_x_dtypes = [torch.float32, torch.float16, torch.bfloat16]
            if x.get_dtype() not in template_x_dtypes:
                return False
            if 1 in x.get_size() or 1 in index.get_size():
                return False
            if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
                return False
            if isinstance(index, TensorBox) and isinstance(index.data, ir.BaseView):
                return False

            return True
        if not should_use_template():
            return lowering.gather(x, dim, index, sparse_grad)

        index_loader = index.make_loader()
        loader_name = x.data.get_name()
        x_loader = x.data.make_indexer()
        index_boundary = size[dim]

        def fn(idx):
            idx = list(idx)
            index_value = index_loader(idx)
            gather_idx = ops.indirect_indexing(index_value, size[dim])
            if len(idx) == 0:
                idx = [gather_idx]
            else:
                idx[dim] = gather_idx
            
            return ops.gather_template(loader_name, x_loader(idx), index_value, gather_idx, int(index_boundary))

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=index.get_size(),
        )

    
    def index_put_impl_(self, indices, values, accumulate, check, may_realize=False):
        if may_realize:

            def try_get_name(x):
                if isinstance(x, ir.TensorBox):
                    x = x.data
                if isinstance(x, ir.BaseView):
                    x = x.unwrap_view()
                if isinstance(x, ir.StorageBox):
                    x = x.data
                return x.get_name() if isinstance(x, ir.Buffer) else None

            def indice_slice_from_randperm(indice):
                # For this specific pattern, indices is unique as coming from torch.randperm.
                # However, as the content of the indices is unknown, we have to check this specific pattern.
                if isinstance(indice, TensorBox) and isinstance(indice.data, ir.BaseView):
                    indice = indice.data.unwrap_view()
                    return (
                        isinstance(indice, ir.StorageBox)
                        and isinstance(indice.data, ir.ExternKernel)
                        and getattr(indice.data, "fx_node", None)
                        and indice.data.fx_node.target == torch.ops.aten.randperm.default
                    )
                return False

            if try_get_name(self) in values.get_read_names() and not all(
                indice_slice_from_randperm(indice) for indice in indices
            ):
                # When self and values have memory overlapping, indices may
                # contain duplicate values, potentially causing incorrect results since
                # the load of `values` might contain modified value from the store of `self`.
                # To address this, store values in a temporary buffer in such cases.
                values.realize()

        # Dispatch to masked fill for single boolean index with single value
        if (
            values.get_numel() == 1
            and len(indices) == 1
            and indices[0].get_dtype() in (torch.bool, torch.uint8)
        ):
            mask = indices[0]
            for _ in range(len(mask.get_size()), len(self.get_size())):
                mask = unsqueeze(mask, -1)
            return index_put_as_masked_fill(self, [mask], values, accumulate)

        # Fallback in torch deterministic mode
        if torch.are_deterministic_algorithms_enabled():
            return index_put_fallback(self, indices, values, accumulate)

        # Fallback if there is a boolean index
        for index in indices:
            if index is not None and index.get_dtype() in (torch.bool, torch.uint8):
                return index_put_fallback(self, indices, values, accumulate)

        x_size = self.get_size()
        x_ndim = len(x_size)

        if accumulate and needs_fallback_due_to_atomic_add_limitations(self.get_dtype()):
            # self is an scalar Tensor
            if x_ndim == 0:
                self = view(self, [1])
            self = index_put_fallback(self, indices, values, accumulate)
            if x_ndim == 0:
                self = view(self, [])
            return self

        values = to_dtype(values, self.get_dtype())

        try:
            # Note that code will only get here when dtype is uint32
            indices, tensor_indices = check_and_broadcast_indices(
                indices, self.get_device()
            )
        except NotImplementedError:
            return index_put_fallback(self, indices, values, accumulate)

        indices_loaders = [i.make_loader() if i is not None else None for i in indices]

        assert isinstance(self, TensorBox)
        self.realize()

        # self is an scalar Tensor
        if x_ndim == 0:
            self = view(self, [1])

        # We can use the first one since they are all required to be the same size
        tensor_size = list(indices[tensor_indices[0]].get_size())
        indexed_size = [x_size[i] for i in range(len(indices))]

        expected_vals_size, inner_fn = index_output_size_and_inner_fn(
            x_size,
            indices,
            tensor_indices,
            tensor_size,
            indices_loaders,
            indexed_size,
            None,
            check=check,
        )

        values = expand(values, expected_vals_size)
        # all guards are set above during broadcast_tensors and expand

        def should_use_template():
            if accumulate:
                return False

            # indices have same dim with self, last dim is indice
            if len(indices_loaders) == x_ndim and indices_loaders[-1] is not None:
                return False
            # self dims is 1 or self 1 in size or indice is 1
            if x_ndim == 1 or 1 in x_size or tensor_size[0] == 1:
                return False
            valid_indices = [indice for indice in indices if indice]
            if len(valid_indices) != 1:
                return False
            if isinstance(self, TensorBox) and isinstance(self.data, ir.BaseView):
                return False
            if isinstance(valid_indices[0], TensorBox) and isinstance(valid_indices[0].data, ir.BaseView):
                return False
            if isinstance(values, TensorBox) and isinstance(values.data, ir.BaseView):
                return False

            return True

        if should_use_template():
            valid_index = next(i for i, indice in enumerate(indices_loaders) if indice)
            boundary = int(x_size[valid_index])
            scatter = IndexputTemplate(
                device=self.get_device(),
                dtype=self.get_dtype(),
                inner_fn=values.make_loader(), # load values
                ranges=expected_vals_size,  # iter_ranges,
                output_indexer=inner_fn, # store values
                scatter_mode=None,
                boundary=boundary
            )
        else:
            scatter = ir.Scatter(
                device=self.get_device(),
                dtype=self.get_dtype(),
                inner_fn=values.make_loader(), # load values
                ranges=expected_vals_size,  # iter_ranges,
                output_indexer=inner_fn, # store values
                scatter_mode="atomic_add" if accumulate else None,
            )
        buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.MutationLayoutSHOULDREMOVE(self),
            data=scatter,
        )
        buffer.name = V.graph.register_buffer(buffer)
        V.graph.register_operation(buffer)

        if x_ndim == 0:
            self = view(self, [])
        return self
    
    # All the indexing decompositions are written in terms of index, index_put, and index_put_
    # We cannot have this lowering as a decomposition as it introduces
    # mutation in the graph, which is bad for Aot Autograd. Aot Autograd runs dead
    # code elimination and common subexpression elimination optimizations, which
    # assume graphs to be side-effect free. More details at
    @register_lowering(aten.index_put)
    def index_put(x, indices, values, accumulate=False):
        return index_put_impl_(
            clone(x), indices, values, accumulate, check=True, may_realize=False
        )


    @register_lowering(aten._unsafe_index_put)
    def _unsafe_index_put(x, indices, values, accumulate=False):
        return index_put_impl_(
            clone(x), indices, values, accumulate, check=False, may_realize=False
        )

    @register_lowering(aten.index_put_, type_promotion_kind=None)
    def index_put_(self, indices, values, accumulate=False):
        return index_put_impl_(
            self, indices, values, accumulate, check=True, may_realize=True
        )


    @register_lowering(aten.scatter_reduce_, type_promotion_kind=None)
    def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool = True):
        assert reduce in (None, "sum", "prod", "mean", "amax", "amin")
        assert (
            len(aten.scatter_reduce_.overloads()) == 1
            and "two" in aten.scatter_reduce_.overloads()
        ), "aten.scatter_reduce_.two is not the unique overload of aten.scatter_reduce_"

        if isinstance(src, Number):
            src = full_like(self, src)

        fallback_result = scatter_fallback(
            aten.scatter_reduce_.two,
            self,
            dim,
            index,
            src,
            reduce=reduce,
            include_self=include_self,
        )

        if fallback_result:
            return fallback_result

        assert isinstance(self, TensorBox)
        assert "int" in str(index.get_dtype())

        ndim = len(self.get_size())
        if ndim == 0:
            self = view(self, [1])

        if isinstance(src, TensorBox) and len(src.get_size()) == 0:
            src = view(src, [1])

        if isinstance(index, TensorBox) and len(index.get_size()) == 0:
            index = view(index, [1])

        if index.get_numel() == 0:
            return self

        dim = _validate_dim(self, dim)

        self.realize()
        index_loader = index.make_loader()
        src_loader = src.make_loader() if isinstance(src, TensorBox) else None

        def output_indexer(idx):
            # self is captured from the end of the function, so it may have 0 dim
            shape = self.get_size()
            ndim = len(shape)
            indirect_idx = list(idx)
            indirect_idx[dim] = ops.indirect_indexing(
                index_loader(idx), 1 if ndim == 0 else shape[dim], wrap_neg=False
            )
            return indirect_idx

        def template_output_indexer(idx):
            # self is captured from the end of the function, so it may have 0 dim
            shape = self.get_size()
            ndim = len(shape)
            indirect_idx = list(idx)
            indirect_idx[dim] = ops.indirect_indexing(
                index_loader(idx), 1 if ndim == 0 else shape[dim], wrap_neg=False
            )
            return indirect_idx, shape[dim]

        def fn(idx):
            if src_loader:
                return src_loader(idx)
            else:
                # src is a scalar
                return ops.constant(src, self.get_dtype())

        def backend_reduce_str(reduce):
            if reduce == "sum":
                return "atomic_add"
            else:
                assert reduce is None
                return None

        if not include_self:
            # zero out the corresponding elements first
            zero_out = ir.Scatter(
                device=self.get_device(),
                dtype=self.get_dtype(),
                inner_fn=lambda index: ops.constant(0, self.get_dtype()),
                ranges=index.get_size(),
                output_indexer=output_indexer,
                scatter_mode=None,
            )
            buffer = ir.ComputedBuffer(
                name=None,
                layout=ir.MutationLayoutSHOULDREMOVE(self),
                data=zero_out,
            )
            buffer.name = V.graph.register_buffer(buffer)
            V.graph.register_operation(buffer)

        def should_use_template():
            if reduce:
                return False
            if 1 in index.get_size() or 1 in self.get_size() or 1 in src.get_size():
                return False
            if isinstance(index, TensorBox) and isinstance(index.data, ir.BaseView):
                return False
            if isinstance(self, TensorBox) and isinstance(self.data, ir.BaseView):
                return False
            if isinstance(src, TensorBox) and isinstance(src.data, ir.BaseView):
                return False

            return True

        if should_use_template():
            scatter = ScatterTemplate(
                device=self.get_device(),
                dtype=self.get_dtype(),
                inner_fn=fn,
                ranges=index.get_size(),
                output_indexer=template_output_indexer,
                scatter_mode=backend_reduce_str(reduce),
            )
        else:
            scatter = ir.Scatter(
                device=self.get_device(),
                dtype=self.get_dtype(),
                inner_fn=fn,
                ranges=index.get_size(),
                output_indexer=output_indexer,
                scatter_mode=backend_reduce_str(reduce),
            )
        buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.MutationLayoutSHOULDREMOVE(self),
            data=scatter,
        )
        buffer.name = V.graph.register_buffer(buffer)
        V.graph.register_operation(buffer)

        if ndim == 0:
            self = view(self, [])
        return self

    @register_lowering(aten.scatter_reduce, type_promotion_kind=None)
    def scatter_reduce(x, dim: int, index, src, reduction_type, **kwargs):
        return scatter_reduce_(clone(x), dim, index, src, reduction_type, **kwargs)

    @register_lowering(aten.scatter_, type_promotion_kind=None)
    def scatter_(self, dim: int, index, src, *, reduce=None):
        assert reduce in (None, "add", "multiply")
        if reduce is None:
            op_overload = getattr(aten.scatter_, V.graph.current_node.target._overloadname)  # type: ignore[union-attr]
            fallback_result = scatter_fallback(
                op_overload, self, dim, index, src, reduce=reduce
            )
            if fallback_result is not None:
                return fallback_result

        if reduce == "add":
            reduce = "sum"
        elif reduce == "multiply":
            reduce = "prod"
        return scatter_reduce_(self, dim, index, src, reduce)

    @register_lowering(aten.scatter, type_promotion_kind=None)
    def scatter(x, dim: int, index, src, **kwargs):
        return scatter_(clone(x), dim, index, src, **kwargs)

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

    @register_lowering(aten.index, type_promotion_kind=None)
    def index(x, indices):
        # check whether is high dim index_select
        def should_use_template():
            x_size = x.get_size()
            valid_indices = [indice for indice in indices if indice]
            # x only one dim | 1 in data size
            if len(x_size) == 1 or 1 in x_size:
                return False

            if len(valid_indices) != 1:
                return False
            select_dim = indices.index(valid_indices[0])
            if select_dim == len(x_size) - 1:
                return False
            if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
                return False
            if isinstance(valid_indices[0], TensorBox) and isinstance(valid_indices[0].data, ir.BaseView):
                return False
            return True

        if should_use_template():
            valid_indices = [indice for indice in indices if indice]
            select_dim = indices.index(valid_indices[0])
            return lowering_index_select(x, select_dim, valid_indices[0], 'index_select')

        return lowering.index(x, indices)

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

    @register_lowering(aten.native_layer_norm)
    def native_layer_norm(
        x,
        normalized_shape,
        weight=None,
        bias=None,
        eps=1e-5
    ):
        # Performance consideration: fallback for bfloat16 and float16
        if get_soc_version() >= 250 and \
            (x.dtype == torch.bfloat16 or x.dtype == torch.float16):
            return fallback_handler(aten.native_layer_norm.default)(x, normalized_shape, weight, bias, eps)
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
            return_mean=True
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
    
    @register_lowering(triton_kernel_wrapper_mutation)
    def triton_kernel_wrap_(
        *,
        kernel_idx,
        constant_args_idx,
        grid,
        tma_descriptor_metadata,
        kwargs,
    ):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        constant_args = kernel_side_table.get_constant_args(constant_args_idx)
        ir.UserDefinedTritonKernel(
            kernel_idx=kernel_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kernel_args={**kwargs, **constant_args},
        )
        return {key: val for key, val in kwargs.items() if isinstance(val, TensorBox)}
