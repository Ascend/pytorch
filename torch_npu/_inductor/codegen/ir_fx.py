import traceback
import typing
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Union
)
from typing import Optional
from unittest.mock import patch

import sympy
import torch
from sympy import Expr
from torch._inductor import config
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor.ir import (NopKernel, SliceView, IRNode, StorageBox, FlexibleLayout, FixedLayout, NonOwningLayout,
                                Pointwise, TensorBox, ComputedBuffer, View, log, Layout)
from torch._inductor.virtualized import ops, OpsValue, V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Identity

import torch_npu
from torch_npu._inductor import ir as npu_ir
from torch_npu._inductor.codegen.triton_utils import get_byte_per_numel
from torch_npu._inductor import config as npu_config
from ..lowering import (
    fetch_graphs,
    merge_traced_graphs,
    node_id,
    clone,
    create_fake_input,
    subtract_graph
)


def _patch_loops_get_name(self):
    return self.node_name


def _patch_loops_get_traced_graph(self):
    return self.traced_graph


@classmethod
def _patch_loops_create(cls, *args, **kwargs):
    origin_node = kwargs.pop("origin_node", None)
    traced_graph = kwargs.pop("traced_graph", None)
    node_name = kwargs.pop("node_name", None)
    tb = kwargs.pop("traceback", None)
    r = cls(*args, **kwargs)
    # Need to explicitly set origin_node here to propagate it down.
    # todo(chilli): I think it would be better for IRNode to directly set
    # origin_node
    r._post_init_setattr("origin_node", origin_node)
    r._post_init_setattr("traceback", tb or r.traceback)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return ir.TensorBox.create(r)


def _patch_pointwise_constant_to_device(self, device, traced_graph=None, node_name=None):
    """Move this to a given device. Requires that all reads are to constants."""
    loader = self.make_loader()
    loader = patch.object(ir.ConstantBuffer, "override_device", device)(loader)

    r = ir.Pointwise(device=device, dtype=self.dtype, inner_fn=loader, ranges=self.ranges)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


@classmethod
def _patch_reduction_create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: ir.Sequence[Expr],
        reduction_ranges: ir.Sequence[Expr],
        reduction_type: str,
        reduction_hint: ir.ReductionHint = ir.ReductionHint.DEFAULT,
        input_node: Optional[ir.IRNode] = None,
        traced_graph=None,
        node_name: str = None
) -> ir.TensorBox:
    reduction_numel = V.graph.sizevars.simplify(ir.sympy_product(reduction_ranges))

    if reduction_numel == 0:
        # N.B. This is a hack to generate the literal of the given type
        # Ideally, we should be fixing `def constant` in triton.py
        # but it breaks due to hardcoded dtypes in other places
        def py_cnst(val: object) -> Union[bool, float, int]:
            if dst_dtype == torch.bool:
                return bool(val)
            elif dst_dtype.is_floating_point:
                if not isinstance(val, typing.SupportsFloat):
                    raise RuntimeError("assert val must support float conversion")
                return float(val)
            else:
                if not isinstance(val, typing.SupportsInt):
                    raise RuntimeError("assert val must support int conversion")
                return int(val)

        rtypes_to_inits = {
            "sum": py_cnst(0),
            "xor_sum": py_cnst(0),
            "prod": py_cnst(1),
            "any": py_cnst(0),
            # "all" is desugared to `!any(!val)`
        }

        if reduction_type not in rtypes_to_inits:
            raise RuntimeError(f"assert {reduction_type} not supported for zero-dimension tensors!")

        def const_fn(index: int) -> ir.OpsValue:
            return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

        return ir.Pointwise.create(
            device=device,
            dtype=src_dtype,
            inner_fn=const_fn,
            ranges=list(ranges),
            traced_graph=traced_graph,
            node_name=node_name
        )

    if reduction_numel == 1:
        # this reduction is actually a pointwise op
        if reduction_type in ("argmin", "argmax"):

            def fn(index: int) -> ir.OpsValue:
                return ops.constant(0, dst_dtype)

        else:

            def fn(index: int) -> ir.OpsValue:
                reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                return inner_fn(index, reduction_index)

        return ir.Pointwise.create(
            device=device, dtype=dst_dtype, inner_fn=fn, ranges=ranges,
            traced_graph=traced_graph,
            node_name=node_name
        )

    if (
            isinstance(reduction_numel, ir.Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < config.unroll_reductions_threshold
            and (ir.sympy_product(ranges) != 1 or ir.is_gpu(device.type))
    ):
        # NB: This works around pytorch issues 140457
        # since turning reductions into pointwise ops can exacerbate this problem
        return ir.Pointwise.create(
            device=device,
            dtype=dst_dtype,
            inner_fn=cls._unroll_reduction_fn(
                inner_fn, reduction_ranges, reduction_type, src_dtype
            ),
            ranges=ranges,
            traced_graph=traced_graph,
            node_name=node_name
        )

    # triton doesn't support reduce to single element well, so break it up
    hint, split = cls.num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node,
    )
    # intermediate reduction in split can contain complex indexing,
    # and num_splits will fail to correctly set the hint
    # reuse the passed hint if available
    if reduction_hint == ir.ReductionHint.DEFAULT:
        reduction_hint = hint
    if split == -1:
        if input_node is None:
            raise RuntimeError("assert input_node cannot be None")
        new_ranges, new_reduction_ranges = ir.extract_input_node_reduction_ranges(
            input_node
        )
        if new_ranges is None:
            raise RuntimeError("assert new_ranges cannot be None")
        if new_reduction_ranges is None:
            raise RuntimeError("assert new_reduction_ranges cannot be None")
        r = cls.create_multilayer_existing_ranges(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
        )
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r
    elif split > 1:
        # triton doesn't support reduce to single element well, so break it up
        r = cls.create_multilayer(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            split,
            reduction_hint,
        )
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    r = ir.Reduction(
        device=device,
        dtype=dst_dtype,
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
        reduction_type=reduction_type,
        src_dtype=src_dtype,
        reduction_hint=reduction_hint,
    )
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)

    return ir.TensorBox.create(r)


def _patch_baseview_get_traced_graph(self):
    if hasattr(self, 'traced_graph') and self.traced_graph is not None:
        return self.traced_graph
    return self.data.get_traced_graph()


def _patch_base_view_get_reads(self):
    with patch.object(ir.FlexibleLayout, "allow_indexing", True):
        r = ir.extract_read_writes(
            self.make_loader(),
            self.get_size(),
        ).reads
    for md in r:
        if md.index.has(ir.ModularIndexing):
            if md.index.has(ir.FloorDiv):
                self.realize()
                return r
            else:
                for m in md.index.find(ir.ModularIndexing):
                    for arg in m.args:
                        if arg.has(ir.ModularIndexing):
                            self.realize()
                            return r
    return r


def has_buffer(inp):
    if not hasattr(inp, 'data'):
        return False
    if isinstance(inp.data, ir.Buffer):
        return True
    return has_buffer(inp.data)


def try_get_buffer(inp):
    if not hasattr(inp, 'data'):
        return False
    if isinstance(inp.data, ir.Buffer):
        return inp.data
    return try_get_buffer(inp.data)


def _patch_baseview_realize(self):
    if hasattr(self, 'traced_graph') and self.traced_graph is not None:
        r = self.data.realize()
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel, npu_ir.ConcatKernel, ir.ExternKernelOut, ir.MultiTemplateBuffer)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(self.traced_graph, traced_graph, node_name=buf_name)
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta['val'] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.realize()


def _patch_baseview_realize_hint(self):
    if hasattr(self, 'traced_graph') and self.traced_graph is not None:
        r = self.data.realize_hint()
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel, npu_ir.ConcatKernel, ir.ExternKernelOut)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(self.traced_graph, traced_graph, node_name=buf_name)
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta['val'] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.realize_hint()


def _patch_mark_reuse(self, users):
    if hasattr(self, 'traced_graph') and self.traced_graph is not None:
        r = self.data.mark_reuse(users)
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel, npu_ir.ConcatKernel, ir.ExternKernelOut)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(self.traced_graph, traced_graph, node_name=buf_name)
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta['val'] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.mark_reuse(users)


@classmethod
def _patch_expandview_create(cls, x, new_size, traced_graph=None, node_name=None):
    new_size = cls._normalize_size(x, new_size)

    if ir.is_storage_and_layout(x):
        storage, old_layout = ir.as_storage_and_layout(x)
        skip = len(new_size) - len(old_layout.size)
        if skip < 0:
            raise RuntimeError(f"assert Internal error: skip must be non-negative, got {skip}")
        new_stride = [sympy.Integer(0)] * skip
        for stride, size in zip(old_layout.stride, old_layout.size):
            new_stride.append(
                stride
                if not V.graph.sizevars.shape_env.evaluate_expr(
                    sympy.Eq(size, 1), size_oblivious=True
                )
                else sympy.Integer(0)
            )
        new_layout = ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            list(new_size),
            new_stride,
            old_layout.offset,
        )

        r = ir.ReinterpretView(data=storage, layout=new_layout)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    r = ir.ExpandView(data=x, size=new_size)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)

    return r


@classmethod
def _patch_permuteview_create(cls, x, dims, traced_graph=None, node_name=None):
    dims = cls._map_neg_dims(dims)
    if OrderedSet(dims) != OrderedSet(range(len(dims))):
        raise RuntimeError("assert OrderedSet(dims) != OrderedSet(range(len(dims)))")
    if ir.is_storage_and_layout(x):
        storage, old_layout = ir.as_storage_and_layout(x)
        new_layout = ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            [old_layout.size[i] for i in dims],
            [old_layout.stride[i] for i in dims],
            old_layout.offset,
        )
        r = ir.ReinterpretView(data=storage, layout=new_layout)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    r = ir.PermuteView(data=x, dims=dims)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


@classmethod
def _patch_view_create(cls, x, new_size, traced_graph=None, node_name=None):
    if not isinstance(new_size, (tuple, list)):
        raise RuntimeError("assert new_size must be tuple, list, or tuple")
    old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)
    # Skip pointless views
    if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
        return x

    unbacked_symbols_in_sizes = False
    if (
            len(ir.free_unbacked_symbols(old_size)) > 0
            or len(ir.free_unbacked_symbols(new_size)) > 0
    ):
        unbacked_symbols_in_sizes = True

    if 0 in new_size:

        def fake_reindex(index):
            return tuple([0] * len(old_size))

        r = cls(data=x, size=list(new_size), reindex=fake_reindex)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    #  next: a new class for FixedTransferLayout that output layout is constrained by input layout
    elif (ir.is_contiguous_storage_and_layout(
            x) or unbacked_symbols_in_sizes):  # and not isinstance(x.data, ir.ReinterpretView):
        if unbacked_symbols_in_sizes and (not ir.is_contiguous_storage_and_layout(x)):
            # realize x; otherwise, the dynamic_reshape_indexer below will fail
            # due to the size_hint's inability to process unbacked SymInts
            x = ir.ExternKernel.realize_input(x)

        storage, old_layout = ir.as_storage_and_layout(x, want_contiguous=True)
        new_layout = ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            new_size,
            ir.FlexibleLayout.contiguous_strides(new_size),
            old_layout.offset,
        )

        r = ir.ReinterpretView(data=storage, layout=new_layout)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    reindex = cls.dynamic_reshape_indexer(old_size, new_size)

    r = cls(data=x, size=list(new_size), reindex=reindex)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


@classmethod
def _patch_sliceview_create(cls, x, dim, start, end, step=1, clamp=True, traced_graph=None,
                            node_name=None):  # next: crm, clamp=True
    step = sympy.expand(step)
    if not (isinstance(step, sympy.Expr) or step > 0):
        raise RuntimeError("assert step must be a sympy.Expr or a positive number")
    try:
        if start == 0 and end >= 2 ** 63 - 1 and step == 1:
            return x
    except TypeError:
        pass
    sizevars = V.graph.sizevars
    new_size = list(x.get_size())

    if clamp:
        start, end = cls.normalize_start_end(x, dim, start, end)

    new_size[dim] = ir.FloorDiv(end - start + (step - 1), step)

    if ir.is_storage_and_layout(x):
        # Fast path
        storage, old_layout = ir.as_storage_and_layout(x)
        new_stride = list(old_layout.stride)
        new_stride[dim] = new_stride[dim] * step
        new_layout = ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            new_size,
            new_stride,
            old_layout.offset + old_layout.stride[dim] * start,
        )
        r = ir.ReinterpretView(data=storage, layout=new_layout)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r

    def reindex(index):
        if len(index) != len(new_size):
            raise RuntimeError(f"assert wrong ndim {index} {new_size}")
        index = list(index)
        index[dim] = index[dim] * step + start
        return index

    # redirect to a generic view
    r = ir.SliceView(data=x, size=new_size, reindex=reindex)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


def _patch_buffer_get_traced_graph(self):
    return self.traced_graph


@classmethod
def _patch_concatkernel_create(cls, inputs, dim):
    device = inputs[0].get_device()
    dtype = inputs[0].get_dtype()
    new_size = list(inputs[0].get_size())
    offsets_start = [0]
    offsets_end = [new_size[dim]]
    if not (0 <= dim < len(new_size)):
        raise RuntimeError(f"assert dim ({dim}) must be between 0 and {len(new_size) - 1}")
    for i in range(1, len(inputs)):
        input_size = inputs[i].get_size()
        offsets_start.append(new_size[dim])
        if len(input_size) != len(new_size):
            raise RuntimeError(
                f"assert input_size and new_size is not same. Got {len(input_size)} vs {len(new_size)}")
        if inputs[i].get_dtype() != dtype:
            raise RuntimeError(f"assert Expected dtype {dtype}, but got {inputs[i].get_dtype()}")
        if inputs[i].get_device() != device:
            raise RuntimeError(f"assert Expected device {device}, but got {inputs[i].get_device()}")

        for j in range(len(new_size)):
            if j == dim:
                new_size[j] = new_size[j] + input_size[j]
            else:
                new_size[j] = V.graph.sizevars.guard_equals(
                    new_size[j], input_size[j]
                )
        offsets_end.append(new_size[dim])

    output_stride = ir.FlexibleLayout.contiguous_strides(new_size)
    # If any of the inputs is in CL format, use CL format for the output
    for i in range(len(inputs)):
        x = inputs[i]
        if ir.is_storage_and_layout(x):
            layout = x.get_layout()
            if (
                    isinstance(layout, ir.FixedLayout)
                    and layout.is_channels_last_contiguous(layout.size, layout.stride)
            ):
                # use CL stride for the output
                output_stride = ir.make_channels_last_strides_for(new_size)
                break

    any_input_is_storage_and_layout = any(ir.is_storage_and_layout(x) for x in inputs)
    fx_node_args = V.graph.current_node.args[0]
    if not isinstance(fx_node_args, list):
        raise RuntimeError("assert fx_node_args must be a list")
    # If any of the inputs has meta tensor and the meta tensor is in CL format, use CL format for the output
    if any_input_is_storage_and_layout is False and any(
            "val" in arg.meta
            and (
                    arg.meta["val"].is_contiguous(memory_format=torch.channels_last)
                    or arg.meta["val"].is_contiguous(memory_format=torch.channels_last_3d)
            )
            for arg in fx_node_args
    ):
        output_stride = ir.make_channels_last_strides_for(new_size)

    concat_kernel = ir.ConcatKernel(
        name=None,
        layout=ir.FixedLayout(
            device=device,
            dtype=dtype,
            size=new_size,
            stride=output_stride,
        ),
        inputs=[],
    )

    kernel = ir.StorageBox(concat_kernel)
    op_names = []
    for i in range(len(inputs)):
        input_buffer = cls.realize_into(
            inputs[i],
            ir.SliceView.create(
                kernel, dim, offsets_start[i], offsets_end[i], clamp=False
            ),
        )
        concat_kernel.inputs.append(input_buffer)

        if isinstance(inputs[i].data, ir.BaseView):
            input_unwrapped = inputs[i].data.unwrap_view()
        else:
            input_unwrapped = inputs[i].data

        if (
                input_unwrapped.is_input_buffer()
                and ir.is_gpu(inputs[i].get_device().type)
                and not ir.is_dynamic(input_buffer)
        ):
            op_names.append(input_buffer.get_operation_name())

    if len(op_names) > 1 and V.graph.has_feature(device, ir.BackendFeature.FOREACH):
        V.graph.register_operation_list(op_names)

    cat_inputs = [ir.TensorBox(ir.StorageBox(inp)) for inp in concat_kernel.inputs]
    input_graphs = fetch_graphs([cat_inputs])
    node_name = f'cat_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.cat, node_name, dim=dim)

    concat_kernel._post_init_setattr("name", V.graph.register_buffer(concat_kernel))
    concat_kernel._post_init_setattr("inputs", cls.unwrap_storage(concat_kernel.inputs))
    concat_kernel._post_init_setattr("traced_graph", new_graph)
    concat_kernel._post_init_setattr("node_name", node_name)

    return kernel


def _patch_concatkernel_get_traced_graph(self):
    return self.traced_graph


@classmethod
def _patch_concatkernel_realize_into(cls, src, dst):
    # Attempt to turn this into a ReinterpretView rather than assert.
    # This has concessions around layout, as as_storage_and_layout
    # can cause us to go from flexible to fixed layout.
    if not isinstance(dst, ir.ReinterpretView):
        if ir.is_storage_and_layout(dst):
            storage, layout = ir.as_storage_and_layout(dst)
            dst = ir.ReinterpretView(data=storage, layout=layout)
    if not isinstance(dst, ir.ReinterpretView):
        raise RuntimeError(f"assert Expected dst to be an instance of ir.ReinterpretView. Got: {dst}")
    if isinstance(src, ir.TensorBox):
        # unwrap a TensorBox
        return cls.realize_into(src.data, dst)
    if isinstance(src, ir.StorageBox):
        src.realize()
        # ExternKernelAlloc has specific requirements for output layout, should create a copy
        if not hasattr(src.data, "layout"):
            raise RuntimeError("assert src.data has no attribute 'layout'")
        if cls.can_realize_into_without_copy(src):
            src.data.layout = ir.NonOwningLayout(dst)
            return src.data
    pw = clone(src, memory_format=torch.contiguous_format)
    return cls.realize_into(pw, dst)


@classmethod
def _patch__npu_concatkernel_create(cls, inputs, dim, is_reindex):
    new_size = list(inputs[0].get_size())
    offsets_start = [0]
    offsets_end = [new_size[dim]]

    for i in range(1, len(inputs)):
        input_size = inputs[i].get_size()
        offsets_start.append(new_size[dim])
        new_size[dim] = new_size[dim] + input_size[dim]
        offsets_end.append(new_size[dim])

    output_stride: Sequence[int] = FlexibleLayout.contiguous_strides(new_size)

    concat_kernel = npu_ir.ConcatKernel(
        name=None,
        layout=FixedLayout(
            device=inputs[0].get_device(),
            dtype=inputs[0].get_dtype(),
            size=new_size,
            stride=output_stride,
        ),
        inputs=[],
    )
    kernel = StorageBox(concat_kernel)

    if is_reindex:
        for i, inp in enumerate(inputs):
            input_buffer = cls.single_realize_into(inp, SliceView.create(
                kernel, dim, offsets_start[i], offsets_end[i], clamp=False))
            concat_kernel.inputs.append(input_buffer)
    else:
        max_numel_in_per_kernel = npu_config.max_cat_size_in_per_kernel // get_byte_per_numel(inputs[0].get_dtype())
        input_sub = []
        prev = 0
        for i, inp in enumerate(inputs):
            input_sub.append(inp)
            if i == len(inputs) - 1 or offsets_end[i + 1] - offsets_start[prev] > max_numel_in_per_kernel:
                input_buffer = cls.realize_into(input_sub, SliceView.create(
                    kernel, dim, offsets_start[prev], offsets_end[i], clamp=False
                ), dim)
                concat_kernel.inputs.append(input_buffer)
                input_sub = []
                prev = i + 1

    concat_kernel.name = V.graph.register_buffer(concat_kernel)
    concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
    V.graph.register_operation(concat_kernel)

    cat_inputs = [ir.TensorBox(ir.StorageBox(inp)) for inp in concat_kernel.inputs]
    input_graphs = fetch_graphs([cat_inputs])
    node_name = f'cat_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.cat, node_name, dim=dim)

    concat_kernel._post_init_setattr("traced_graph", new_graph)
    concat_kernel._post_init_setattr("node_name", node_name)

    return kernel


@classmethod
def _patch_npu_concatkernel_realize_into(cls, inputs: Sequence[IRNode], dst: IRNode, dim) -> IRNode:
    if len(inputs) == 1:
        return cls.single_realize_into(inputs[0], dst)

    inputs_ranges = [0]
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end + inp.get_size()[dim]))
        prev_end = inputs_ranges[-1]

    output_size = list(inputs[0].get_size())
    output_size[dim] = inputs_ranges[-1]

    def inner_fn_insert_slice(idx):
        idx_load = list(idx)
        output = ops.index_expr(output_size[dim], torch.float32)
        for i, inp in enumerate(inputs):
            output = ops.cat_insert_slice(output, inp.make_loader()(idx_load), int(inputs_ranges[i]),
                                          int(inp.get_size()[dim]), int(output_size[dim]))
        return output

    def inner_fn_store(idx):
        idx_load = list(idx)
        output = ops.index_expr(output_size[dim], torch.float32)
        for i, inp in enumerate(inputs):
            idx_output = list(idx)
            idx_output[dim] = Identity(idx_output[dim] + inputs_ranges[i])
            output = ops.cat_store(dst.get_name(), inp.make_loader()(idx_load), int(inp.get_size()[dim]),
                                   dst.make_indexer()(idx_output), dst.make_indexer()(idx_load))
        return output

    input_graphs = fetch_graphs([inputs])
    node_name = f'cat_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.cat, node_name, dim=dim)

    input_strides = [inp.get_stride()[dim - 1] == output_size[dim] for inp in inputs if inp.maybe_get_stride() is not None]
    is_split_inputs = input_strides and all(input_strides)
    if npu_config.use_store_in_cat or is_split_inputs:
        pw = npu_ir.ConcatOutputKernel.create(
            device=inputs[0].get_device(),
            dtype=inputs[0].get_dtype(),
            inner_fn=inner_fn_store,
            ranges=output_size,
            traced_graph=new_graph,
            node_name=node_name
        )
    else:
        pw = Pointwise.create(
            device=inputs[0].get_device(),
            dtype=inputs[0].get_dtype(),
            inner_fn=inner_fn_insert_slice,
            ranges=output_size,
            traced_graph=new_graph,
            node_name=node_name
        )

    pw.realize()
    pw.data.data.layout = NonOwningLayout(dst)
    return pw.data.data


@classmethod
def _patch_npu_concatkernel_single_realize_into(cls, src, dst):
    pw = clone(src, memory_format=torch.contiguous_format)
    pw.realize()
    pw.data.data.layout = NonOwningLayout(dst)
    return pw.data.data


def _patch_externkernel_copy_input(x):
    traced_graph = x.get_traced_graph()
    node_name = x.get_name()
    if traced_graph is None:
        traced_graph = fetch_graphs([x])[0]
        node_name = f'getitem_{next(node_id)}'

    pw = ir.Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
        traced_graph=traced_graph,
        node_name=node_name
    )
    pw.realize()
    return pw


@classmethod
def _patch_externkernel_convert_to_reinterpret_view(cls, x):
    """
    In order to pass this to an extern kernel we need a
    ReinterpretView not a View.  This allows us to avoid some
    unneeded copies.
    """
    if not isinstance(x, ir.BaseView):
        raise RuntimeError(f"assert Expected type {ir.BaseView}, got {type(x)}")
    if isinstance(x, ir.ReinterpretView):
        return x

    # NOTE: Don't use extract_read_writes here as it fails when
    # make_loader() inlines the computation
    x_unwrap_view = x.unwrap_view()
    buf = V.graph.get_buffer(x_unwrap_view.get_name())
    if buf is None:
        raise RuntimeError("assert buf cannot be None")
    x_unwrap_view_fx_node = buf.get_origin_node()
    # Prefer channels last format according to how the format is set from eager.
    if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view.layout, ir.FlexibleLayout)
            and (
            x_unwrap_view_fx_node.meta["val"].is_contiguous(
                memory_format=torch.channels_last
            )
            or x_unwrap_view_fx_node.meta["val"].is_contiguous(
        memory_format=torch.channels_last_3d
    )
    )
    ):
        x_unwrap_view.freeze_layout_with_same_order(
            ir.make_channels_last_strides_for(x_unwrap_view.get_size())
        )
    else:
        x_unwrap_view.freeze_layout()

    index_args, var_ranges = ir.dependencies.index_vars_squeeze(
        x.get_size(), prefix="r"
    )
    range_vars = index_args[0]
    index = x.make_indexer()(range_vars)

    index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
    strides = V.graph.sizevars.stride_vars(index, range_vars)
    offset = V.graph.sizevars.offset_var(index, range_vars)
    expected = ir.sympy_dot(range_vars, strides) + offset

    if index != expected:
        ir.log.debug(
            "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
            strides,
            offset,
            index,
        )
        raise NotImplementedError

    r = ir.ReinterpretView(
        data=x.data,
        layout=ir.FixedLayout(
            device=x.get_device(),
            dtype=x.get_dtype(),
            size=x.get_size(),
            stride=strides,
            offset=offset,
        ),
    )
    r._post_init_setattr("traced_graph", x.get_traced_graph())
    r._post_init_setattr("node_name", x.get_name())
    return r


@classmethod
def _patch_devicecopy_create(cls, x, device, non_blocking, traced_graph=None, node_name=None):
    if (
            not x.is_extern()
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.aot_inductor.use_runtime_constant_folding
    ):
        return x.constant_to_device(device)

    V.graph.add_device_info(device)
    V.graph.add_device_info(x.get_device())

    ir.developer_warning("DeviceCopy in input program")
    constant_args = (non_blocking,)
    r = ir.DeviceCopy(
        ir.FlexibleLayout(
            device=device,
            dtype=x.get_dtype(),
            size=x.get_size(),
        ),
        [cls.realize_input(x)],
        constant_args,
    )
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


def _patch_devicecopy_get_traced_graph(self):
    return self.traced_graph


def _patch_multioutput_get_traced_graph(self):
    return None


ir.MultiOutput.get_traced_graph = _patch_multioutput_get_traced_graph


def _patch_mutablebox_get_name(self):
    return self.data.get_name()


def _patch_mutablebox_get_traced_graph(self):
    return self.data.get_traced_graph()


@classmethod
def _patch_mutationlayout_realize_into(cls, src, dst, unsafe_alias=False):
    dst.realize()
    # NOTE: We must realize users of `dst` before we realize `src`, since
    # realization order determines scheduling order. Otherwise, src's
    # mutation would be scheduled before the existing users of dst!
    V.graph.mark_buffer_mutated(dst.get_name())

    if isinstance(src, ir.TensorBox):
        src = src.data

    # We copy the contents of src into dst. In most cases this should
    # be fused into a single kernel by the scheduler.
    # NOTE: We cannot change src's layout to mutate dst directly as this
    # would alias src to dst, which is not correct as further s to
    # dst would effect users of src. However if there are no more users of
    # dst, we can alias src to dst.
    src.realize_hint()

    if not unsafe_alias:
        input_graphs = fetch_graphs([dst, src])
        node_name = f'copy__{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.copy, node_name)

        src = ir.Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
            traced_graph=new_graph,
            node_name=node_name,
        ).data

    src.realize()
    if not isinstance(src.data.layout, ir.FlexibleLayout):
        raise RuntimeError("assert src.data.layout should be isinstance if ir.FlexibleLayout")
    src.data.layout = ir.MutationLayoutSHOULDREMOVE(dst)
    return src.data


def _patch_npu_inductor_ir():
    ir.Reduction.create = _patch_reduction_create
    ir.BaseView.get_traced_graph = _patch_baseview_get_traced_graph
    ir.BaseView.get_reads = _patch_base_view_get_reads
    ir.BaseView.realize = _patch_baseview_realize
    ir.BaseView.realize_hint = _patch_baseview_realize_hint
    ir.BaseView.mark_reuse = _patch_mark_reuse
    ir.ExpandView.create = _patch_expandview_create
    ir.PermuteView.create = _patch_permuteview_create
    ir.View.create = _patch_view_create
    ir.SliceView.create = _patch_sliceview_create
    ir.Buffer.traced_graph = None
    ir.Buffer.get_traced_graph = _patch_buffer_get_traced_graph
    ir.ConcatKernel.create = _patch_concatkernel_create
    ir.ConcatKernel.get_traced_graph = _patch_concatkernel_get_traced_graph
    ir.ConcatKernel.realize_into = _patch_concatkernel_realize_into
    npu_ir.ConcatKernel.create = _patch__npu_concatkernel_create
    npu_ir.ConcatKernel.get_traced_graph = _patch_concatkernel_get_traced_graph
    npu_ir.ConcatKernel.realize_into = _patch_npu_concatkernel_realize_into
    npu_ir.ConcatKernel.single_realize_into = _patch_npu_concatkernel_single_realize_into
    ir.ExternKernel.copy_input = _patch_externkernel_copy_input
    ir.ExternKernel.convert_to_reinterpret_view = _patch_externkernel_convert_to_reinterpret_view
    ir.DeviceCopy.create = _patch_devicecopy_create
    ir.DeviceCopy.get_traced_graph = _patch_devicecopy_get_traced_graph
    ir.MutableBox.get_name = _patch_mutablebox_get_name
    ir.MutableBox.get_traced_graph = _patch_mutablebox_get_traced_graph
    ir.Loops.get_name = _patch_loops_get_name
    ir.Loops.get_traced_graph = _patch_loops_get_traced_graph
    ir.Loops.create = _patch_loops_create
    ir.Pointwise.constant_to_device = _patch_pointwise_constant_to_device
    ir.MutationLayoutSHOULDREMOVE.realize_into = _patch_mutationlayout_realize_into
