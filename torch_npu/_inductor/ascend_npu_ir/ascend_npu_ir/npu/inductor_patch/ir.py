import traceback
from unittest.mock import patch

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import functools

import sympy
from sympy import Expr, Integer

import torch
from torch._inductor import ir
from torch._inductor import config

from torch._inductor.virtualized import ops, V
from torch._subclasses import FakeTensor
from torch.utils._ordered_set import OrderedSet

from .lowering import (
    fetch_graphs,
    merge_traced_graphs,
    node_id,
    clone,
    create_fake_input,
    subtract_graph,
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


ir.Loops.get_name = _patch_loops_get_name
ir.Loops.get_traced_graph = _patch_loops_get_traced_graph
ir.Loops.create = _patch_loops_create


def _patch_pointwise_constant_to_device(
    self, device, traced_graph=None, node_name=None
):
    """Move this to a given device. Requires that all reads are to constants."""
    loader = self.make_loader()
    loader = patch.object(ir.ConstantBuffer, "override_device", device)(loader)

    r = ir.Pointwise(device, self.dtype, loader, self.ranges)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


ir.Pointwise.constant_to_device = _patch_pointwise_constant_to_device


@classmethod
def _patch_reduction_create(  # type: ignore[override]
    cls,
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    inner_fn: Callable[..., Any],
    ranges: List[Expr],
    reduction_ranges: List[Expr],
    reduction_type: str,
    reduction_hint: ir.ReductionHint = ir.ReductionHint.DEFAULT,
    input_node: Optional[ir.IRNode] = None,
    traced_graph=None,
    node_name: str = None,
):
    reduction_numel = V.graph.sizevars.simplify(ir.sympy_product(reduction_ranges))

    if reduction_numel == 0:
        # N.B. This is a hack to generate the literal of the given type
        # Ideally, we should be fixing `def constant` in triton.py
        # but it breaks due to hardcoded dtypes in other places
        def py_cnst(val):
            return (
                bool(val)
                if dst_dtype == torch.bool
                else float(val) if dst_dtype.is_floating_point else int(val)
            )

        rtypes_to_inits = {
            "sum": py_cnst(0),
            "xor_sum": py_cnst(0),
            "prod": py_cnst(1),
            "any": py_cnst(0),
            # "all" is desugared to `!any(!val)`
        }

        assert (
            reduction_type in rtypes_to_inits.keys()
        ), f"{reduction_type} not supported for zero-dimension tensors!"

        def const_fn(index):
            return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

        return ir.Pointwise.create(
            device=device,
            dtype=src_dtype,
            inner_fn=const_fn,
            ranges=list(ranges),
            traced_graph=traced_graph,
            node_name=node_name,
        )

    if reduction_numel == 1:
        # this reduction is actually a pointwise op
        if reduction_type in ("argmin", "argmax"):

            def fn(index):
                return ops.constant(0, dst_dtype)

        else:

            def fn(index):
                reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                return inner_fn(index, reduction_index)

        return ir.Pointwise.create(
            device=device,
            dtype=dst_dtype,
            inner_fn=fn,
            ranges=ranges,
            traced_graph=traced_graph,
            node_name=node_name,
        )

    if (
        isinstance(reduction_numel, sympy.Integer)
        and V.graph.sizevars.size_hint(reduction_numel)
        < config.unroll_reductions_threshold
        and ir.sympy_product(ranges) != 1
    ):
        return ir.Pointwise.create(
            device=device,
            dtype=dst_dtype,
            inner_fn=cls._unroll_reduction_fn(
                inner_fn, reduction_ranges, reduction_type, src_dtype
            ),
            ranges=ranges,
            traced_graph=traced_graph,
            node_name=node_name,
        )

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


ir.Reduction.create = _patch_reduction_create


def _patch_baseview_get_traced_graph(self):
    if hasattr(self, "traced_graph") and self.traced_graph is not None:
        return self.traced_graph
    return self.data.get_traced_graph()


ir.BaseView.get_traced_graph = _patch_baseview_get_traced_graph


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


ir.BaseView.get_reads = _patch_base_view_get_reads


def try_get_buffer(inp):
    if not hasattr(inp, "data"):
        return False
    if isinstance(inp.data, ir.Buffer):
        return inp.data
    return try_get_buffer(inp.data)


def _patch_baseview_realize(self):
    if hasattr(self, "traced_graph") and self.traced_graph is not None:
        r = self.data.realize()
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(
            self.traced_graph, traced_graph, node_name=buf_name
        )
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta["val"] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.realize()


def _patch_baseview_realize_hint(self):
    if hasattr(self, "traced_graph") and self.traced_graph is not None:
        r = self.data.realize_hint()
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(
            self.traced_graph, traced_graph, node_name=buf_name
        )
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta["val"] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.realize_hint()


def _patch_mark_reuse(self, users):
    if hasattr(self, "traced_graph") and self.traced_graph is not None:
        r = self.data.mark_reuse(users)
        buffer = try_get_buffer(self)
        if not buffer:
            return r
        if isinstance(buffer, (ir.MultiOutput, ir.InputBuffer, ir.ConcatKernel)):
            return r
        traced_graph = buffer.data.get_traced_graph()
        buf_name = buffer.get_name()
        new_traced_graph, placeholder = subtract_graph(
            self.traced_graph, traced_graph, node_name=buf_name
        )
        if placeholder is not None:
            placeholder.name = buf_name
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            size = buffer.get_size()
            stride = buffer.get_stride()
            fake_input = create_fake_input(size, stride, device, dtype)
            placeholder.meta["val"] = fake_input
        self._post_init_setattr("traced_graph", new_traced_graph)
        return r
    else:
        return self.data.mark_reuse(users)


ir.BaseView.realize = _patch_baseview_realize
ir.BaseView.realize_hint = _patch_baseview_realize_hint
ir.BaseView.mark_reuse = _patch_mark_reuse


@classmethod
def _patch_expandview_create(cls, x, new_size, traced_graph=None, node_name=None):
    new_size = cls._normalize_size(x, new_size)

    if ir.is_storage_and_layout(x):
        storage, old_layout = ir.as_storage_and_layout(x)
        skip = len(new_size) - len(old_layout.size)
        assert skip >= 0
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


ir.ExpandView.create = _patch_expandview_create


@classmethod
def _patch_permuteview_create(cls, x, dims, traced_graph=None, node_name=None):
    dims = cls._map_neg_dims(dims)
    assert OrderedSet(dims) == OrderedSet(range(len(dims)))

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


ir.PermuteView.create = _patch_permuteview_create


@classmethod
def _patch_view_create(cls, x, new_size, traced_graph=None, node_name=None):
    assert isinstance(new_size, (tuple, list))
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

        def fake_reindex(index):  # type: ignore[no-untyped-def]
            return tuple([0] * len(old_size))

        r = cls(data=x, size=list(new_size), reindex=fake_reindex)
        r._post_init_setattr("traced_graph", traced_graph)
        r._post_init_setattr("node_name", node_name)
        return r
    # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
    elif ir.is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
        if unbacked_symbols_in_sizes and (not ir.is_contiguous_storage_and_layout(x)):
            # realize x; otherwise, the dynamic_reshape_indexer below will fail
            # due to the size_hint's inability to process unbacked SymInts
            # Need to require contiguous here instead of realize, see:
            x = ir.ExternKernel.require_exact_strides(
                x, ir.FlexibleLayout.contiguous_strides(x.get_size())
            )

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


ir.View.create = _patch_view_create


@classmethod
def _patch_sliceview_create(
    cls, x, dim, start, end, step=1, clamp=True, traced_graph=None, node_name=None
): 
    step = sympy.expand(step)
    assert isinstance(step, sympy.Expr) or step > 0
    try:
        if start == 0 and end >= 2**63 - 1 and step == 1:
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
        assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
        index = list(index)
        index[dim] = index[dim] * step + start
        return index

    # redirect to a generic view
    r = ir.SliceView(data=x, size=new_size, reindex=reindex)
    r._post_init_setattr("traced_graph", traced_graph)
    r._post_init_setattr("node_name", node_name)
    return r


ir.SliceView.create = _patch_sliceview_create


def _patch_buffer_get_traced_graph(self):
    return self.traced_graph


ir.Buffer.traced_graph = None
ir.Buffer.get_traced_graph = _patch_buffer_get_traced_graph


def _patch_concatkernel_get_traced_graph(self):
    return None


@classmethod
def _patch_concatkernel_realize_into(cls, src, dst):
    # Attempt to turn this into a ReinterpretView rather than assert.
    # This has concessions around layout, as as_storage_and_layout
    # can cause us to go from flexible to fixed layout.
    if not isinstance(dst, ir.ReinterpretView):
        if ir.is_storage_and_layout(dst):
            storage, layout = ir.as_storage_and_layout(dst)
            dst = ir.ReinterpretView(data=storage, layout=layout)
    assert isinstance(dst, ir.ReinterpretView), dst
    if isinstance(src, ir.TensorBox):
        # unwrap a TensorBox
        return cls.realize_into(src.data, dst)
    if isinstance(src, ir.StorageBox):
        src.realize()
        # ExternKernelAlloc has specific requirements for output layout, should create a copy
        assert hasattr(src.data, "layout")
        if cls.can_realize_into_without_copy(src):
            src.data.layout = ir.NonOwningLayout(dst)
            return src.data
    # introduce a copy
    input_graphs = fetch_graphs(src)
    node_name = f"clone_{next(node_id)}"
    new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.clone, node_name)
    pw = ir.Pointwise.create(
        device=src.get_device(),
        dtype=src.get_dtype(),
        inner_fn=src.make_loader(),
        ranges=[
            V.graph.sizevars.check_equals(a, b)
            for a, b in zip(src.get_size(), dst.get_size())
        ],
        traced_graph=new_graph,
        node_name=node_name,
    )
    return cls.realize_into(pw, dst)


ir.ConcatKernel.get_traced_graph = _patch_concatkernel_get_traced_graph
ir.ConcatKernel.realize_into = _patch_concatkernel_realize_into


def _patch_externkernel_copy_input(x):
    traced_graph = x.get_traced_graph()
    node_name = x.get_name()
    if traced_graph is None:
        traced_graph = fetch_graphs([x])[0]
        node_name = f"getitem_{next(node_id)}"
    pw = ir.Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
        traced_graph=traced_graph,
        node_name=node_name,
    )
    pw.realize()
    return pw


ir.ExternKernel.copy_input = _patch_externkernel_copy_input


@classmethod
def _patch_externkernel_convert_to_reinterpret_view(cls, x):
    """
    In order to pass this to an extern kernel we need a
    ReinterpretView not a View.  This allows us to avoid some
    unneeded copies.
    """
    assert isinstance(x, ir.BaseView)
    if isinstance(x, ir.ReinterpretView):
        return x

    # NOTE: Don't use extract_read_writes here as it fails when
    # make_loader() inlines the computation
    x_unwrap_view = x.unwrap_view()
    buf = V.graph.get_buffer(x_unwrap_view.get_name())
    assert buf is not None
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


ir.ExternKernel.convert_to_reinterpret_view = (
    _patch_externkernel_convert_to_reinterpret_view
)


@classmethod
def _patch_devicecopy_create(
    cls, x, device, non_blocking, traced_graph=None, node_name=None
):
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


ir.DeviceCopy.create = _patch_devicecopy_create
ir.DeviceCopy.get_traced_graph = _patch_devicecopy_get_traced_graph


def _patch_multioutput_get_traced_graph(self):
    return None


ir.MultiOutput.get_traced_graph = _patch_multioutput_get_traced_graph


def _patch_mutablebox_get_name(self):
    return self.data.get_name()


def _patch_mutablebox_get_traced_graph(self):
    return self.data.get_traced_graph()


ir.MutableBox.get_name = _patch_mutablebox_get_name
ir.MutableBox.get_traced_graph = _patch_mutablebox_get_traced_graph


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
        node_name = f"copy__{next(node_id)}"
        new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.copy, node_name)

        src = ir.Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.check_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
            traced_graph=new_graph,
            node_name=node_name,
        ).data

    src.realize()
    assert isinstance(src.data.layout, ir.FlexibleLayout)
    src.data.layout = ir.MutationLayoutSHOULDREMOVE(dst)
    return src.data


ir.MutationLayoutSHOULDREMOVE.realize_into = _patch_mutationlayout_realize_into
