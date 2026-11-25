"""Convert-once session used by the torch.compile backend.

``compile_executor`` exports an IR graph and compiles/loads it into a GE session once.
``CompiledModel.execute`` synchronizes torch's current NPU stream, then runs GE synchronously. GE allocates
each output at the InferShape'd size and ownership of that buffer moves to the
returned tensor.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # registers NPU module / _C helpers

from fxrt.ir import GraphExecutor  # producer side, provided by torch_npu

# ge::DataType → torch.dtype (values from CANN graph/c_types.h).
_GE_DTYPE_TO_TORCH = {
    0: torch.float32,  # DT_FLOAT
    1: torch.float16,  # DT_FLOAT16
    2: torch.int8,  # DT_INT8
    3: torch.int32,  # DT_INT32
    4: torch.uint8,  # DT_UINT8
    6: torch.int16,  # DT_INT16
    9: torch.int64,  # DT_INT64
    11: torch.float64,  # DT_DOUBLE
    12: torch.bool,  # DT_BOOL
    27: torch.bfloat16,  # DT_BF16
}


def _ge_dtype_to_torch(ge_dtype: int) -> torch.dtype:
    dt = _GE_DTYPE_TO_TORCH.get(int(ge_dtype))
    if dt is None:
        raise RuntimeError(f"fx_converter: unsupported ge::DataType {ge_dtype}")
    return dt


def _adopt_ge_outputs(views, device: torch.device) -> List[torch.Tensor]:
    """Adopt GE buffers once and reconstruct their logical output views (no copy).

    ``views`` carry ``(device_addr, owned_ctx, free_fn)``. The adoption itself lives in
    torch_npu's ``_fxrt_torch`` as a generic primitive rather than in a bridge module of
    our own: it needs torch and torch_npu C++ types, which this ABI=0 package cannot
    compile against, and it is not GE-specific. torch_npu guarantees every buffer is
    either adopted by exactly one tensor or freed once, so this never leaks even when one
    record is malformed.

    The GE dtype mapping stays on this side: torch_npu must not learn ``ge::DataType``.
    """
    from fxrt import _fxrt_torch  # pylint: disable=import-outside-toplevel

    data_ptrs, shapes, dtypes, owned_ctxs, free_fns = [], [], [], [], []
    storage_shapes, storage_formats, strides = [], [], []
    nonempty = [i for i, view in enumerate(views) if 0 not in view.shape]
    empty = [i for i, view in enumerate(views) if 0 in view.shape]
    for i in nonempty:
        view = views[i]
        data_ptrs.append(int(view.device_addr))
        shapes.append([int(d) for d in view.shape])
        dtypes.append(_ge_dtype_to_torch(view.dtype))
        owned_ctxs.append(int(view.owned_ctx))
        free_fns.append(int(view.free_fn))
        storage_shapes.append([int(d) for d in view.storage_shape])
        storage_formats.append(int(view.format))
        strides.append([int(step) for step in view.stride])

    index = device.index
    # An empty output has no payload to adopt. Construct it locally instead of
    # exercising the bridge's empty-Tensor nanobind/pybind11 conversion.
    bases = [None] * len(views)
    try:
        args = (data_ptrs, shapes, dtypes, owned_ctxs, free_fns, 0 if index is None else int(index))
        # Schema 5's bridge has no private-format arguments. ND needs none;
        # schema 6's bridge accepts the storage descriptor for private formats.
        if any(code != 2 for code in storage_formats) or any(strides):
            args += (storage_formats, storage_shapes)
        if any(strides):
            args += (strides,)
        adopted = _fxrt_torch.adopt_external_device_buffers(*args)
        for i, base in zip(nonempty, adopted):
            bases[i] = base
        for i in empty:
            bases[i] = torch.empty(views[i].shape, dtype=_ge_dtype_to_torch(views[i].dtype), device=device)
    finally:
        from fxrt.gert_backend import _fxrt_fx_converter
        for i in empty:
            _fxrt_fx_converter.free_owned(int(views[i].owned_ctx), int(views[i].free_fn))
    # Adopt each physical owner exactly once. as_strided creates only a header:
    # all logical aliases retain that base's storage and its single deleter.
    outputs = [None] * sum(len(view.logical_views) for view in views)
    for base, view in zip(bases, views):
        for spec in view.logical_views:
            outputs[spec.output_index] = (
                base.as_strided(spec.shape, spec.stride, spec.storage_offset)
                if spec.preserve_layout else base
            )
    return outputs


def _current_stream_handle(device_id: int = 0) -> int:
    """Raw aclrtStream of torch's current NPU stream, as an integer handle.

    The runner synchronizes this stream before GE reads the caller's input buffers.
    """
    return int(torch.npu.current_stream(device_id).npu_stream)


def _restore_symbolic_as_strided_offsets(exported) -> None:
    """Copy symbolic as_strided offsets from ordered args into the packed layout ABI.

    Older torch_npu exporters preserve ordered ``arguments`` but only pack concrete
    offsets into ``int_list``. The importer already understands an aligned ``sym_list``;
    restore that slot here so the converter can evaluate the expression per Execute.
    """
    if not isinstance(exported, dict):
        raise TypeError("fx_converter: exported IR must be a dictionary")
    nodes = exported.get("nodes", ())
    for node in nodes:
        if node.get("kind") != "op" or node.get("op") != "as_strided_view":
            continue
        arguments = node.get("arguments", ())
        if len(arguments) < 4:
            continue
        offset = arguments[3]
        offset_kind = offset.get("kind")
        if offset_kind == "int":
            continue
        if offset_kind == "none":
            continue
        if offset_kind == "ref":
            raise RuntimeError(
                "fx_converter: data-dependent as_strided storage_offset is unsupported; "
                "use a Slice/Gather node or a shape-derived SymInt offset"
            )
        if offset_kind != "symbol":
            raise RuntimeError(
                "fx_converter: unsupported as_strided storage_offset export kind "
                f"{offset_kind!r}"
            )
        expression = offset.get("value")
        if not isinstance(expression, str) or not expression:
            raise RuntimeError("fx_converter: symbolic as_strided storage_offset is empty")

        packed = node.get("int_list")
        if not packed:
            size_arg, stride_arg = arguments[1], arguments[2]
            if (size_arg.get("kind") != "tuple" or size_arg.get("elements") or
                    stride_arg.get("kind") != "tuple" or stride_arg.get("elements")):
                raise RuntimeError(
                    "fx_converter: symbolic as_strided offset has no packed size/stride layout"
                )
            packed = [0, 0]
            node["int_list"] = packed

        if not isinstance(packed, list) or not packed or not isinstance(packed[0], int):
            raise RuntimeError("fx_converter: malformed packed as_strided layout")
        rank = packed[0]
        expected_size = 2 * rank + 2
        if rank < 0 or len(packed) != expected_size:
            raise RuntimeError("fx_converter: incomplete packed as_strided layout")
        offset_index = 1 + 2 * rank
        sym_list = node.setdefault("sym_list", [""] * len(packed))
        if not isinstance(sym_list, list) or len(sym_list) != len(packed):
            raise RuntimeError("fx_converter: malformed symbolic as_strided layout")
        current = sym_list[offset_index]
        if current and current != expression:
            raise RuntimeError("fx_converter: conflicting symbolic as_strided storage_offset")
        sym_list[offset_index] = expression
        packed[offset_index] = -1


# ge::FORMAT_ND. Host tensors and a failed format query use this.
_FORMAT_ND = 2


def _query_feed_layouts(inputs: Sequence[torch.Tensor]):
    """Read format and storage shape together, once per feed, in C++.

    The converter module's ``feed_layouts`` calls ``at_npu::native::get_npu_format``
    from the ABI=1 side. That is the same read fxrt's updater does, and it replaces
    both ``_fxrt_torch.get_storage_shape`` and ``torch.ops.npu.get_npu_format``.
    Returns None when this build has no such entry (the caller keeps the old path).
    """
    try:
        from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    query = getattr(_fxrt_fx_converter, "feed_layouts", None)
    if query is None:
        return None
    raw_shapes, formats = query(list(inputs))
    storage_shapes = []
    for tensor, shape in zip(inputs, raw_shapes):
        storage_shapes.append(list(tensor.shape) if shape is None else list(shape))
    codes = [_FORMAT_ND if int(code) < 0 else int(code) for code in formats]
    return storage_shapes, codes


def _npu_format(tensor: torch.Tensor) -> int:
    """aclFormat of a real tensor, passed through unchanged.

    Same read as fxrt UpdateTensorFromTorch. Host tensors have no npu format.
    """
    if tensor.device.type != "npu":
        return _FORMAT_ND
    layouts = _query_feed_layouts([tensor])
    if layouts is not None:
        return layouts[1][0]
    try:
        code = int(torch_npu.get_npu_format(tensor))
    except (AttributeError, RuntimeError):
        return _FORMAT_ND
    if code < 0:
        return _FORMAT_ND
    return code


def _input_layouts(inputs: Sequence[torch.Tensor]):
    """Collect logical and physical descriptors in storage-base coordinates."""
    shapes = [list(t.shape) for t in inputs]
    strides = [list(t.stride()) for t in inputs]
    storage_offsets = [int(t.storage_offset()) for t in inputs]
    layouts = _query_feed_layouts(inputs)
    if layouts is not None:
        storage_shapes, formats = layouts
    else:
        from fxrt import _fxrt_torch  # pylint: disable=import-outside-toplevel

        storage_shapes = []
        for tensor in inputs:
            try:
                storage_shapes.append(list(_fxrt_torch.get_storage_shape(tensor)))
            except AttributeError:
                storage_shapes.append(list(tensor.shape))
        formats = [_npu_format(tensor) for tensor in inputs]
    for i, (shape, stride) in enumerate(zip(shapes, strides)):
        offset = storage_offsets[i]
        if offset < 0:
            raise ValueError(f"fx_converter: negative input storage_offset {offset}")
        # Private formats have their own physical packing; logical strides do
        # not describe that storage, and a logical offset cannot be applied to it.
        if formats[i] not in (0, 1, 2, 30):
            if offset != 0:
                raise ValueError("fx_converter: nonzero storage_offset is unsupported for private formats")
            continue
        span = 0 if 0 in shape else offset + 1 + sum(
            (size - 1) * step for size, step in zip(shape, stride))
        capacity = 1
        for size in storage_shapes[i]:
            capacity *= size
        storage_nbytes = inputs[i].untyped_storage().nbytes()
        item_size = inputs[i].element_size()
        if item_size <= 0 or storage_nbytes % item_size != 0:
            raise ValueError("fx_converter: input storage size is not aligned to its element size")
        tensor_capacity = storage_nbytes // item_size
        if span > tensor_capacity:
            raise ValueError(
                f"fx_converter: input view span {span} exceeds storage capacity {tensor_capacity}"
            )
        if capacity != tensor_capacity:
            # Layout metadata describes the tensor view; stride/offset address the
            # full backing allocation, whose capacity comes from PyTorch storage.
            storage_shapes[i] = [tensor_capacity]
    return shapes, strides, storage_shapes, storage_offsets, formats


def _storage_base_ptr(tensor: torch.Tensor) -> int:
    """Return the storage base; storage_offset remains a separate descriptor field."""
    return int(tensor.untyped_storage().data_ptr())


class CompiledModel:
    """Convert-once session: repeatedly Execute with new tensor addresses."""

    def __init__(self, session, device_id: int = 0):
        self._session = session
        self._device_id = device_id
        self._tensor_inputs = None
        try:
            from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel
        except ImportError:
            # Source-only users and session test doubles need no native buffer.
            return

        buffer_type = getattr(_fxrt_fx_converter, "TensorInputBuffer", None)
        self._tensor_inputs = (
            buffer_type() if buffer_type is not None and isinstance(session, _fxrt_fx_converter.ConverterSession)
            else None
        )

    def __del__(self):
        # Best-effort: Dynamo dropping the compiled callable should release its GE session,
        # matching GraphExecutor.__del__ on the fx_backend path.
        try:
            self.release()
        except Exception:  # pylint: disable=broad-except
            pass

    def release(self) -> None:
        """Drop this session's reference to the GE runner.

        Safe to call while previously returned tensors are alive: each of them holds its
        own reference to the runner that owns the allocator their memory came from.
        """
        if self._session is not None:
            self._session.release()
            self._session = None

    def execute(self, inputs: Sequence[torch.Tensor], scalar_syms=None) -> List[torch.Tensor]:
        """Run the graph synchronously through GE's session API.

        GE allocates each output at its InferShape'd size and the buffer is adopted by
        the returned tensor (no copy). The tensors own their memory, so they stay valid
        independently of later ``execute`` calls.

        The runner synchronizes torch's current stream before GE reads the inputs.
        Output shapes, addresses and data are complete when this call returns.

        ``scalar_syms`` is a list of ``(symbol name, value)`` pairs carrying the runtime
        values of symbolic scalars (Dynamo-lifted SymInt placeholders, e.g. an axis
        passed as a function argument).
        """
        if self._session is None:
            raise RuntimeError("CompiledModel: session has been released")
        if not inputs:
            raise RuntimeError("fx_converter: execute requires at least one input for device")
        device = inputs[0].device
        stream = _current_stream_handle(device.index if device.index is not None else self._device_id)
        if self._tensor_inputs is not None:
            views = self._tensor_inputs.execute(self._session, inputs, stream, scalar_syms or [])
            return _adopt_ge_outputs(views, device)
        in_shapes, in_strides, in_storage_shapes, in_storage_offsets, in_formats = _input_layouts(inputs)
        # TensorV2 consumers interpret storage_offset relative to the backing
        # storage. data_ptr() is already advanced to the logical view start.
        in_addrs = [int(t.data_ptr()) if offset == 0 else _storage_base_ptr(t)
                    for t, offset in zip(inputs, in_storage_offsets)]
        runtime_storage_shapes = in_storage_shapes
        runtime_storage_offsets = in_storage_offsets
        views = self._session.execute_alloc(
            in_addrs, in_shapes, in_strides, runtime_storage_shapes, runtime_storage_offsets, stream,
            scalar_syms or [], in_formats,
        )
        return _adopt_ge_outputs(views, device)


def _sync_cube_precision(module) -> None:
    """Mirror torch's hf32 switches into the converter's shared OpExecuteOptions.

    torch-npu keeps allow_hf32 in its own option map and hands aclnn a cubeMathType
    per call; a GE-built graph instead reads it back from OpExecuteOptions at launch.
    Without this the converter runs every matmul in full fp32 while eager runs hf32 --
    more accurate, but a mismatch the model's own precision check reports as a failure.
    """
    module.set_allow_hf32(bool(torch_npu.npu.conv.allow_hf32), bool(torch_npu.npu.matmul.allow_hf32))


def _register_fallback_apis(module) -> None:
    """Hand the ABI=1 torch/python call entry points to the ABI=0 converter."""
    try:
        from fxrt import _fxrt_torch_bridge  # pylint: disable=import-outside-toplevel
    except ImportError:
        return

    module.register_torch_fallback_api(_fxrt_torch_bridge.fallback_api_address())
    module.register_python_call_api(_fxrt_torch_bridge.python_call_api_address())


def compile_executor(executor: GraphExecutor, device_id: int = 0) -> CompiledModel:
    """Export ``executor``, Convert + Init ModelV2 once, return a reusable session."""
    from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel
    from fxrt.gert_backend.op_register import register_overlay_ops  # pylint: disable=import-outside-toplevel

    # Apply the Python overlay table so a row appended in op_register.py takes effect here.
    register_overlay_ops()
    # Hand the ABI=1 torch/python fallback entry points to the ABI=0 converter.
    _register_fallback_apis(_fxrt_fx_converter)
    _sync_cube_precision(_fxrt_fx_converter)
    # Device memory is wired to torch_npu's caching allocator inside Convert
    # (FxRegisterTorchPoolAllocator), so nothing to hand across from here.
    exported = executor.export_ir_graph()
    _restore_symbolic_as_strided_offsets(exported)
    layouts = getattr(executor, "_converter_feed_layouts", {})
    feed_strides = getattr(executor, "_converter_feed_strides", {})
    counts = {"input": 0, "parameter": 0}

    def literal(value):
        if isinstance(value, torch.SymInt):
            return int(value.node.expr) if value.node.expr.is_Integer else -1
        return int(value)
    for node in exported["nodes"]:
        kind = node["kind"]
        if kind not in counts or kind not in layouts:
            continue
        index = counts[kind]
        counts[kind] += 1
        if index >= len(layouts[kind]):
            raise RuntimeError("converter feed metadata does not match exported inputs")
        example = layouts[kind][index]
        node["stride"] = [literal(value) for value in example.stride()]
        if kind in feed_strides:
            node.setdefault("attrs", {})["fxrt_feed_stride"] = feed_strides[kind][index]
        node.setdefault("attrs", {})["fxrt_proven_dense"] = {
            "kind": "bool", "value": bool(example.is_contiguous())}
        if all(size >= 0 for size in node["shape"] + node["stride"]):
            span = 0 if 0 in node["shape"] else 1 + sum(
                (size - 1) * stride for size, stride in zip(node["shape"], node["stride"]))
            capacity = 1
            for size in node.get("storage_shape", node["shape"]):
                capacity *= size
            if span > capacity:
                node["storage_shape"] = [span]
    # Execute uses torch's current stream. Init loads on a private stream and
    # drains only that one — syncing torch's stream here deadlocks inductor.
    session = _fxrt_fx_converter.convert(exported, device_id, _current_stream_handle(device_id))
    return CompiledModel(session, device_id)
