"""Build an fxrt ir::Graph from a torch.fx GraphModule, ready for Convert.

This is the converter's FX prelude. It reuses fx_backend's FX-to-IR hooks but stops
before fxrt would build and run an executor, because what happens next is Convert, not
an fxrt run.

It lives in this package rather than in torch_npu because it has no other consumer:
every pass here exists to make a graph that GE can lower unambiguously -- views are
normalized against the layout contract in _layout_legalize, matmul operand layouts are
pinned so DualStage can resolve transpose_x1/x2, and a dynamic dim is handed over as a
symbol rather than a sentinel so the ABI0 side has something to evaluate. torch_npu's own
backend, fx_backend.backend, has its own prelude and shares none of this.

Keeping it here also keeps the dependency one-way: this package imports torch_npu, and
torch_npu knows nothing about this package.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node

from fxrt.gert_backend import rewrites
from fxrt.gert_backend._layout_legalize import _is_as_strided_node, apply_torch_layout_contract_
from fxrt.gert_backend._symbolic_shape import ConverterSymbolicShapeManager
from fxrt.gert_backend._fallback_metadata import fallback_output_contract
from fxrt.gert_backend._view_metadata import VIEW_LAYOUT_ATTR, infer_native_view_graph_layouts, is_native_index_view
from fxrt.gert_backend.rewrites._common import int_tuple_node

# Producer side, all provided by torch_npu. Note the flat module paths: torch_npu merged
# fxrt with its python/fxrt/torch/ subpackage flattened away.
from fxrt.ir import GraphExecutor, Op
from fxrt import fx_backend as fxb
from fxrt.compiled_kernel_adapter import (
    normalize_fresh_allocation_layouts_,
    _rewrite_noncontiguous_empty_strided_,
    is_compiled_kernel_wrapper_mutation_target,
    restore_compiled_kernel_arg_layouts_,
)
from fxrt.view_offset import (
    OFFSET_SEMANTICS_ATTR, classify_inductor_view_offset,
    has_absolute_as_strided_offset, view_offset_semantics,
)
from fxrt.copy_elimination import eliminate_redundant_copy_
from fxrt.decompose_impl import _decompose_ops_with_fake_mode, _should_decompose_setitem
from fxrt.utils import get_collective_info_from_torch, set_device_context


# This module intentionally reaches into fx_backend internals to drive its conversion
# pipeline; the converter-only op mappings / hooks / graph rewrites live in the rewrites
# package. The overlays stay in fx_backend's global tables for the rest of the process;
# mixing the plain fx_backend conversion path and the converter backend in one process is
# not supported.
# pylint: disable=protected-access


# Compatibility import for diagnostics that inspected the previous classifier.
_has_absolute_as_strided_offset = has_absolute_as_strided_offset


def _mark_inductor_intermediate_as_strided_offsets(gm: GraphModule) -> None:
    """Validate all host-view spellings, including views directly on feeds."""
    has_compiled_kernel = any(
        node.op == "call_function" and is_compiled_kernel_wrapper_mutation_target(node.target)
        for node in gm.graph.nodes
    )
    if not has_compiled_kernel and not gm.meta.get("_fxrt_inductor_host_graph", False):
        return
    for node in gm.graph.nodes:
        classify_inductor_view_offset(node)


def _is_tensor_getitem_base(base) -> bool:
    """True when getitem base is a tensor FX node (not a Python/FX tuple)."""
    if not isinstance(base, Node):
        return False
    if base.type == torch.Tensor:
        return True
    example_value = base.meta.get("example_value", None)
    return isinstance(example_value, (torch.Tensor, FakeTensor))


def _converter_should_decompose_getitem(node: Node) -> bool:
    """
    Converter predicate for operator.getitem decomposition.

    Skip decomposition when getitem_impl can lower the node directly:
      - tuple/list base -> tuple_getitem
      - tensor + int/slice/tuple/Node/tensor -> getitem_slice / gather_v2
    Otherwise allow decomposition for exotic index types.
    """
    if not node.args:
        return False

    base = node.args[0]

    if isinstance(base, (tuple, list)):
        return False

    if isinstance(base, Node):
        example_value = base.meta.get("example_value", None)
        if isinstance(example_value, (tuple, list)):
            return False

    if _is_tensor_getitem_base(base) and len(node.args) >= 2:
        index = node.args[1]
        if isinstance(index, (int, slice, tuple, Node, torch.Tensor, FakeTensor)):
            return False

    return True


def _converter_decompose_targets() -> Dict[Any, Optional[Callable[[Node], bool]]]:
    """Return decomposition targets used by the converter path."""
    return {
        operator.setitem: _should_decompose_setitem,
        operator.getitem: _converter_should_decompose_getitem,
    }


@contextmanager
def _converter_backend_context():
    """Mark, in one place, the exact delta the converter applies on top of fx_backend.

    Everything registered here lives in the rewrites package: converter-only op mappings
    (op_map) and the arg / ops / output lowering hooks; the as_strided graph rewrite
    self-registers at import. The overlays are idempotent dict assignments that stay for
    the rest of the process -- deliberately no snapshot/restore, because the converter
    does not support sharing one process with the plain fx_backend conversion path.
    """
    rewrites.op_map.register()
    rewrites.arg.register()
    rewrites.ops.register()
    rewrites.output.register()
    rewrites.op_support.register()
    rewrites.npu_fusion_attention.register()
    yield


def _bridge_make_fx_meta_to_example_value(gm: GraphModule) -> None:
    """make_fx(tracing_mode=symbolic) stores FakeTensor/SymInt in meta['val'];
    fx_backend reads meta['example_value'] (Dynamo). Bridge so sympy shapes bind.
    """
    for node in gm.graph.nodes:
        if "example_value" in node.meta:
            continue
        if "val" in node.meta:
            node.meta["example_value"] = node.meta["val"]


def _stamp_feed_formats(gm: GraphModule, example_inputs: List[torch.Tensor]) -> None:
    """Copy real feed formats onto the FakeTensor examples FX tracing kept.

    Dynamo's placeholder example is a FakeTensor, and get_npu_format rejects it.
    The backend is also handed the real feeds, in placeholder order. Private
    formats have to be on those examples before from_torch builds the IR.
    """
    from fxrt.gert_backend._session import _npu_format  # pylint: disable=import-outside-toplevel

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    # dynamic=True inserts SymInt placeholders. Pair tensors in order; a length
    # mismatch means a rewrite added a feed this list does not know about.
    tensor_nodes = []
    for node in placeholders:
        example = node.meta.get("example_value", None)
        if torch.is_tensor(example):
            tensor_nodes.append(node)
    real_tensors = [real for real in example_inputs if torch.is_tensor(real)]
    if len(tensor_nodes) != len(real_tensors):
        return
    for node, real in zip(tensor_nodes, real_tensors):
        example = node.meta.get("example_value", None)
        code = _npu_format(real)
        try:
            example._fx_converter_format = code  # pylint: disable=protected-access
        except (AttributeError, TypeError):
            continue


def _fallback_output_dense(value: Any) -> bool:
    """Prove the bridge exports this result as a dense, zero-offset tensor."""
    if not torch.is_tensor(value):
        return False
    try:
        contiguous = value.is_contiguous()
        offset = value.storage_offset()
        base = value._base
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if type(contiguous) is not bool:
        return False
    if isinstance(offset, torch.SymInt):
        expr = offset.node.expr
        offset_zero = bool(expr.is_Integer and int(expr) == 0)
    else:
        offset_zero = isinstance(offset, int) and offset == 0
    if contiguous:
        return offset_zero
    # PR47984's bridge calls contiguous() before exporting a non-contiguous
    # result. Allow that copy only when FakeTensor metadata proves this result
    # is freshly allocated; a view may alias an input and must keep its contract.
    return offset_zero and base is None


def _fallback_output_dense_flags(value: Any) -> tuple:
    if torch.is_tensor(value):
        return (_fallback_output_dense(value),)
    if isinstance(value, (tuple, list)):
        return tuple(flag for item in value for flag in _fallback_output_dense_flags(item))
    return ()


def build_c_graph_executor(gm: GraphModule, example_inputs: List[torch.Tensor]) -> GraphExecutor:
    """Convert FX GraphModule to FxRT C++ IR graph and return GraphExecutor.

    This reuses fx_backend conversion hooks but skips FxRT BuildExecutor/run.
    Converter-specific mappings/hooks live in the rewrites package and are
    registered below; fx_backend.py source stays untouched. The overlays are
    idempotent dict assignments that stay for the rest of the process --
    deliberately no snapshot/restore, because the converter does not support
    sharing one process with the plain fx_backend conversion path.
    """
    with _converter_backend_context():
        _bridge_make_fx_meta_to_example_value(gm)
        graph_id = fxb._next_unique_graph_id()
        _mark_inductor_intermediate_as_strided_offsets(gm)
        fxb._remove_matched_nodes(gm, fxb._OP_MATCHERS)
        # Converter-only rewrites. fx_backend.backend does not run this list.
        rewrites.apply_converter_graph_rewrites(gm)
        # Same first step as fx_backend.backend: a compiled kernel addresses its args with
        # the strides its FX node carries, so those have to be pinned down while the view
        # chains that produced them are still in the graph to read.
        normalize_fresh_allocation_layouts_(gm)
        restore_compiled_kernel_arg_layouts_(gm, preserve_feed_offsets=True)
        # Standalone non-contiguous allocations are not kernel args; the converter can
        # only prove contiguity for concrete strides, so express them as alloc + view.
        _rewrite_noncontiguous_empty_strided_(gm)
        eliminate_redundant_copy_(gm)
        # Factory mappings preserve full's scalar as a Fill input. The regular
        # fx_backend retains its own zeros + fill_ decomposition.
        _decompose_ops_with_fake_mode(gm, targets=_converter_decompose_targets())
        # Decomposition is what turns elementwise chains into symbolic empty_strided
        # allocations, so the rewrite has to see the graph again after it.
        _rewrite_noncontiguous_empty_strided_(gm)
        apply_torch_layout_contract_(gm)
        # Recheck nodes created by rewrites; source nodes were classified before their
        # metadata could be changed by graph transformations.
        _mark_inductor_intermediate_as_strided_offsets(gm)
        fxb._init_pre_flatten_hooks()
        fxb._init_arg_mapping_hooks()
        fxb._init_ops_mapping_hooks()
        fxb._init_output_mapping_hooks()
        fxb._init_fxrt_config()
        # fx_backend init overwrites converter table hooks; re-apply after init.
        # op_support refreshes the catalog name set so a Python overlay registered
        # since the last graph is visible to this build.
        rewrites.arg.register()
        rewrites.ops.register()
        rewrites.output.register()
        rewrites.op_support.register()
        # _init_* above replaces hook dict entries; this op's hooks must win after that.
        rewrites.npu_fusion_attention.register()

        forced_names = rewrites.op_support._forced_custom_call_names()
        forced_nodes = {
            node for node in gm.graph.nodes
            if node.op in {"call_function", "call_method"}
            and getattr(fxb._get_op(node.target), "name", "") in forced_names
        }
        native_layouts = infer_native_view_graph_layouts(gm, forced_nodes)

        executor = GraphExecutor(f"fx_graph_{graph_id}")
        sym_mgr = ConverterSymbolicShapeManager()
        env: Dict[Node, Any] = {}

        get_collective_info_from_torch(gm)
        set_device_context()
        with executor:
            fx_input_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
            _stamp_feed_formats(gm, example_inputs)
            fxb._handle_input_nodes(fx_input_nodes, executor, env, sym_mgr)
            for input_node in fx_input_nodes:
                example = fxb._get_node_meta_value(input_node)
                if torch.is_tensor(example) and hasattr(env[input_node], "set_attr"):
                    env[input_node].set_attr("fxrt_feed_stride", sym_mgr.from_torch_with_sym(tuple(example.stride())))

            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "get_attr":
                    fxb._handle_get_attr_node(node, gm, executor, env)
                elif node.op in ("call_function", "call_method"):
                    node.meta.pop("_fxrt_converter_custom_call", None)
                    layouts = native_layouts.get(node)
                    if layouts is not None and is_native_index_view(node):
                        # Basic indexing is a relative layout operation. Preserve
                        # its symbolic PyTorch geometry before legacy hooks try
                        # to concretize slice bounds or expand a stepped slice.
                        value = fxb._get_node_meta_value(node)
                        arguments = [env[node.args[0]],
                                     int_tuple_node(executor, sym_mgr, tuple(value.shape)),
                                     int_tuple_node(executor, sym_mgr, tuple(value.stride())),
                                     executor.add_value_node(sym_mgr.from_torch_with_sym(value.storage_offset()))]
                        env[node] = executor.add_op_node(
                            Op.as_strided_view, arguments, sym_mgr.from_torch_with_sym(value))
                    else:
                        fxb._handle_call_node(node, executor, env, sym_mgr)
                    if layouts is not None and hasattr(env[node], "set_attr"):
                        env[node].set_attr(VIEW_LAYOUT_ATTR, sym_mgr.from_torch_with_sym(layouts))
                    if _is_as_strided_node(node) and hasattr(env[node], "set_attr"):
                        env[node].set_attr(
                            OFFSET_SEMANTICS_ATTR, sym_mgr.from_torch_with_sym(int(view_offset_semantics(node)))
                        )
                    output_dense = _fallback_output_dense_flags(fxb._get_node_meta_value(node))
                    if output_dense and hasattr(env[node], "set_attr"):
                        env[node].set_attr("fxrt_fallback_output_dense", sym_mgr.from_torch_with_sym(output_dense))
                    if node.meta.get("_fxrt_converter_custom_call", False):
                        aliases, fresh = fallback_output_contract(node)
                        env[node].set_attr("fxrt_fallback_read_aliases", sym_mgr.from_torch_with_sym(aliases))
                        env[node].set_attr("fxrt_fallback_fresh_dense", sym_mgr.from_torch_with_sym(fresh))
                elif node.op == "call_module":
                    raise NotImplementedError("call_module is not supported")
                elif node.op == "output":
                    fxb._handle_output_node(node, executor, env, sym_mgr)
                else:
                    raise NotImplementedError(f"Unsupported node op: {node.op}")

    # The producer's FakeTensor bridge retains shape but can replace symbolic
    # input strides with row-major defaults. Carry feed metadata to the
    # converter exporter without changing fx_backend's tensor ABI.
    executor._converter_feed_layouts = {
        "input": [fxb._get_node_meta_value(n) for n in fx_input_nodes
                  if isinstance(fxb._get_node_meta_value(n), torch.Tensor)
                  and not isinstance(fxb._get_node_meta_value(n), torch.nn.Parameter)],
        "parameter": [fxb._get_node_meta_value(n) for n in fx_input_nodes
                      if isinstance(fxb._get_node_meta_value(n), torch.nn.Parameter)],
    }
    # The installed torch_npu exporter need not preserve input-node attributes.
    # Carry their symbolic stride through the same dictionary ABI used by ops.
    executor._converter_feed_strides = {
        kind: [sym_mgr.export_int_list(value.stride()) for value in values]
        for kind, values in executor._converter_feed_layouts.items()
    }
    return executor
