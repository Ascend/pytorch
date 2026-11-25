"""FxConverter torch.compile backend.

Flow (Convert once, Execute many)::

    GraphModule
        │  build_c_graph_executor   (reuse fx_backend FX→IR hooks)
        ▼
    GraphExecutor.export_ir_graph
        │  compile_executor         (ABI0 rebuild + Convert + ModelV2 Init)
        ▼
    CompiledModel (session, owned by the returned callable — same as fx_backend)
        │  each call:
        │    1. keep model-feed args (Dynamo may add SymInts / 0-d closure scalars)
        │    2. session.execute(tensors)  — GE allocates outputs, ownership moves to torch
        │    3. pack return to match FX output nesting (_output_pack)
        ▼
    user tensor(s)

Output shapes come from Main InferShape (GE-aligned). GE allocates each output at exactly
that size and the buffer is adopted by the returned torch tensor (no copy); the tensor's
deleter returns it to GE, so outputs of different calls are independent and stay valid for
as long as they are referenced.

Debug dumps (same switches as fx_backend / Convert):
  FXRT_DEV_DUMP_IR=1          FX GraphModule + FxRT IR → graph_{pid}.txt
  FX_CONVERTER_DUMP_GRAPH=1   GE ExecuteGraph (Convert, stderr)
"""

from __future__ import annotations

import weakref
from itertools import count
from typing import Any, List, Sequence

import torch
from torch.fx import GraphModule, Node

from fxrt.gert_backend._fx_to_c_graph import build_c_graph_executor
from fxrt.gert_backend._output_pack import fx_output_tree, pack_like_fx
from fxrt.gert_backend.rewrites.output import add_out_return_refs
from fxrt.gert_backend._session import compile_executor

# Producer side, all provided by torch_npu. Note the flat module paths: torch_npu merged
# fxrt with its python/fxrt/torch/ subpackage flattened away, so fxrt.torch.fx_backend
# does not exist there even though it still does in the fxrt repository.
from fxrt import fx_backend as fxb
from fxrt.utils import set_device_context

_DUMP_GRAPH_ID = count(1)


class _CompiledSessionRegistry:
    """Expose live sessions to tests without extending their ownership lifetime."""

    def __init__(self):
        self._refs = []

    def _live_sessions(self):
        refs = []
        sessions = []
        for ref in self._refs:
            session = ref()
            if session is not None:
                refs.append(ref)
                sessions.append(session)
        self._refs = refs
        return sessions

    def append(self, session):
        self._live_sessions()
        self._refs.append(weakref.ref(session))

    def remove(self, session):
        self._live_sessions()
        for index, ref in enumerate(self._refs):
            if ref() is session:
                del self._refs[index]
                return
        raise ValueError("session is not registered")

    def __getitem__(self, index):
        return self._live_sessions()[index]

    def __iter__(self):
        return iter(self._live_sessions())

    def __len__(self):
        return len(self._live_sessions())


_compiled_sessions = _CompiledSessionRegistry()


def _fx_output_tree(gm: GraphModule) -> Any:
    for node in gm.graph.nodes:
        if node.op == "output":
            return node.args[0] if node.args else ()
    raise RuntimeError("fx_converter_backend: FX graph has no output node")


def _fx_output_dtypes(tree: Any) -> list:
    """Flat expected dtypes for each leaf of the FX output tree (None when unknown)."""
    leaves = (tree,) if isinstance(tree, Node) else tuple(tree)
    expected = []
    for leaf in leaves:
        # Dynamo fills "example_value" on this stack; upstream also knows "val".
        val = None
        if isinstance(leaf, Node):
            val = leaf.meta.get("val", leaf.meta.get("example_value"))
        expected.append(getattr(val, "dtype", None))
    return expected


def _widen_int32_indices(packed: Any, expected_dtypes: Sequence[Any]) -> Any:
    """Widen a GE int32 buffer to int64 where the FX graph declares the output int64.

    GE sort-family bins (TopKV2 / Sort) write INT32 indices while torch declares int64.
    Driving the widening off the FX output declaration covers both the topk
    (values, indices) tuple and the single-tensor argsort return.
    """
    if torch.is_tensor(packed):
        if expected_dtypes and expected_dtypes[0] == torch.int64 and packed.dtype == torch.int32:
            return packed.to(torch.int64)
        return packed
    if isinstance(packed, (tuple, list)):
        items = list(packed)
        for i, item in enumerate(items):
            if (i < len(expected_dtypes) and expected_dtypes[i] == torch.int64 and torch.is_tensor(item) and
                    item.dtype == torch.int32):
                items[i] = item.to(torch.int64)
        return type(packed)(items)
    return packed


def _pack_like_fx(flat: Sequence[torch.Tensor], tree: Any) -> Any:
    """Rebuild return value with the same nesting as the FX output node."""
    if torch.is_tensor(tree) or isinstance(tree, Node):
        if len(flat) != 1:
            raise RuntimeError("fx_converter_backend: expected a single tensor output")
        return flat[0]
    if isinstance(tree, (tuple, list)):
        it = iter(flat)
        packed = []
        for item in tree:
            if isinstance(item, (tuple, list)):
                raise RuntimeError("fx_converter_backend: nested output tuples are not supported yet")
            try:
                packed.append(next(it))
            except StopIteration as exc:
                raise RuntimeError("fx_converter_backend: fewer outputs than FX tree expects") from exc
        leftover = list(it)
        if leftover:
            raise RuntimeError("fx_converter_backend: more outputs than FX tree expects")
        return type(tree)(packed)
    raise RuntimeError(f"fx_converter_backend: unsupported FX output tree {type(tree)}")


def _device_id_from_args(args: Sequence[Any]) -> int:
    for x in args:
        if not torch.is_tensor(x):
            continue
        try:
            idx = x.device.index
        except (AttributeError, RuntimeError):
            continue
        return 0 if idx is None else int(idx)
    return 0


def _consumed_as_op_tensor(node) -> bool:
    """True when a placeholder is consumed by an op as a tensor input.

    Dynamo lifts closed-over scalars / python-number args as 0-d tensor placeholders
    whose only consumer is an ``item`` call (the value folds to a constant at import).
    A 0-d placeholder consumed directly by an op (e.g. a rank-0 ``scales`` for
    npu_quantize) is a real tensor input and must stay a runtime feed. Mirrors
    ConsumedAsTensorInput (mrt_node_utils.cc).
    """
    # aten._local_scalar_dense is the traced form of Tensor.item; there is no public alias.
    item_targets = (
        torch.ops.aten.item.default,
        torch.ops.aten._local_scalar_dense.default,  # pylint: disable=protected-access
    )
    for user in node.users:
        if user.op == "call_method" and user.target == "item":
            continue
        if user.op == "call_function" and user.target in item_targets:
            continue
        return True
    return False


def _is_model_feed(node) -> bool:
    """True when a placeholder becomes a ModelV2 Data feed.

    Mirrors IsModelFeedTensor: SymInt placeholders and 0-d tensors consumed only
    through item are folded constants. A 0-d tensor consumed by an op stays a feed.
    Missing meta stays a feed and is checked again from the runtime argument.
    """
    meta = node.meta.get("example_value", None)
    if meta is None or (torch.is_tensor(meta) and meta.dim() > 0):
        return True
    return bool(torch.is_tensor(meta) and _consumed_as_op_tensor(node))


def _model_feed_positions(gm: GraphModule) -> List[int]:
    """Placeholder positions in ModelV2 Data order.

    Export writes every non-Parameter feed as input_* and then every Parameter
    feed as param_*. Each group keeps placeholder order. Dynamo calls the graph
    with parameters before user inputs, so placeholder order puts a weight in
    Gather's index slot. Call after build_c_graph_executor so make_fx
    meta['val'] is bridged to meta['example_value'].
    """
    user_feeds: List[int] = []
    param_feeds: List[int] = []
    for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder"):
        if not _is_model_feed(node):
            continue
        meta = node.meta.get("example_value", None)
        if isinstance(meta, torch.nn.Parameter):
            param_feeds.append(idx)
        else:
            user_feeds.append(idx)
    return user_feeds + param_feeds


def _feed_shape_symbol_names(gm: GraphModule) -> set:
    """Symbol names that are dimensions of a tensor placeholder.

    ``dynamic=True`` lifts each dynamic extent to its own SymInt placeholder. Those
    values are the feed tensor's shape, which Execute rebinds from the feed. They are
    not scalar operands.
    """
    names = set()
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        meta = node.meta.get("example_value", None)
        if not torch.is_tensor(meta):
            continue
        for dim in meta.shape:
            if isinstance(dim, torch.SymInt) and dim.node.expr.is_Symbol:
                names.add(str(dim.node.expr))
    return names


def _scalar_sym_slots(gm: GraphModule) -> List:
    """(placeholder position, symbol name) for each scalar SymInt placeholder.

    A Dynamo-lifted scalar (e.g. an axis passed as a function argument under
    ``dynamic=True``) arrives as a SymInt placeholder. It is not a model feed, but its
    runtime value is needed to refresh the materialized axis payload, so the compiled
    callable binds it into the symbol env on every call. The name is the sympy symbol
    text (e.g. ``s1``), which is exactly what SymbolicShapeManager interned at export.
    Shape-dimension placeholders are omitted; Execute binds those from the feed tensor.
    """
    shape_names = _feed_shape_symbol_names(gm)
    slots = []
    for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder"):
        meta = node.meta.get("example_value", None)
        if isinstance(meta, torch.SymInt) and meta.node.expr.is_Symbol:
            name = str(meta.node.expr)
            if name not in shape_names:
                slots.append((idx, name))
    return slots


def backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """torch.compile backend entry: ``(gm, example_inputs) -> callable``."""
    set_device_context()
    device_id = _device_id_from_args(example_inputs)

    # Compile: FX → IR → Convert + Init (once).
    dump_fx = fxb.is_enable_dump_ir()
    if dump_fx:
        fxb.write_gm_graph(gm, next(_DUMP_GRAPH_ID), fxb.get_ir_file_name())
    executor = build_c_graph_executor(gm, example_inputs)
    if dump_fx:
        fxb.write_fxrt_graph(executor.dump_graph(print_stdout=False), fxb.get_ir_file_name())
    # One Convert session per graph, owned by this callable (fx_backend owns GraphExecutor
    # the same way). Dynamo may keep several compiled graphs; do not evict the previous one.
    compiled = compile_executor(executor, device_id=device_id)
    _compiled_sessions.append(compiled)
    output_tree = fx_output_tree(gm)
    out_refs = add_out_return_refs(gm, output_tree)
    feed_positions = _model_feed_positions(gm)
    sym_slots = _scalar_sym_slots(gm)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    def compiled_callable(*args: Any):
        # Dynamo passes one arg per placeholder; only model-feed positions take tensors.
        tensor_inputs = [args[i] for i in feed_positions if torch.is_tensor(args[i])]
        if not tensor_inputs:
            raise RuntimeError("fx_converter_backend: no tensor inputs")
        # SymInt placeholders arrive as plain ints at runtime; bind them by symbol name
        # so a symbolic axis payload is refreshed with this call's value.
        scalar_syms = [(name, int(args[i])) for i, name in sym_slots]
        set_device_context()
        outs = compiled.execute(tensor_inputs, scalar_syms)
        if outs is None:
            raise RuntimeError("fx_converter_backend: execute returned no tensors")
        for index, position in out_refs:
            outs[index] = args[position]
        return pack_like_fx(outs, output_tree, args, placeholders)

    return compiled_callable
