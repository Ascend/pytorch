"""Build FxRT C++ GraphExecutor from a torch.fx GraphModule (no FxRT run)."""

from __future__ import annotations

import operator
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node

from fxrt.ir import GraphExecutor, Op
from fxrt import fx_backend as fxb
from fxrt.copy_elimination import eliminate_redundant_copy_
from fxrt.decompose_impl import _decompose_ops_with_fake_mode, _should_decompose_setitem
from fxrt.full_decomposition import decompose_full_
from fxrt.symbolic_shape import SymbolicShapeManager
from fxrt.utils import get_collective_info_from_torch, set_device_context


# This module intentionally reaches into fx_backend internals to apply
# converter-only op mappings / hooks without modifying fx_backend.py source.
# pylint: disable=protected-access


# Converter-only op mappings that are not registered in fx_backend.py
# so that the fx_backend source code and flow stay unchanged.
_CONVERTER_OP_MAP_EXTRA = {
    torch.gather: Op.gather,
    torch.ops.aten._unsafe_view.default: Op.view,
    torch.ops.aten._unsafe_view: Op.view,
    torch.ops.aten.repeat.default: Op.repeat,
    torch.ops.aten.constant_pad_nd.default: Op.constant_pad_nd,
    torch.ops.aten.constant_pad_nd: Op.constant_pad_nd,
    torch._C._nn.pad: Op.constant_pad_nd,
    torch.ops.aten.pad.default: Op.constant_pad_nd,
    torch.ops.aten.pad: Op.constant_pad_nd,
    torch.ops.aten.full_like.default: Op.full_like,
    torch.ops.aten.full_like: Op.full_like,
    torch.full_like: Op.full_like,
    torch.ops.aten.gather: Op.gather,
    torch.ops.aten.gather.default: Op.gather,
    torch.ops.aten.ge.Scalar: Op.ge_scalar,
    torch.ops.aten.ge.Tensor: Op.ge,
    torch.ops.aten.ge: Op.ge,
    torch.ops.aten.ne.Scalar: Op.ne,
    torch.ops.aten.ne.Tensor: Op.ne,
    torch.ops.aten.ne: Op.ne,
    torch.ops.aten.mul.Scalar: Op.mul_scalar,
    torch.ops.aten.mul.Scalar_out: Op.mul_scalar,
    torch.ops.aten.sum.dim_IntList: Op.reduce_sum,
    torch.ops.aten.sum.default: Op.reduce_sum,
    torch.sum: Op.reduce_sum,
    "_unsafe_view": Op.view,
    "constant_pad_nd": Op.constant_pad_nd,
    "pad": Op.constant_pad_nd,
    "full_like": Op.full_like,
    "gather": Op.gather,
    "ge": Op.ge,
    "ne": Op.ne,
    "mul_scalar": Op.mul_scalar,
    "reduce_sum": Op.reduce_sum,
    "sum": Op.reduce_sum,
}


# Converter-only size/stride helper. Must not call int(SymInt): that specialises
# Dynamo's live ShapeEnv while inductor compile_graph is still running.
def _concretize_shape_dim_for_abi0(dim: Any) -> Any:
    """Keep static Python ints; leave SymInt / FX nodes symbolic.

    ``int(symint)`` installs equality guards (e.g. batch==120), which breaks
    RecSDK ``mark_dynamic`` when running inside inductor compile_graph:
    ConstraintViolationError if relax is off, or a new compiled graph per
    batch if on. fx_backend binds the same values with
    SymbolicShapeManager.from_torch_with_sym.
    """
    if isinstance(dim, bool):
        return dim
    if isinstance(dim, int):
        return dim
    if isinstance(dim, torch.SymInt):
        # Dynamic dim as ABI0 int_list sentinel. Never int(symint).
        return -1
    if isinstance(dim, Node):
        example = fxb._get_node_meta_value(dim)
        if example is None and "val" in dim.meta:
            example = dim.meta["val"]
        if isinstance(example, bool):
            return dim
        if isinstance(example, int):
            return example
        if isinstance(example, torch.SymInt):
            return -1
    return dim


def _concretize_int_sequence_for_abi0(arg: Any) -> Any:
    """Concretize a size/stride list, Size, or FX Node producing an int sequence."""
    if isinstance(arg, (list, tuple)):
        return type(arg)(_concretize_shape_dim_for_abi0(d) for d in arg)
    if isinstance(arg, torch.Size):
        return tuple(_concretize_shape_dim_for_abi0(d) for d in arg)
    if isinstance(arg, Node):
        example = fxb._get_node_meta_value(arg)
        if example is None and "val" in arg.meta:
            example = arg.meta["val"]
        if isinstance(example, (list, tuple, torch.Size)):
            return tuple(_concretize_shape_dim_for_abi0(d) for d in example)
    return _concretize_shape_dim_for_abi0(arg)


# pylint: disable=unused-argument
def shape_list_concretize_arg_hook(node, flat_args, executor):
    """Pass size lists through; static ints stay ints, SymInt stays symbolic."""
    del node, executor
    if len(flat_args) < 2:
        return flat_args
    args = list(flat_args)
    args[1] = _concretize_int_sequence_for_abi0(args[1])
    return args


# pylint: disable=unused-argument
def as_strided_concretize_arg_hook(node, flat_args, executor):
    """Normalize as_strided size/stride/offset lists without specializing SymInt.

    Static Python ints are kept. Dynamic dims stay SymInt so ShapeEnv is not
    pinned to the example batch. ViewAlias / Path C AsStrided receive concrete
    H/W when those dims are already Python ints.
    """
    del node, executor
    if len(flat_args) < 3:
        return flat_args
    args = list(flat_args)
    args[1] = _concretize_int_sequence_for_abi0(args[1])
    args[2] = _concretize_int_sequence_for_abi0(args[2])
    if len(args) >= 4:
        args[3] = _concretize_shape_dim_for_abi0(args[3])
    return args


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


def _apply_converter_op_mappings() -> None:
    """Register converter-only op mappings without modifying fx_backend.py."""
    fxb._OP_MAP.update(_CONVERTER_OP_MAP_EXTRA)
    fxb._VIEW_OP_SWITCH_NAMES.setdefault(Op.view, frozenset())
    fxb._VIEW_OP_SWITCH_NAMES[Op.view] = fxb._VIEW_OP_SWITCH_NAMES[Op.view] | frozenset({"_unsafe_view"})


def _register_converter_arg_hooks() -> None:
    """Register converter-only arg hooks without modifying fx_backend.py."""
    for op in (Op.expand, Op.view, Op.repeat):
        fxb.register_arg_mapping_hook(op, shape_list_concretize_arg_hook)
    fxb.register_arg_mapping_hook(Op.as_strided_view, as_strided_concretize_arg_hook)


def _restore_converter_op_mappings(
    old_op_map: Dict[Any, Op], old_view_switch_names: Dict[Op, frozenset]
) -> None:
    """Restore fx_backend op maps to their pre-converter state."""
    fxb._OP_MAP.clear()
    fxb._OP_MAP.update(old_op_map)
    fxb._VIEW_OP_SWITCH_NAMES.clear()
    fxb._VIEW_OP_SWITCH_NAMES.update(old_view_switch_names)


def _restore_converter_arg_hooks(old_hooks: Dict[Any, Optional[Callable]]) -> None:
    """Restore arg mapping hooks modified by the converter path."""
    for op, hook in old_hooks.items():
        if hook is None:
            fxb._ARG_MAPPING_HOOKS.pop(op, None)
        else:
            fxb._ARG_MAPPING_HOOKS[op] = hook


@contextmanager
def _converter_backend_context():
    """
    Temporarily apply converter-only op mappings and arg hooks around FX-to-IR
    conversion so that fx_backend.py global state is restored afterward.
    """
    old_op_map = dict(fxb._OP_MAP)
    old_view_switch_names = {k: frozenset(v) for k, v in fxb._VIEW_OP_SWITCH_NAMES.items()}
    hook_ops = (Op.expand, Op.view, Op.repeat, Op.as_strided_view)
    old_arg_hooks = {op: fxb.get_arg_mapping_hook(op) for op in hook_ops}

    _apply_converter_op_mappings()
    _register_converter_arg_hooks()

    try:
        yield
    finally:
        _restore_converter_op_mappings(old_op_map, old_view_switch_names)
        _restore_converter_arg_hooks(old_arg_hooks)


def _bridge_make_fx_meta_to_example_value(gm: GraphModule) -> None:
    """make_fx(tracing_mode=symbolic) stores FakeTensor/SymInt in meta['val'];
    fx_backend reads meta['example_value'] (Dynamo). Bridge so sympy shapes bind.
    """
    for node in gm.graph.nodes:
        if "example_value" in node.meta:
            continue
        if "val" in node.meta:
            node.meta["example_value"] = node.meta["val"]


def build_c_graph_executor(gm: GraphModule, example_inputs: List[torch.Tensor]) -> GraphExecutor:
    """Convert FX GraphModule to FxRT C++ IR graph and return GraphExecutor.

    This reuses fx_backend conversion hooks but skips FxRT BuildExecutor/run.
    Converter-specific mappings/hooks are applied locally so that fx_backend.py
    source code and flow are not affected.
    """
    del example_inputs  # FX meta comes from node.meta during conversion
    with _converter_backend_context():
        _bridge_make_fx_meta_to_example_value(gm)
        graph_id = fxb._next_unique_graph_id()
        fxb._remove_matched_nodes(gm, fxb._OP_MATCHERS)
        eliminate_redundant_copy_(gm)
        decompose_full_(gm)
        _decompose_ops_with_fake_mode(gm, targets=_converter_decompose_targets())
        fxb._init_pre_flatten_hooks()
        fxb._init_arg_mapping_hooks()
        fxb._init_ops_mapping_hooks()
        fxb._init_output_mapping_hooks()
        fxb._init_fxrt_config()

        executor = GraphExecutor(f"fx_graph_{graph_id}")
        sym_mgr = SymbolicShapeManager()
        env: Dict[Node, Any] = {}

        get_collective_info_from_torch(gm)
        set_device_context()
        with executor:
            fx_input_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
            fxb._handle_input_nodes(fx_input_nodes, executor, env, sym_mgr)

            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "get_attr":
                    fxb._handle_get_attr_node(node, gm, executor, env)
                elif node.op in ("call_function", "call_method"):
                    fxb._handle_call_node(node, executor, env, sym_mgr)
                elif node.op == "call_module":
                    raise NotImplementedError("call_module is not supported")
                elif node.op == "output":
                    fxb._handle_output_node(node, executor, env, sym_mgr)
                else:
                    raise NotImplementedError(f"Unsupported node op: {node.op}")

    return executor
