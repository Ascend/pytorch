from typing import Any, Callable, Dict, Optional, Sequence

import sympy

from torch_npu.npu._aclgraph_update_plan.resolver import (
    ACLGRAPH_UPDATE_PLAN_GLOBAL,
    _get_update_specs,
    _normalize_op_name,
)
from torch_npu.utils._error_code import ErrCode, pta_error


def _handler_for_op(op_name: str) -> Optional[Any]:
    import torch_npu.npu._npugraph_handlers  # noqa: F401
    from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

    return _NPU_GRAPH_OP_HANDLERS.get(op_name)


def _matches_aclgraph_update_exclusion(op_name: str, bound: Dict[str, Any]) -> bool:
    normalized_op = _normalize_op_name(op_name)
    if normalized_op in {
        "npu_fusion_attention_v3",
        "npu_fusion_attention_grad_v3",
    }:
        return bound.get("input_layout") == "BNSD"
    return False


def _literal_to_source(value: Any) -> Dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        raise RuntimeError(
            "Unsupported ACLGraph update Tensor constant; Tensor update values must be graph inputs",
            pta_error(ErrCode.PARAM),
        )
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (int, float, bool, str)):
        return {"kind": "constant", "value": value}
    raise RuntimeError(
        f"Unsupported ACLGraph update source: {value!r}",
        pta_error(ErrCode.PARAM),
    )


def _maybe_sympy_expr(value: Any) -> Optional[sympy.Expr]:
    if isinstance(value, sympy.Expr):
        return value
    import torch

    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return value.node.expr
    return None


def _try_get_name(value: Any) -> Optional[str]:
    import torch

    if isinstance(value, torch.Tensor):
        return None

    if hasattr(value, "get_name"):
        try:
            return value.get_name()
        except (AttributeError, NotImplementedError):
            pass

    data = getattr(value, "data", None)
    if data is not None and data is not value:
        name = _try_get_name(data)
        if name is not None:
            return name

    if hasattr(value, "unwrap_view"):
        try:
            return _try_get_name(value.unwrap_view())
        except (AttributeError, NotImplementedError):
            pass

    return None


def _exprs_equal(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
    try:
        return bool(sympy.simplify(lhs - rhs) == 0)
    except Exception:
        return lhs == rhs


def _lookup_graph_input_index(
    value: Any,
    graph_input_names: Sequence[str],
    graph_inputs: Dict[str, Any],
) -> Optional[int]:
    name = _try_get_name(value)
    if name in graph_input_names:
        return list(graph_input_names).index(name)

    expr = _maybe_sympy_expr(value)
    for idx, input_name in enumerate(graph_input_names):
        input_value = graph_inputs.get(input_name)
        if value is input_value:
            return idx
        input_expr = _maybe_sympy_expr(input_value)
        if expr is not None and input_expr is not None and _exprs_equal(expr, input_expr):
            return idx

    return None


def _inductor_value_to_source(
    value: Any,
    graph_input_names: Sequence[str],
    graph_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    index = _lookup_graph_input_index(value, graph_input_names, graph_inputs)
    if index is not None:
        return {"kind": "input", "index": index}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "list",
            "items": [
                _inductor_value_to_source(item, graph_input_names, graph_inputs)
                for item in value
            ],
        }
    return _literal_to_source(value)


def _bind_by_schema(target: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    schema = getattr(target, "_schema", None)
    if schema is None:
        return {}

    bound = {}
    args = list(args)
    for idx, arg in enumerate(schema.arguments):
        name = arg.name
        if idx < len(args):
            bound[name] = args[idx]
        elif name in kwargs:
            bound[name] = kwargs[name]
    return bound


def _build_update_plan_entry_from_bound_args(
    op_name: str,
    bound: Dict[str, Any],
    source_builder: Callable[[Any], Dict[str, Any]],
    supported_keys: set,
) -> Optional[Dict[str, Any]]:
    updates = {}
    for key, value in bound.items():
        if key in supported_keys:
            updates[key] = source_builder(value)
    if not updates:
        return None
    return {"op": op_name, "updates": updates}


def build_aclgraph_update_plan_entry_for_inductor(
    op_overload: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    graph_input_names: Sequence[str],
    graph_inputs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    op_name = getattr(op_overload, "__name__", str(op_overload))
    handler_cls = _handler_for_op(op_name)
    if handler_cls is None:
        return None

    bound = _bind_by_schema(op_overload, args, kwargs)
    if _matches_aclgraph_update_exclusion(op_name, bound):
        return None

    supported_keys = {key for _, _, key in _get_update_specs(handler_cls, op_name)}
    return _build_update_plan_entry_from_bound_args(
        op_name,
        bound,
        lambda value: _inductor_value_to_source(value, graph_input_names, graph_inputs),
        supported_keys,
    )


def should_generate_inductor_aclgraph_update_plan() -> bool:
    from torch._inductor import config
    from torch._inductor.virtualized import V

    return (
        config.triton.cudagraphs
        and config.triton.cudagraph_trees
        and getattr(V.graph, "disable_cudagraphs_reason", None) is None
    )


def append_inductor_aclgraph_update_plan_for_codegen_node(wrapper: Any, node: Any) -> None:
    if not should_generate_inductor_aclgraph_update_plan():
        return

    op_overload = getattr(node, "op_overload", None)
    if op_overload is None:
        return

    if hasattr(node, "unflatten_args"):
        args, kwargs = node.unflatten_args(node.inputs, node.constant_args)
    else:
        args = [*node.inputs, *node.constant_args]
        kwargs = getattr(node, "kwargs", {})

    entry = build_aclgraph_update_plan_entry_for_inductor(
        op_overload,
        args,
        kwargs,
        wrapper.get_graph_input_names(),
        wrapper.get_graph_inputs(),
    )
    if entry is None:
        return

    if not hasattr(wrapper, "torch_npu_aclgraph_update_plan"):
        wrapper.torch_npu_aclgraph_update_plan = []
    wrapper.torch_npu_aclgraph_update_plan.append(entry)


def emit_inductor_aclgraph_update_plan_for_wrapper(
    wrapper: Any,
    result: Any,
    is_graph_partition_subgraph: bool,
) -> None:
    from torch._inductor import config

    if not should_generate_inductor_aclgraph_update_plan():
        return

    plan = getattr(wrapper, "torch_npu_aclgraph_update_plan", [])
    if not plan:
        return

    if config.graph_partition and not is_graph_partition_subgraph:
        return

    result.writeline(f"{wrapper.launcher_fn_name}.{ACLGRAPH_UPDATE_PLAN_GLOBAL} = {plan!r}")
