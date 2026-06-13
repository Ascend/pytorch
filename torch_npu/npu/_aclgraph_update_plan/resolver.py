from typing import Any, Dict, List, Sequence

from torch_npu.utils._error_code import ErrCode, pta_error


ACLGRAPH_UPDATE_PLAN_GLOBAL = "_torch_npu_aclgraph_update_plan"


def _normalize_op_name(op_name: str) -> str:
    for suffix in (".default", ".out"):
        if op_name.endswith(suffix):
            return op_name[: -len(suffix)]
    return op_name


def _op_names_compatible(expected_op: str, actual_op: str) -> bool:
    return expected_op == actual_op or _normalize_op_name(expected_op) == _normalize_op_name(actual_op)


def _get_update_specs(handler_cls: Any, op_name: str) -> List[Any]:
    get_update_specs = getattr(handler_cls, "get_update_specs", None)
    if get_update_specs is not None:
        return get_update_specs(op_name)
    return getattr(handler_cls, "UPDATE_SPECS", {}).get(op_name, [])


def _consumable_keys(record: Any) -> set:
    from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

    op_name = record.op_cache_entry.__name__
    keys = set(getattr(record, "kwargs", {}).keys())
    handler_cls = _NPU_GRAPH_OP_HANDLERS.get(op_name)
    if handler_cls is not None:
        keys.update(key for _, _, key in _get_update_specs(handler_cls, op_name))
    return keys


def resolve_aclgraph_update_plan(
    plan: Sequence[Dict[str, Any]],
    new_inputs: Sequence[Any],
) -> List[Dict[str, Any]]:
    cpu_update_input: List[Dict[str, Any]] = []
    for entry_idx, entry in enumerate(plan or []):
        updates = entry.get("updates", {})
        resolved = {}
        for key, source in updates.items():
            resolved[key] = _resolve_source(entry_idx, key, source, new_inputs)
        cpu_update_input.append(resolved)
    return cpu_update_input


def validate_aclgraph_update_plan(
    plan: Sequence[Dict[str, Any]],
    graph_dispatch_records: Sequence[Any],
) -> None:
    plan = plan or []
    if not plan and graph_dispatch_records:
        ops = [record.op_cache_entry.__name__ for record in graph_dispatch_records]
        raise RuntimeError(
            "Captured updatable ACLGraph operators but missing ACLGraph update plan: "
            f"{ops}. This may be caused by reusing cached compiled code generated "
            "with ACLGraph disabled or by using the npugraphs backend, which does "
            "not support ACLGraph update plans yet.",
            pta_error(ErrCode.PARAM),
        )

    if len(plan) != len(graph_dispatch_records):
        raise RuntimeError(
            "ACLGraph update plan length mismatch: "
            f"plan has {len(plan)} entries but graph captured "
            f"{len(graph_dispatch_records)} updatable records",
            pta_error(ErrCode.PARAM),
        )

    for idx, (entry, record) in enumerate(zip(plan, graph_dispatch_records)):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"ACLGraph update plan has invalid plan entry at index {idx}: {entry!r}",
                pta_error(ErrCode.PARAM),
            )
        expected_op = entry.get("op")
        actual_op = record.op_cache_entry.__name__
        if not isinstance(expected_op, str):
            raise RuntimeError(
                "ACLGraph update plan has invalid plan entry: "
                f"entry {idx} op must be a string, got {expected_op!r}",
                pta_error(ErrCode.PARAM),
            )
        if not _op_names_compatible(expected_op, actual_op):
            raise RuntimeError(
                "ACLGraph update plan op mismatch: "
                f"entry {idx} expects {expected_op!r}, captured {actual_op!r}",
                pta_error(ErrCode.PARAM),
            )

        updates = entry.get("updates", {})
        if not isinstance(updates, dict):
            raise RuntimeError(
                "ACLGraph update plan has invalid plan entry: "
                f"entry {idx} updates must be a dict, got {updates!r}",
                pta_error(ErrCode.PARAM),
            )
        if not updates:
            raise RuntimeError(
                "ACLGraph update plan entry has no updates: "
                f"entry {idx}, op {actual_op!r}",
                pta_error(ErrCode.PARAM),
            )

        consumable = _consumable_keys(record)
        unknown = sorted(set(updates) - consumable)
        if unknown:
            raise RuntimeError(
                "ACLGraph update plan has key(s) that captured op cannot consume: "
                f"entry {idx}, op {actual_op!r}, keys {unknown}",
                pta_error(ErrCode.PARAM),
            )

        for key, source in updates.items():
            _validate_source(idx, key, source)


def build_cpu_update_input_for_graph(
    plan: Sequence[Dict[str, Any]],
    new_inputs: Sequence[Any],
    graph_dispatch_records: Sequence[Any],
) -> List[Dict[str, Any]]:
    validate_aclgraph_update_plan(plan, graph_dispatch_records)
    return resolve_aclgraph_update_plan(plan, new_inputs)


def validate_aclgraph_update_plan_for_graph(
    plan: Sequence[Dict[str, Any]],
    graph: Any,
) -> None:
    if graph is None or not graph.auto_dispatch_capture:
        return
    validate_aclgraph_update_plan(
        plan,
        graph.graph_dispatch_mode.graph_dispatch_records,
    )


def update_aclgraph_records_for_graph(
    plan: Sequence[Dict[str, Any]],
    graph: Any,
    new_inputs: Sequence[Any],
) -> bool:
    if graph is None or not graph.auto_dispatch_capture:
        return False
    if not plan:
        return False

    graph.update(resolve_aclgraph_update_plan(plan, new_inputs))
    return True


def _resolve_source(
    entry_idx: int,
    key: str,
    source: Dict[str, Any],
    new_inputs: Sequence[Any],
) -> Any:
    if not isinstance(source, dict):
        raise RuntimeError(
            f"ACLGraph update plan entry {entry_idx} key {key} "
            f"has invalid source {source!r}",
            pta_error(ErrCode.PARAM),
        )
    kind = source.get("kind")
    if kind == "input":
        index = source.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(new_inputs):
            raise RuntimeError(
                f"ACLGraph update plan entry {entry_idx} key {key} "
                f"input index {index} is out of range for {len(new_inputs)} inputs",
                pta_error(ErrCode.PARAM),
            )
        return new_inputs[index]
    if kind == "constant":
        return source.get("value")
    if kind == "none":
        return None
    if kind == "list":
        items = source.get("items")
        if not isinstance(items, list):
            raise RuntimeError(
                f"ACLGraph update plan entry {entry_idx} key {key} "
                f"has invalid list source items {items!r}",
                pta_error(ErrCode.PARAM),
            )
        return [_resolve_source(entry_idx, key, item, new_inputs) for item in items]
    raise RuntimeError(
        f"ACLGraph update plan entry {entry_idx} key {key} "
        f"has unsupported source kind {kind!r}",
        pta_error(ErrCode.PARAM),
    )


def _validate_literal_constant(value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        raise RuntimeError(
            "Unsupported ACLGraph update Tensor constant; Tensor update values must be graph inputs",
            pta_error(ErrCode.PARAM),
        )
    if value is None or isinstance(value, (int, float, bool, str)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_literal_constant(item)
        return
    raise RuntimeError(
        f"Unsupported ACLGraph update constant: {value!r}",
        pta_error(ErrCode.PARAM),
    )


def _validate_source(entry_idx: int, key: str, source: Dict[str, Any]) -> None:
    if not isinstance(source, dict):
        raise RuntimeError(
            f"ACLGraph update plan entry {entry_idx} key {key} "
            f"has invalid source {source!r}",
            pta_error(ErrCode.PARAM),
        )
    kind = source.get("kind")
    if kind == "input":
        index = source.get("index")
        if not isinstance(index, int) or index < 0:
            raise RuntimeError(
                f"ACLGraph update plan entry {entry_idx} key {key} "
                f"has invalid input index {index}",
                pta_error(ErrCode.PARAM),
            )
    elif kind == "constant":
        try:
            _validate_literal_constant(source.get("value"))
        except RuntimeError as exc:
            raise RuntimeError(
                f"ACLGraph update plan entry {entry_idx} key {key} "
                f"has unsupported constant value {source.get('value')!r}",
                pta_error(ErrCode.PARAM),
            ) from exc
    elif kind == "none":
        return
    elif kind == "list":
        items = source.get("items")
        if not isinstance(items, list):
            raise RuntimeError(
                f"ACLGraph update plan entry {entry_idx} key {key} "
                f"has invalid list source items {items!r}",
                pta_error(ErrCode.PARAM),
            )
        for item in items:
            _validate_source(entry_idx, key, item)
    else:
        raise RuntimeError(
            f"ACLGraph update plan entry {entry_idx} key {key} "
            f"has unsupported source kind {kind!r}",
            pta_error(ErrCode.PARAM),
        )
