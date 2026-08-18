"""DVM-aware fusion region estimation for NPU overlap scheduling.

This module predicts DVM graph-fusion regions without mutating the FX graph.
It is intentionally conservative: only pure lightweight regions receive fused
costs, while regions containing heavy compute keep the normal benchmark runtime
path.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

aten = torch.ops.aten

_DVM_HEAVY_OPS: set[object] = {
    aten.mm.default,
    aten.bmm.default,
    aten.addmm.default,
    aten.sum.dim_IntList,
    aten.sum.default,
    aten.amax.default,
    aten.amin.default,
}


def _iter_tensor_values(value: object) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    def add_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensors.append(tensor)
        return tensor

    torch.utils._pytree.tree_map_only(torch.Tensor, add_tensor, value)
    return tensors


def _get_tensor_nbytes(tensor: torch.Tensor) -> int | None:
    try:
        numel = 1
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                if not dim.node.has_hint():
                    return None
                dim = dim.node.hint
            numel *= int(dim)
        return numel * tensor.dtype.itemsize
    except Exception:
        return None


def _get_npu_dram_gb_per_s() -> float:
    env_bw = os.environ.get("TORCH_NPU_DVM_FUSION_REGION_BW_GBPS")
    if env_bw is not None:
        try:
            bw = float(env_bw)
            if bw > 0:
                return bw
        except ValueError:
            log.warning(
                "Ignoring invalid TORCH_NPU_DVM_FUSION_REGION_BW_GBPS=%r",
                env_bw,
            )

    # Ascend910B3 does not expose HBM bandwidth through
    # torch.npu.get_device_properties(), so use the local NPU roofline value.
    return 2000.0


def _trace_diagnostics_enabled() -> bool:
    return os.environ.get("TORCH_NPU_DVM_FUSION_REGION_TRACE", "0") == "1"


def is_view_node(node: fx.Node) -> bool:
    return isinstance(node.target, torch._ops.OpOverload) and (
        node.target.is_view and node.target.namespace in ("aten", "prims")
    )


def _get_dvm_op_registry() -> dict[object, object]:
    try:
        from torch_npu._inductor.dvm.op_emitter import DVM_OP_REGISTRY

        return DVM_OP_REGISTRY
    except Exception:
        return {}


def _is_dvm_lightweight_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    ):
        return False
    if node.target in _DVM_HEAVY_OPS:
        return False

    # Read the patched upstream predicate lazily so NPU-specific compute ops
    # remain excluded from lightweight DVM regions.
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    if is_compute_node(node):
        return False

    registry = _get_dvm_op_registry()
    if node.target not in registry:
        return False
    _, rule = registry[node.target]
    return bool(rule(node))


def _single_tensor_meta(node: fx.Node) -> torch.Tensor | None:
    val = node.meta.get("val")
    tensors = _iter_tensor_values(val) if val is not None else []
    if len(tensors) != 1:
        return None
    return tensors[0]


def _shape_with_hints(tensor: torch.Tensor) -> tuple[int, ...] | None:
    dims: list[int] = []
    try:
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                if not dim.node.has_hint():
                    return None
                dim = dim.node.hint
            dims.append(int(dim))
    except Exception:
        return None
    return tuple(dims)


def _is_broadcastable_to(
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
) -> bool:
    src_reversed = list(reversed(src_shape))
    dst_reversed = list(reversed(dst_shape))
    for idx, dst_dim in enumerate(dst_reversed):
        src_dim = src_reversed[idx] if idx < len(src_reversed) else 1
        if src_dim != 1 and src_dim != dst_dim:
            return False
    return True


def _dvm_can_fuse_fx_edge(producer: fx.Node, consumer: fx.Node) -> bool:
    if is_view_node(producer) or is_view_node(consumer):
        return True

    producer_val = _single_tensor_meta(producer)
    consumer_val = _single_tensor_meta(consumer)
    if producer_val is None or consumer_val is None:
        return False

    producer_shape = _shape_with_hints(producer_val)
    consumer_shape = _shape_with_hints(consumer_val)
    if producer_shape is None or consumer_shape is None:
        return False

    return producer_shape == consumer_shape or _is_broadcastable_to(
        producer_shape,
        consumer_shape,
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[fx.Node, fx.Node] = {}
        self.rank: dict[fx.Node, int] = {}

    def find(self, node: fx.Node) -> fx.Node:
        parent = self.parent.get(node, node)
        if parent != node:
            parent = self.find(parent)
        self.parent[node] = parent
        return parent

    def union(self, lhs: fx.Node, rhs: fx.Node) -> None:
        lhs_root = self.find(lhs)
        rhs_root = self.find(rhs)
        if lhs_root == rhs_root:
            return
        lhs_rank = self.rank.get(lhs_root, 0)
        rhs_rank = self.rank.get(rhs_root, 0)
        if lhs_rank < rhs_rank:
            lhs_root, rhs_root = rhs_root, lhs_root
            lhs_rank, rhs_rank = rhs_rank, lhs_rank
        self.parent[rhs_root] = lhs_root
        if lhs_rank == rhs_rank:
            self.rank[lhs_root] = lhs_rank + 1


def _split_connected_components(nodes: list[fx.Node]) -> list[list[fx.Node]]:
    node_set = OrderedSet(nodes)
    union_find = _UnionFind()
    for node in nodes:
        union_find.find(node)

    for node in nodes:
        for inp in node.all_input_nodes:
            if inp in node_set and _dvm_can_fuse_fx_edge(inp, node):
                union_find.union(inp, node)

    groups: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    for node in nodes:
        groups[union_find.find(node)].append(node)
    return list(groups.values())


def build_dvm_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Predict lightweight DVM fusion regions without changing the graph."""

    node_to_idx = {node: idx for idx, node in enumerate(gm.graph.nodes)}
    region_of: dict[fx.Node, OrderedSet[fx.Node]] = {}
    current_span: list[fx.Node] = []

    def flush_span() -> None:
        nonlocal current_span
        if len(current_span) < 2:
            current_span = []
            return
        for component in _split_connected_components(current_span):
            non_view_count = sum(1 for node in component if not is_view_node(node))
            if non_view_count < 2:
                continue
            sorted_component = sorted(component, key=lambda node: node_to_idx[node])
            node_set = OrderedSet(sorted_component)
            for node in sorted_component:
                region_of[node] = node_set
        current_span = []

    for node in gm.graph.nodes:
        if _is_dvm_lightweight_node(node):
            current_span.append(node)
        else:
            flush_span()
    flush_span()
    return region_of


def _estimate_boundary_transfer_ms(
    external_inputs: list[object],
    external_outputs: list[object],
) -> float | None:
    bytes_count = 0
    for value in (*external_inputs, *external_outputs):
        for tensor in _iter_tensor_values(value):
            nbytes = _get_tensor_nbytes(tensor)
            if nbytes is None:
                return None
            bytes_count += nbytes

    if bytes_count == 0:
        return 0.0

    bw_bytes_per_ms = _get_npu_dram_gb_per_s() * 1024**3 / 1e3
    if not math.isfinite(bw_bytes_per_ms) or bw_bytes_per_ms <= 0:
        return None
    return bytes_count / bw_bytes_per_ms


def estimate_dvm_fused_node_costs(
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, float]:
    """Estimate per-node costs for predicted lightweight DVM fusion regions."""

    costs: dict[fx.Node, float] = {}
    seen: OrderedSet[int] = OrderedSet()
    trace_diagnostics = _trace_diagnostics_enabled()
    diagnostics: list[dict[str, object]] = []

    for node_set in region_of.values():
        region_id = id(node_set)
        if region_id in seen:
            continue
        seen.add(region_id)

        region_nodes = list(node_set)
        applied_nodes = []
        for node in region_nodes:
            external_inputs = [
                inp.meta.get("val")
                for inp in node.all_input_nodes
                if inp not in node_set and inp.meta.get("val") is not None
            ]
            has_external_user = any(user not in node_set for user in node.users)
            external_outputs = []
            if has_external_user and node.meta.get("val") is not None:
                external_outputs.append(node.meta["val"])

            if not external_inputs and not external_outputs:
                cost = 0.0
            else:
                estimated = _estimate_boundary_transfer_ms(
                    external_inputs,
                    external_outputs,
                )
                if estimated is None:
                    if trace_diagnostics:
                        boundary_tensors = [
                            tensor
                            for value in (*external_inputs, *external_outputs)
                            for tensor in _iter_tensor_values(value)
                        ]
                        log.info(
                            "Skip DVM fusion-region estimation: node=%s, "
                            "target=%s, boundary_tensors=%s",
                            node.name,
                            node.target,
                            [
                                {
                                    "shape": tuple(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                }
                                for tensor in boundary_tensors
                            ],
                        )
                    continue
                cost = estimated
            costs[node] = cost
            applied_nodes.append(node.name)

        if trace_diagnostics and applied_nodes:
            diagnostics.append(
                {
                    "nodes": [node.name for node in region_nodes],
                    "targets": [str(node.target) for node in region_nodes],
                    "estimated_nodes": applied_nodes,
                }
            )

    if diagnostics:
        from torch._logging import trace_structured

        log.info(
            "NPU DVM fusion region predictor estimated %d lightweight regions "
            "and %d nodes",
            len(diagnostics),
            len(costs),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "npu_dvm_fusion_region_estimations",
                "encoding": "json",
            },
            payload_fn=lambda: diagnostics,
        )
    return costs
