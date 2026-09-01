from __future__ import annotations

from typing import Any, Iterable


_WELFORD_OUTPUT_GROUP = "_torch_npu_welford_output_group"


def mark_welford_output_group(outputs: Iterable[Any]) -> None:
    """Mark the realized outputs produced by one Welford reduction."""
    group = object()
    for output in outputs:
        storage = getattr(output, "data", None)
        computed_buffer = getattr(storage, "data", None)
        if computed_buffer is not None:
            setattr(computed_buffer, _WELFORD_OUTPUT_GROUP, group)


def _welford_output_group(
    node: Any, *, allow_unmarked_epilogues: bool = False
) -> object | None:
    group = None
    for subnode in node.get_nodes():
        candidate = getattr(subnode.node, _WELFORD_OUTPUT_GROUP, None)
        if candidate is None:
            if allow_unmarked_epilogues:
                continue
            return None
        if group is not None and candidate is not group:
            return None
        group = candidate
    return group


def is_same_welford_output_group(node1: Any, node2: Any) -> bool:
    device1 = node1.get_device()
    if device1 is None or device1.type != "npu" or device1 != node2.get_device():
        return False

    group1 = _welford_output_group(node1)
    group2 = _welford_output_group(node2)
    return group1 is not None and group1 is group2


def is_welford_epilogue_fusion(producer: Any, consumer: Any) -> bool:
    if consumer.is_reduction() or producer.get_device() != consumer.get_device():
        return False
    device = producer.get_device()
    if device is None or device.type != "npu":
        return False
    if _welford_output_group(producer, allow_unmarked_epilogues=True) is None:
        return False

    produced_names = producer.get_buffer_names()
    consumer_reads = {dep.name for dep in consumer.read_writes.reads}
    return bool(produced_names & consumer_reads)


def contains_welford_group(nodes: Iterable[Any]) -> bool:
    """Return True if any scheduler node produces a Welford output group."""
    for node in nodes:
        if _welford_output_group(node, allow_unmarked_epilogues=True) is not None:
            return True
    return False


def patch_inductor_choices() -> None:
    """Keep sibling Welford outputs together despite dynamic index mismatch."""
    from torch._inductor.choices import InductorChoices

    if getattr(InductorChoices, "_torch_npu_welford_fusion_patch", False):
        return

    original_can_fuse = InductorChoices.can_fuse
    original_can_fuse_horizontal = InductorChoices.can_fuse_horizontal

    def can_fuse(scheduler, node1, node2, shared_data_score):
        if (
            is_same_welford_output_group(node1, node2)
            or is_welford_epilogue_fusion(node1, node2)
        ):
            return True
        return original_can_fuse(scheduler, node1, node2, shared_data_score)

    def can_fuse_horizontal(scheduler, node1, node2, shared_data_score):
        if is_same_welford_output_group(node1, node2):
            return True
        return original_can_fuse_horizontal(
            scheduler, node1, node2, shared_data_score
        )

    InductorChoices.can_fuse = staticmethod(can_fuse)
    InductorChoices.can_fuse_horizontal = staticmethod(can_fuse_horizontal)
    InductorChoices._torch_npu_welford_fusion_patch = True
