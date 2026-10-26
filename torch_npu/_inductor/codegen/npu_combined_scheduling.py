from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ..autotune_process import FusedCATLASSBenchmarkRequest
from .. import config as npu_config
from ..config import log
from .catlass.catlass_scheduling import CATLASSScheduling
from .scheduling import (
    NPUNoLinearTritonScheduling,
    NPUTritonScheduling,
    _canonical_group_size,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


class NPUCombinedScheduling(CUDACombinedScheduling, TritonScheduling):
    """
    Scheduler for NPU Kernels, which delegates calls as appropriate
    to the CATLASS and Triton Schedulers, which both work for NPU devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CATLASS code,
    this would also be the place to do it.

    instead of patch is_triton() function, we make NPUCombinedScheduling extends TritonScheduling
    """

    def __init__(self, scheduler: Scheduler | None) -> None:
        BaseScheduling.__init__(self, scheduler)
        self._nolinear_triton_scheduling = NPUNoLinearTritonScheduling(scheduler)
        self._triton_scheduling = NPUTritonScheduling(scheduler)
        self._catlass_scheduling = CATLASSScheduling(scheduler)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if self._catlass_scheduling.is_catlass_template(node):
            return self._catlass_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if self._catlass_scheduling.can_fuse_vertical(node1, node2):
            return True
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        for node in (node1, node2):
            if self._catlass_scheduling.is_catlass_template(node):
                return self._catlass_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        if not npu_config.enable_welford:
            # Preserve the backend default ordering when the Welford path is
            # disabled.
            return super().get_fusion_pair_priority(node1, node2)

        from ..choices import (
            is_same_welford_output_group,
            is_welford_epilogue_fusion,
        )

        # Merge the mean/M2/count siblings first.  Their pointwise consumer can
        # then fuse vertically with the complete Welford reduction.
        if is_same_welford_output_group(node1, node2):
            return 0
        if is_welford_epilogue_fusion(node1, node2):
            return 1
        # Pairs without Welford nodes keep the baseline ordering so unrelated
        # fusion decisions stay identical to the upstream default.
        return super().get_fusion_pair_priority(node1, node2)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        if self._catlass_scheduling.is_catlass_template(template_node):
            assert not prologue_nodes  # noqa: S101
            return self._catlass_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def _regroup_for_nolinear(self, nodes: Sequence[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        snodes = list(dict.fromkeys(snode for node in nodes for snode in node.get_nodes()))
        use_canonical_group = False
        if npu_config.enable_welford:
            from ..choices import contains_welford_group

            use_canonical_group = contains_welford_group(snodes)
        for snode in snodes:
            group = self._nolinear_triton_scheduling.group_fn(snode._sizes)
            if use_canonical_group:
                group = tuple(_canonical_group_size(size) for size in group)
            snode.group = (snode.group[0], group)
        return snodes

    def _split_nolinear_nodes(self, snodes: Sequence[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        # Fusion was accepted with index groups.  Revalidate it with the
        # fallback backend after rebuilding scalar groups from merged loops.
        assert self.scheduler is not None
        snodes = self.scheduler.topological_sort_schedule(list(snodes))
        groups: list[BaseSchedulerNode] = []
        backend = self._nolinear_triton_scheduling
        for snode in snodes:
            if groups:
                previous = groups[-1]
                vertical = bool(previous.get_operation_names() & snode.ancestors)
                can_fuse = backend.can_fuse_vertical if vertical else backend.can_fuse_horizontal
                if can_fuse(previous, snode):
                    groups[-1] = FusedSchedulerNode.fuse(previous, snode)
                    continue
            groups.append(snode)
        return groups

    def _codegen_nolinear_groups(self, node: BaseSchedulerNode, groups: Sequence[BaseSchedulerNode]) -> None:
        assert self.scheduler is not None
        scheduler = self.scheduler
        # Preserve buffers needed outside the original fused node, then compute
        # last uses at the new kernel boundaries (including aliases/mutations).
        future_used_buffers = OrderedSet(
            scheduler.mutation_real_name.get(name, name)
            for name in node.used_or_aliased_buffer_names()
        ) - node.last_usage
        for group in reversed(groups):
            group.set_last_usage(future_used_buffers, scheduler.mutation_real_name)
            future_used_buffers.update(group.last_usage)
            for snode in group.get_nodes():
                scheduler.name_to_fused_node[snode.get_name()] = group

        # The outer scheduler queued all of node.last_usage before entering
        # codegen.  Each subkernel must free only its own last uses instead.
        scheduler.buffer_names_to_free.difference_update(node.last_usage)
        # Keep current_node as the original entry in scheduler.nodes: wrapper
        # memory planning uses its position in that list for allocations/frees.
        for group in groups:
            scheduler.buffer_names_to_free.update(group.last_usage)
            self._nolinear_triton_scheduling.codegen_node(group)
            scheduler.available_buffer_names.update(group.get_buffer_names())
            scheduler.completed_operations.update(group.get_operation_names())

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode):
        removed_buffers = V.graph.removed_buffers.copy()
        inplaced_to_remove = V.graph.inplaced_to_remove.copy()
        try:
            return self._triton_scheduling.codegen_node(node)
        except Exception:
            log.debug(
                "index codegen for node %s failed, falling back to no-linear codegen",
                node,
                exc_info=True,
            )
        # A failed fused kernel may have marked intermediates as kernel-local.
        # Split fallback kernels need those buffers to be materialized again.
        V.graph.removed_buffers.clear()
        V.graph.removed_buffers.update(removed_buffers)
        V.graph.inplaced_to_remove.clear()
        V.graph.inplaced_to_remove.update(inplaced_to_remove)
        snodes = self._regroup_for_nolinear([node])
        groups = self._split_nolinear_nodes(snodes)
        return self._codegen_nolinear_groups(node, groups)

    def benchmark_fused_nodes(self, nodes):
        try:
            return self._triton_scheduling.benchmark_fused_nodes(nodes)
        except Exception:
            log.debug(
                "linear benchmark_fused_nodes for nodes %s raise error, fallback to origin benchmark_fused_nodes",
                {n.get_name() for n in nodes},
                exc_info=True,
            )
        snodes = list(dict.fromkeys(snode for node in nodes for snode in node.get_nodes()))
        original_groups = [(snode, snode.group) for snode in snodes]
        try:
            self._regroup_for_nolinear(nodes)
            groups = self._split_nolinear_nodes(snodes)
            if len(groups) != 1:
                # A split kernel is not the fused candidate being benchmarked.
                return float("inf"), ""
            return self._nolinear_triton_scheduling.benchmark_fused_nodes(groups[0].get_nodes())
        finally:
            for snode, group in original_groups:
                snode.group = group

    def benchmark_codegened_module(self, module):
        if isinstance(module, FusedCATLASSBenchmarkRequest):
            return module.benchmark()
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False, hint_override=None):
        if self._catlass_scheduling.is_catlass_template(nodes[0]):
            return self._catlass_scheduling.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel, hint_override
            )
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override
        )

    def codegen_combo_kernel(self, node) -> None:
        for snode, subkernel_node in zip(node.get_nodes(), node.get_subkernel_nodes()):
            group_fn = self._nolinear_triton_scheduling.group_fn
            snode.group = (snode.group[0], group_fn(snode._sizes))
            subkernel_node.group = snode.group
        return self._triton_scheduling.codegen_combo_kernel(node)
