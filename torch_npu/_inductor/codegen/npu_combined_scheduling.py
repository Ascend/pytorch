from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy

from torch._inductor import metrics
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from ..autotune_process import FusedCATLASSBenchmarkRequest
from .. import config as npu_config
from ..config import log
from .catlass.catlass_scheduling import CATLASSScheduling
from .npu_kernel_features import NumelList
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

    def node_can_linear(self):
        # user config use linear scheduling for ascend.
        from ..config import inductor_ascend_linear_mode

        if inductor_ascend_linear_mode != "linear":
            return False
        return True

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode):
        nodes = node.get_nodes()
        use_linear = self.node_can_linear()
        if use_linear and npu_config.enable_welford:
            from ..choices import contains_welford_group

            _, (_, reduction_numel) = max(
                nodes, key=lambda candidate: int(candidate.is_reduction())
            ).group
            if isinstance(reduction_numel, NumelList):
                reduction_numel = reduction_numel.numels()
            use_linear = not (
                contains_welford_group(nodes)
                and not isinstance(reduction_numel, (int, sympy.Integer))
            )

        if use_linear:
            kernel_count = metrics.generated_kernel_count
            try:
                return self._triton_scheduling.codegen_node(node)
            except Exception:
                metrics.generated_kernel_count = kernel_count
                log.debug(
                    "linear codegen for node %s raise error, fallback to origin codegen",
                    node,
                    exc_info=True,
                )
        # regroup snode
        use_canonical_group = False
        if npu_config.enable_welford:
            from ..choices import contains_welford_group

            # Canonicalize group sizes only for kernels containing Welford
            # reductions; unrelated kernels keep the raw (baseline) groups.
            use_canonical_group = contains_welford_group(nodes)
        for snode in nodes:
            group_fn = self._nolinear_triton_scheduling.group_fn
            if use_canonical_group:
                snode.group = (
                    snode.group[0],
                    tuple(_canonical_group_size(size) for size in group_fn(snode._sizes)),
                )
            else:
                snode.group = (snode.group[0], group_fn(snode._sizes))
        return self._nolinear_triton_scheduling.codegen_node(node)

    def benchmark_fused_nodes(self, nodes):
        if self.node_can_linear():
            try:
                return self._triton_scheduling.benchmark_fused_nodes(nodes)
            except Exception:
                log.debug(
                    "linear benchmark_fused_nodes for nodes %s raise error, fallback to origin benchmark_fused_nodes",
                    {n.get_name() for n in nodes},
                    exc_info=True,
                )
        # regroup snode
        for node in nodes:
            for snode in node.get_nodes():
                group_fn = self._nolinear_triton_scheduling.group_fn
                snode.group = (snode.group[0], group_fn(snode._sizes))
        return self._nolinear_triton_scheduling.benchmark_fused_nodes(nodes)

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
        subkernel_nodes = node.get_subkernel_nodes()
        if self.node_can_linear():
            kernel_count = metrics.generated_kernel_count
            original_groups = [
                (snode, snode.group)
                for subkernel_node in subkernel_nodes
                for snode in subkernel_node.get_nodes()
            ]
            original_groups.extend(
                (subkernel_node, subkernel_node.group)
                for subkernel_node in subkernel_nodes
            )
            try:
                group_fn = self._nolinear_triton_scheduling.group_fn
                for subkernel_node in subkernel_nodes:
                    # get_nodes() is flattened, whereas get_subkernel_nodes()
                    # contains fused groups. Zipping them assigns later groups
                    # the dimensions of an unrelated node as soon as one
                    # subkernel has multiple nodes.
                    subnodes = subkernel_node.get_nodes()
                    for snode in subnodes:
                        snode.group = (snode.group[0], group_fn(snode._sizes))
                    subkernel_node.group = max(
                        subnodes, key=lambda snode: int(snode.is_reduction())
                    ).group
                return self._triton_scheduling.codegen_combo_kernel(node)
            except Exception:
                metrics.generated_kernel_count = kernel_count
                for snode, group in original_groups:
                    snode.group = group
                log.debug(
                    "linear combo codegen for node %s raise error, "
                    "fallback to individual kernels",
                    node,
                    exc_info=True,
                )

        # Ordinary codegen retains its own linear-to-nonlinear fallback.
        # The scheduler queued last_usage for the whole combo. Do not let the
        # first ordinary kernel free inputs still needed by later subkernels.
        pending_frees = self.scheduler.buffer_names_to_free.copy()
        self.scheduler.buffer_names_to_free.clear()
        try:
            for subkernel_node in subkernel_nodes:
                self.codegen_node(subkernel_node)
        finally:
            self.scheduler.buffer_names_to_free.update(pending_frees)
        self.free_buffers_in_scheduler()
