from __future__ import annotations

from typing import Sequence, Union, TYPE_CHECKING

import torch
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from ..autotune_process import FusedCATLASSBenchmarkRequest
from .catlass.catlass_scheduling import CATLASSScheduling
from .scheduling import NPUTritonScheduling

if TYPE_CHECKING:
    from torch._inductor.common import BackendFeature
    from torch.utils._ordered_set import OrderedSet


class NPUCombinedScheduling(BaseScheduling):
    """
    Scheduler for NPU Kernels, which delegates calls as appropriate
    to the CATLASS and Triton Schedulers, which both work for NPU devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CATLASS code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__(scheduler)
        self._scheduler = scheduler
        self._triton_scheduling = NPUTritonScheduling(scheduler)
        self._catlass_scheduling = CATLASSScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return self._triton_scheduling.get_backend_features(device)

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

    def group_fn(self, sizes):
        return self._triton_scheduling.group_fn(sizes)

    def fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> FusedSchedulerNode:
        """
        Fuse two nodes
        """
        return super().fuse(node1, node2)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        if self._catlass_scheduling.is_catlass_template(template_node):
            assert not prologue_nodes
            return self._catlass_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]):
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self):
        return self._triton_scheduling.codegen_sync()

    def flush(self):
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args, **kwargs):
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(self, nodes):
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        if isinstance(module, FusedCATLASSBenchmarkRequest):
            return module.benchmark()
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
        if self._catlass_scheduling.is_catlass_template(nodes[0]):
            return self._catlass_scheduling.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel
            )
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel
        )

    def benchmark_combo_kernel(self, node_list):
        return self._triton_scheduling.benchmark_combo_kernel(node_list)
