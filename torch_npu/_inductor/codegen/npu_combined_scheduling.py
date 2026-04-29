from __future__ import annotations
from typing import Sequence, Union, TYPE_CHECKING, Optional

from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from .catlass.catlass_scheduling import CATLASSScheduling
from .scheduling import NPUTritonScheduling, NPUNoLinearTritonScheduling
from ..autotune_process import FusedCATLASSBenchmarkRequest
from ..config import log, is_ascend950

if TYPE_CHECKING:
    from torch._inductor.common import BackendFeature
    from torch.utils._ordered_set import OrderedSet


class NPUCombinedScheduling(CUDACombinedScheduling):
    """
    Scheduler for NPU Kernels, which delegates calls as appropriate
    to the CATLASS and Triton Schedulers, which both work for NPU devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CATLASS code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
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

    def node_can_linear(self, node: Union[FusedSchedulerNode, SchedulerNode]):
        # user config use linear scheduling for ascend.
        nodes = node.get_nodes()
        cat_kernel = False
        for n in nodes:
            for body_index in n._body.indexing_exprs.values():
                if 'cat_store' in str(body_index):
                    cat_kernel = True
        if cat_kernel:
            return True

        from ..config import inductor_ascend_linear_mode
        if inductor_ascend_linear_mode != "linear":
            return False
        return True

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]):
        if not is_ascend950:
            return self._triton_scheduling.codegen_node(node)

        if self.node_can_linear(node):
            try:
                return self._triton_scheduling.codegen_node(node)
            except Exception as e:
                log.exception(f"linear codegen for node {node} raise error: {e}, fallback to origin codegen")
        # regroup snode
        for snode in node.get_nodes():
            group_fn = self._nolinear_triton_scheduling.group_fn
            snode.group = (snode.group[0], group_fn(snode._sizes))
        return self._nolinear_triton_scheduling.codegen_node(node)

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
