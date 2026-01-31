import itertools
import logging
from typing import List, Optional, Sequence, Set, cast

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import code_hash, get_path
from torch._inductor.codegen.common import BackendFeature, IndentedBuffer
from torch._inductor.ir import Buffer, ComputedBuffer, Pointwise
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    MemoryDep,
    Scheduler,
    SchedulerNode,
    WhyNoFuse,
)
from torch._inductor.utils import (
    Placeholder,
    get_fused_kernel_name,
    get_kernel_metadata,
    sympy_product,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_npu._inductor.codegen.catlass.catlass_python_evg import (
    CatlassEVGCodegen,
    MockCatlassHandler,
)

from ...autotune_process import FusedCATLASSBenchmarkRequest
from ...config import catlass as catlass_config
from .catlass_kernel import CATLASSTemplateBuffer

log = logging.getLogger("torch._inductor")


class WhyNoFuseNames(WhyNoFuse):
    def __init__(self, name1: str, name2: str) -> None:
        self.name1 = name1
        self.name2 = name2

    # re-write this func since it's not compatible with torch v2.7.1
    def __str__(self) -> str:
        return f"cannot fuse {self.name1} with {self.name2}: " + (
            self.reason % self.args
        )


class CATLASSScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for CATLASS Template Kernels.
    This class is intended to be used in combination with NPUTritonScheduling,
    and delegated to by NPUCombinedScheduling.

    It handles fusion decisions and CATLASS specific template code generation.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_catlass_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CATLASSTemplateBuffer
        )

    def is_catlass_fused_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, FusedSchedulerNode) and self.is_catlass_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_catlass_template(node1) and isinstance(node2, BaseSchedulerNode):
            assert node1.node, "node1.node should not be None"
            return self._can_fuse_epilogue_impl(
                cast(CATLASSTemplateBuffer, node1.node),
                [],
                node2,  # type: ignore[arg-type]
            )
        elif self.is_catlass_fused_template(node1) and isinstance(
            node2, BaseSchedulerNode
        ):
            assert node1.node, "node1.node should not be None"
            assert node2.node, "node2.node should not be None"
            fnode1 = cast(FusedSchedulerNode, node1)
            return self._can_fuse_epilogue_impl(
                fnode1.get_template_node(),  # type: ignore[arg-type]
                self._unwrap_epilogue_nodes(fnode1),
                node2,  # type: ignore[arg-type]
            )

        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["catlass", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.catlass(r'''")
            compile_wrapper.splice(src_code, strip=True)
            is_mix = False
            for node in node_schedule:
                if isinstance(node, SchedulerNode) and isinstance(
                    node.get_template_node(), CATLASSTemplateBuffer
                ):
                    is_mix = node.get_template_node().is_mix
                    break
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)}, is_mix={str(is_mix)})"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        only_src_code=False,
    ):
        """
        Codegen a CATLASS template, possibly with fused epilogues
        """
        counters["inductor"]["catlass_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_catlass_template(
            template_node
        ), "Template node passed to CATLASSScheduler.codegen_template must be a SchedulerNode that wraps a CATLASSTemplateBuffer"
        template_node = cast(SchedulerNode, template_node)
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        ctb: CATLASSTemplateBuffer = cast(CATLASSTemplateBuffer, template_node.node)
        epilogue_ir_nodes: List[Buffer] = [n.node for n in epilogue_nodes]  # type: ignore[misc]
        assert all(
            isinstance(n, ComputedBuffer) for n in epilogue_ir_nodes
        ), "Epilogue nodes must all be instances of ir.ComputedBuffer"
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()

            # typically there is a codegen pass which runs after mark_run
            # for this kernel we've already generated the C++ code, but we still
            # need to let the kernel know about loads/stores that occur in the fused
            # kernel for memory planning to properly optimize allocations
            ctb.emulate_store_fn()
            for node in epilogue_ir_nodes:
                with V.set_ops_handler(MockCatlassHandler(V.get_ops_handler())):
                    assert isinstance(
                        node, ComputedBuffer
                    )  # Not sure why we need to do this again
                    node.get_store_function()(CatlassEVGCodegen.get_index_vars(node))

        with V.set_kernel_handler(kernel):
            src_code = render()
            if not only_src_code:
                node_schedule = [template_node, *epilogue_nodes]
                kernel_name = self.define_kernel(src_code, node_schedule)

        # debug printing values of intermediate tensors
        if not only_src_code:
            _, call_args, arg_signatures, _ = kernel.args.python_argdefs()
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args, kernel_name, arg_signatures, kernel
            )
            with debug_printer_manager:
                kernel.call_kernel(kernel_name, ctb)

            V.graph.removed_buffers |= kernel.removed_buffers
            self.free_buffers_in_scheduler()

        size_args = V.graph.sizevars.size_hints(kernel.get_layout_args())
        return src_code, size_args

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
        template_node = nodes[0]
        epilogue_nodes = nodes[1:]
        src_code, extra_args = self.codegen_template(
            template_node, epilogue_nodes, [], only_src_code=True
        )
        kernel_name = "catlass_fused_tmp"
        src_code = src_code.replace("KERNEL_NAME", kernel_name)
        if benchmark_kernel:
            return FusedCATLASSBenchmarkRequest(
                kernel_name, src_code, template_node, epilogue_nodes, extra_args
            )
        else:
            return src_code

    @staticmethod
    def _unwrap_epilogue_nodes(
        fused_node: FusedSchedulerNode,
    ) -> List[BaseSchedulerNode]:
        nodes = fused_node.get_nodes()
        template_node = fused_node.get_template_node()
        assert all(
            n.node is not None for n in nodes
        ), "All epilogue nodes should have an IRNode"
        return cast(
            list[BaseSchedulerNode], [n for n in nodes if n.node is not template_node]
        )

    def _can_fuse_epilogue_impl(
        self,
        catlass_template_buffer: CATLASSTemplateBuffer,
        existing_epilogue_nodes: List[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        """
        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            catlass_template_buffer: A CATLASSTemplateBuffer object representing the CATLASS template and its result buffer
            existing_epilogue_nodes: List[SchedulerNode]: The list of already fused epilogue nodes.
            node_to_fuse: The SchedulerNode node to be checked if it can be fused with the epilogue.
        Returns:
            - bool: True if the given node can be fused with the epilogue, False otherwise.
        """

        why = WhyNoFuseNames(catlass_template_buffer.get_name(), node_to_fuse.get_name())
        scheduler_nodes_to_fuse = node_to_fuse.get_nodes()

        assert isinstance(catlass_template_buffer, CATLASSTemplateBuffer)
        assert (
            len(existing_epilogue_nodes) == 0
        )  # not support chain-based epilogue fusion yet

        if isinstance(node_to_fuse, FusedSchedulerNode):
            return False

        # Checks on constituent nodes
        for s_node in scheduler_nodes_to_fuse:
            node = s_node.node

            if not isinstance(node, ComputedBuffer):
                why(f"{node} is not a ComputedBuffer")
                return False
            elif not isinstance(node.data, Pointwise):
                why(f"{node} is not a PointWise op")
                return False
            elif not node.get_computed_buffer_name():  # type: ignore[attr-defined]
                why(f"{node} does not have a computed buffer name")
                return False

            name = node.get_computed_buffer_name()  # type: ignore[attr-defined]
            # dtype can differ, and strides can differ as long as they are broadcastable
            if node.get_size() != catlass_template_buffer.get_size():
                why(
                    f"{name}'s size: {node.get_size()} differs from {catlass_template_buffer.get_name()}'s \
                        size: {catlass_template_buffer.get_size()}"
                )
                return False

        assert len(
            existing_epilogue_nodes
        ) or catlass_template_buffer.get_name() in OrderedSet(
            [rd.name for rd in node_to_fuse.read_writes.reads]
        ), "First epilogue node must read from npu template buffer"

        if node_to_fuse.has_aliasing_or_mutation():
            why(f"{node_to_fuse.get_name()} has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction():
            why(f"{node_to_fuse.get_name()} is a reduction which is not yet supported")
            return False
        elif (
            not catlass_config.catlass_epilogue_fusion_enable
            or not config.epilogue_fusion
        ):
            why("CATLASS epilogue fusion is not enabled")
            return False
        elif catlass_template_buffer.epilogue_fusion_type == 0:
            why("epilogue fusion is not supported on current gemm ops")
            return False

        try:
            CatlassEVGCodegen.ir_to_evg_python_code(
                catlass_template_buffer.get_name(),
                existing_epilogue_nodes + list(node_to_fuse.get_nodes()),
                OrderedSet(),
            )
        except NotImplementedError as e:
            why(
                f"Cannot fuse epilogue node {node_to_fuse} into {catlass_template_buffer.name}"
            )
            return False

        try:
            from .gemm_template import CATLASS1xGemmTemplate

            CATLASS1xGemmTemplate._try_fast_fusion(
                existing_epilogue_nodes + list(node_to_fuse.get_nodes()),
                catlass_template_buffer.get_name(),
            )
        except NotImplementedError as e:
            # so we need to check again if current gemm kind supports EVG
            if catlass_template_buffer.epilogue_fusion_type != 2:
                why("Current CATLASS Gemm Template does not support EVG fusion")
                return False

            for s_node in scheduler_nodes_to_fuse:
                node = s_node.node
                if node.get_dtype() == torch.bfloat16:
                    why("CATLASS EVG does not support bfloat16")
                    return False
        return True
