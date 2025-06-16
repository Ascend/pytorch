import collections
import contextlib
import contextlib
import itertools
import itertools
import os
from typing import Dict, Sequence, List, Iterable
from typing import List, Union, Any
from typing import Union, Iterable
import sympy
from torch._dynamo.utils import counters
from torch._dynamo.utils import counters
from torch._inductor import scheduler, metrics
from torch._inductor.codecache import code_hash
from torch._inductor.codecache import code_hash
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._inductor.codegen.simd import DisableReduction, EnableReduction
from torch._inductor.codegen.simd import DisableReduction, EnableReduction, SIMDKernelFeatures, SIMDKernel
from torch._inductor.codegen.simd import schedule_log, scheduler
from torch._inductor.codegen.triton import (TritonScheduling, log, config)
from torch._inductor.codegen.triton import (
    TritonScheduling,
    config,
    schedule_log,
    get_fused_kernel_name,
    get_kernel_category_by_source_code,
    Placeholder,
    get_kernel_metadata,
    get_path,
    IndentedBuffer
)
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.utils import sympy_index_symbol, ModularIndexing, FloorDiv
from torch._inductor.virtualized import (V, )
from torch._inductor.virtualized import (
    V,
)
from torch.fx.immutable_collections import immutable_dict
from torch.fx.immutable_collections import immutable_dict
from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel, flatten

from .kernel_analysis import ReductionAnalysis
from .npu_kernel_features import NumelList, NPUKernelFeatures
from .split_tiling import SplitTiling
from .split_tiling import SplitTiling
from .triton import NPUIndexTritonKernel
from .. import config as npu_config
from ..lowering_fx import (
    create_fx_from_snodes_by_traced_graph,
    create_compile_kwargs,
    generate_fx_graph_code,
    dump_fx_graph_code
)

from ..config import log


def flatten_groups(nums):
    res = []
    for i in nums:
        if isinstance(i, Iterable):
            for x in i:
                res.append(x)
        else:
            res.append(i)
    return res


@classmethod
def create_tiling(
        cls, pw_tiling: Sequence[sympy.Expr], reduction_tiling: Sequence[sympy.Expr]
) -> Dict[str, sympy.Expr]:
    """
    Create a tiling dict from pointwise and reduction splits.
    """

    pw_tiling = flatten_groups(pw_tiling)
    pw_prefixes = ["w", "v", "t", "z", "y", "x"][-len(pw_tiling):]
    reduction_tiling = flatten_groups(reduction_tiling)
    reduction_tiling = [NumelList(reduction_tiling).numels()]
    reduction_prefixes = ["r"][: len(reduction_tiling)]
    tiling = immutable_dict(
        list(zip(pw_prefixes, pw_tiling))
        + list(zip(reduction_prefixes, reduction_tiling)))
    return tiling


class NPUTritonScheduling(TritonScheduling):
    def __init__(self, input_scheduler):
        super().__init__(input_scheduler)
        self.kernel_type = NPUIndexTritonKernel

    def create_kernel_choices(
            self, kernel_features: SIMDKernelFeatures, kernel_args, kernel_kwargs
    ) -> List[SIMDKernel]:

        return [
            self.kernel_type(
                *kernel_args,
                **kernel_kwargs,
            )
        ]

    # transform indexing before call codegen_node_schedule_with_kernel
    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures, nodes):
        node_schedule = kernel_features.node_schedule
        tiling = self.select_tiling(
            node_schedule, kernel_features.numel, kernel_features.reduction_numel
        )

        kernels = self.create_kernel_choices(
            kernel_features, [tiling], {"features": kernel_features}
        )
        kernel = kernels[0]
        setattr(kernel, "node_schedule", node_schedule)
        self.decide_codegen_dims_in_kernel(node_schedule, kernel)

        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        MultiKernel.merge_workspaces_inplace(kernels)
        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()

            V.graph.removed_buffers |= kernel.removed_buffers
            V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

            traced_graph_hash = None
            if npu_config.dump_fx_graph:
                if not npu_config.traced_fx_graph_cache:
                    npu_config.traced_fx_graph_cache = os.path.join(os.getenv("TORCHINDUCTOR_CACHE_DIR"),
                                                                    'traced_fx_graph_cache')
                os.makedirs(npu_config.traced_fx_graph_cache, exist_ok=True)
                traced_graph, fx_call_args, fx_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(nodes)
                if traced_graph is None:
                    log.warning(f"For nodes {nodes}, could not gen fx graph while dump-graph.")
                else:
                    traced_graph_hash = code_hash(traced_graph.print_readable(print_output=False))

            kernel_name, src_code = self.define_kernel(src_code, node_schedule, kernel, traced_graph_hash)

            kernel.kernel_name = kernel_name
            kernel.code_hash = code_hash(src_code)
        del kernel

        final_kernel: Union[SIMDKernel, MultiKernel]
        if len(kernels) > 1:
            final_kernel = MultiKernel(kernels)
        else:
            (final_kernel,) = kernels

        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(final_kernel.kernel_name)

        if npu_config.dump_fx_graph and traced_graph is not None:
            new_compile_kwargs = create_compile_kwargs(final_kernel, fx_call_args, fx_args)
            if new_compile_kwargs:
                compile_kwargs |= new_compile_kwargs
                fx_dump_path = os.path.join(npu_config.traced_fx_graph_cache, traced_graph_hash)
                os.makedirs(fx_dump_path, exist_ok=True)
                fx_code = generate_fx_graph_code(traced_graph.code, src_code, kernel_name, compile_kwargs)
                dump_fx_graph_code(fx_code, fx_dump_path, traced_graph_hash)

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernels[0].kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
                V.graph.wrapper_code.supports_intermediate_hooks
                and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernels[0].args.live_output_buffers()
            for node in kernel_features.scheduler_nodes():
                name = node.get_name()
                if name not in live_outs:
                    continue
                if node.node is None:
                    raise RuntimeError("assert node.node is not None")

                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def define_kernel(self, src_code, node_schedule, kernel, traced_graph_hash: str):
        wrapper = V.graph.wrapper_code
        if (src_code, traced_graph_hash) in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[(src_code, traced_graph_hash)]
            if npu_config.dump_fx_graph:
                src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
                subs_name = kernel_name if config.triton.unique_kernel_names else "triton_"
                src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)
                if traced_graph_hash:
                    src_code = src_code.replace('TRACED_GRAPH_HASH', traced_graph_hash)
                    src_code = src_code.replace('TRACED_GRAPH_DIR', npu_config.traced_fx_graph_cache)
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = "_".join(
                ["triton", kernel_category, fused_name, wrapper.next_kernel_suffix()]
            )
            # use the original src_code as the key
            wrapper.src_to_kernel[(src_code, traced_graph_hash)] = kernel_name
            subs_name = kernel_name if config.triton.unique_kernel_names else "triton_"

            # DESCRIPTIVE_NAME is used for profiling purposes; it shows the full kernel name
            # even when unique_kernel_names is turned off. Meanwhile, KERNEL_NAME is sometimes set
            # to "triton_" to maximize caching opportunities (when unique_kernel_names = False).
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)
            if traced_graph_hash:
                src_code = src_code.replace('TRACED_GRAPH_HASH', traced_graph_hash)
                src_code = src_code.replace('TRACED_GRAPH_DIR', npu_config.traced_fx_graph_cache)

            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "#")

            basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            current_device = V.graph.get_current_device_or_throw()
            compile_wrapper.writeline(f"''', device_str='{current_device.type}')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            # Extra debug message for npu.
            snode_str = ""
            snodes = [node for node in node_schedule if node not in (DisableReduction, EnableReduction)]
            snode_str = f"\n# SchedulerNodes: {snodes}"
            metadata_comment += snode_str + "\n"
            if npu_config.dump_fx_graph:
                from ..lowering_fx import snodes_to_fx
                gm = snodes_to_fx.get(str(snodes), "")
                gm_str = "\n# Graph Module str:\n"
                gm_str += "\n".join([f"# {line}" for line in gm.split("\n")])
                metadata_comment += gm_str + "\n"

            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # log kernel metadata for offline analysis.
            # E.g. one can find all unaligned inner reduction and check if
            # padding helps with the perf kernel by kernel.
            if metrics.is_metric_table_enabled("kernel_metadata"):
                metrics.log_kernel_metadata(kernel_name, kernel_path, src_code)

        return kernel_name, src_code

    def codegen_node(
            self, node: Union[scheduler.FusedSchedulerNode, scheduler.SchedulerNode]
    ):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """

        nodes: List[scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        schedule_log.debug("Schedule:\n %s", node_schedule)

        return self.codegen_node_schedule(
            NPUKernelFeatures(node_schedule, numel, rnumel), nodes
        )

    def decide_codegen_dims_in_kernel(self, node_schedule, kernel):
        def current_reduction_nodes(nodes):
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

        with kernel:
            # 1. transform dims: create new dims to substitute floor_divide and modular expression
            stack = contextlib.ExitStack()
            for _, node in enumerate(node_schedule):
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node._body.transform_dims_in_indexing(index_vars)
            # 2. go through range_tree_nodes to findout, to find one axis could be substituted by others
            self.additional_nodes_to_be_subs(kernel, kernel.range_tree_nodes_substituted)
            # 3.do the substitution on all indexing
            for node in node_schedule:
                if node in (EnableReduction, DisableReduction):
                    continue
                indexing = node._body.indexing
                node._body.substituted_dims_in_indexing(indexing, kernel, kernel.range_tree_nodes_substituted)

            # 4.remove the substituted dims from kernel
            for var, _ in kernel.range_tree_nodes_substituted.items():
                if (var in kernel.range_tree_nodes):
                    root = kernel.range_tree_nodes[var].parent
                    root.remove_entry(var)
            # select split and tiling axis
            split_tiling = SplitTiling(kernel)
            split_tiling.select_split_tiling_axis()
            kernel.load_store_indexing = split_tiling.indexing
            # ReductionAnalysis depends on kernel.load_store_indexing 
            if kernel.inside_reduction:
                kernel.reduce_analysis = ReductionAnalysis(kernel)

    def additional_nodes_to_be_subs(self, kernel, node_to_be_substituted):
        for node in kernel.range_tree_nodes.values():
            if node.expr != sympy_index_symbol(f"{node.parent.prefix}index") \
                    or len(node.parent.var_ranges) == 1 \
                    or node.symbol() in node_to_be_substituted:
                continue
            numel = sympy.Integer(1)
            new_var_expr = sympy.Integer(0)
            for k, s in node.parent.var_ranges.items():
                if k == node.symbol():
                    continue
                numel = numel * s
                sub_node = kernel.range_tree_nodes[k]
                new_var_expr = new_var_expr + sub_node.symbol() * sub_node.divisor

            if numel == node.length:
                node_to_be_substituted[node.symbol()] = [(node.length, new_var_expr)]
            else:
                log.warning("sub nodes (expr%s, numel:%d) can not make up parent node(%s:%d)",
                            new_var_expr, numel, node.symbol(), node.length)
