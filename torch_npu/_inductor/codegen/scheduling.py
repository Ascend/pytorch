import collections
import contextlib
import itertools
import functools
import os
from typing import Dict, Sequence, List, Iterable, Any, Union
import sympy
import torch
from torch._dynamo.utils import counters, preserve_rng_state
from torch._inductor import scheduler, metrics
from torch._inductor.codecache import code_hash, PyCodeCache
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._inductor.codegen.simd import DisableReduction, EnableReduction, SIMDKernelFeatures, SIMDKernel
from torch._inductor.codegen.simd import schedule_log, scheduler, WhyNoFuse, TritonTemplateBuffer
from torch._inductor.codegen.triton import (TritonScheduling, log, config)
from torch._inductor.codegen.triton import (
    TritonScheduling,
    BaseSchedulerNode,
    config,
    schedule_log,
    get_fused_kernel_name,
    get_kernel_category_by_source_code,
    Placeholder,
    get_kernel_metadata,
    get_path,
    IndentedBuffer
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import sympy_index_symbol, ModularIndexing, FloorDiv, sympy_product
from torch._inductor.virtualized import V
from torch.fx.immutable_collections import immutable_dict
from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.simd import CandidateTiling

from .triton import NPUIndexTritonKernel, flatten
from .kernel_analysis import ReductionAnalysis
from .npu_kernel_features import NumelList, NPUKernelFeatures
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
def are_long_distant_nodes(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """
    This function prevents fusion for nodes that can increase memory
    footprint. This problem is more common in horizontal fusion, where nodes
    that are far apart in the original order get fused, lengthening the live
    intervals of tensors. This is very evident in models with activation
    checkpointing, where the recomputed nodes from different checkpointed
    regions get fused and significantly increase the memory footprint.

    The current attempt is a quick, possibly hacky, heuristic to prevent the
    fusion of nodes that are far away in the original order.

    A better but difficult to implement heurisitic would be to use live
    intervals of the buffers, find region of peak pressure in the original
    program and prevent fusion that crosses that peak region. We might need
    special care or good approximation in this implementation, as fusion of
    node changes live intervals, and re-computing live intervals and peak
    memory after each fusion can introduce large compilation overhead.
    """
    proximity_score = max(
        abs(node1.min_order - node2.max_order),
        abs(node2.min_order - node1.max_order),
    )
    # GPU default score is 64, A5 use 20 to avoiding runtime errors in large kernels.
    return proximity_score > 20


@classmethod
def create_tiling(
        cls, pw_tiling: Sequence[sympy.Expr], reduction_tiling: Sequence[sympy.Expr]
) -> Dict[str, sympy.Expr]:
    """
    Create a tiling dict from pointwise and reduction splits.
    """

    pw_tiling = flatten_groups(pw_tiling)
    pw_prefixes = ["w", "v", "t", "z", "y", "x"][-len(pw_tiling):]
    if len(reduction_tiling) == 0:
        reduction_prefixes = []
    else:
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
                    traced_graph_hash = code_hash(traced_graph.print_readable(print_output=False) + torch.__version__)

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
            if len(fused_name) > 35:
                fused_name = fused_name[0:35]
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

    def benchmark_fused_nodes(self, nodes):
        with preserve_rng_state(), torch.npu.device(
            V.graph.get_current_device_or_throw()
        ):
            src_code = self.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel=True
            )
            mod = PyCodeCache.load(src_code)

            def cache_file_path():
                assert mod.__file__ is not None
                return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

            def load_cache():
                path = cache_file_path()
                if os.path.exists(path):
                    with open(path) as fd:
                        return float(fd.read())
                return None

            def store_cache():
                path = cache_file_path()
                with open(path, "w") as fd:
                    fd.write(str(ms))

            log.debug(
                "kernel src code for %s written to: %s",
                {n.get_name() for n in nodes},
                mod.__file__,
            )
            ms = load_cache()
            if ms is not None:
                return ms, mod.__file__

            args = mod.get_args()
            call = mod.call
            wrapped_jit_function = mod.triton_

            # call once to trigger the compilation
            try:
                call(wrapped_jit_function.clone_args(*args)[0])
            except Exception as e:
                log.debug(
                    "Exception (%s) in compiling fused nodes %s",
                    e,
                    {n.get_name() for n in nodes},
                )
                ms = float("inf")
                store_cache()
                return ms, mod.__file__

            launchers = wrapped_jit_function.launchers
            assert len(launchers) == 1
            if launchers[0].n_spills > 0:
                # skip benchmarking the kernel if there are register spills
                ms = float("inf")
            else:
                # We have to clone the inplace updated arguments to avoid earlier calls
                # generating out of range indices for later calls.
                ms = benchmarker.benchmark_gpu(
                    lambda: call(wrapped_jit_function.clone_args(*args)[0])
                )

                # overhead of cloning args gives bias for fusing the kernel
                # in the case of mutating/in-placeable second fusion
                # the input values between benchmarking
                if len(wrapped_jit_function.mutated_arg_names) > 0:
                    ms = ms - benchmarker.benchmark_gpu(
                        lambda: wrapped_jit_function.clone_args(*args)
                    )

            log.debug(
                "The fused kernel for %s took %.3f ms to run",
                {n.get_name() for n in nodes},
                ms,
            )
            store_cache()
            return ms, mod.__file__

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
        if not nodes[0].is_template():
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            tiling = self.select_tiling(node_schedule, numel, rnumel)
            kernel = self.kernel_type(
                tiling,
                features=NPUKernelFeatures(node_schedule, numel, rnumel),
            )
            setattr(kernel, "node_schedule", node_schedule)
            self.decide_codegen_dims_in_kernel(node_schedule, kernel)
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            with config.patch(
                "benchmark_kernel", benchmark_kernel
            ), V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
        else:
            template_node = nodes[0]
            epilogue_nodes = nodes[1:]

            with config.patch("benchmark_kernel", benchmark_kernel):
                src_code = self.codegen_template(
                    template_node, epilogue_nodes, only_gen_src_code=True
                )

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return src_code

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

    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
            node2, scheduler.ForeachKernelSchedulerNode
        ):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

        if isinstance(node1, scheduler.SchedulerNode) and hasattr(node1, "_body") and node1._body.has_op("cat_store"):
            return False

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group
        why = WhyNoFuse(node1, node2)

        if node1.is_split_scan() and not node2.is_split_scan():
            if node2.is_reduction():
                why("Split scan cannot fuse with reductions")
        elif node2.is_split_scan() and not node1.is_split_scan():
            if node1.is_reduction():
                why("Split scan cannot fuse with reductions")

        if node1.is_reduction() and node2.is_reduction():
            reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
            if not reduction_can_fuse:
                why(
                    "numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
            return reduction_can_fuse

        if not node1.is_reduction() and not node2.is_reduction():
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                if not node2.is_template():
                    why(
                        "numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)",
                        numel1,
                        numel2,
                        rnumel1,
                        rnumel2,
                    )
                    return False
                else:
                    # prologue fusion input sizes differ from output group
                    # fuse so long as this node matches the group of existing prologue nodes
                    for node in node2.get_nodes():
                        # dont need to check epilogue nodes for prologue fusion, break after template
                        if node.is_template():
                            break
                        # we would have already restricted prologue from fusing if it had multiple
                        # uses, so it must be fusing into this node
                        if not node.used_buffer_names() & node1.get_buffer_names():
                            continue
                        _, (pro_numel, pro_rnumel) = node.group
                        if not (numel1 == pro_numel and rnumel1 == pro_rnumel):
                            why(
                                "numel/rnumel mismatch prologue mismatch (%s, %s), (%s, %s)",
                                numel1,
                                pro_numel,
                                rnumel1,
                                pro_rnumel,
                            )
                            return False

            for n, node_name in zip((node1, node2), ("node1", "node2")):
                if n.is_template():
                    # Only allow fusion for TritonTemplates for now.
                    # Fusion for CUDATemplates are not supported.
                    is_triton_template = isinstance(
                        n.get_template_node(), TritonTemplateBuffer
                    )
                    if not is_triton_template:
                        why(f"{node_name} is not TritonTemplateBuffer")
                    return is_triton_template

            # check for a bad combined tiling
            tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
            tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
            tiling3 = self.select_tiling(
                node1.get_nodes() + node2.get_nodes(), numel1, rnumel1
            )
            if config.triton.tiling_prevents_pointwise_fusion:
                cond = True
                if len(tiling1) > 2:
                    if len(tiling2) > 2:
                        cond = tiling1 == tiling2 == tiling3
                    else:
                        cond = tiling1 == tiling3
                elif len(tiling2) > 2:
                    cond = tiling2 == tiling3
                if not cond:
                    why(
                        "tiling mismatch (%s, %s, %s)",
                        tiling1,
                        tiling2,
                        tiling3,
                    )
                    return False

            return True

        if not node1.is_reduction() and node2.is_reduction():
            if not (rnumel1 == 1 and rnumel2 != 1):
                raise AssertionError
            if numel1 == numel2 * rnumel2:
                if not all(
                    SIMDKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                    for n in node1.get_nodes()
                ):
                    why("nodes numel/rnumel incompatibility")
                    return False
                if (
                    config.triton.tiling_prevents_reduction_fusion
                    and not node1.is_template()
                ):
                    valid_tiling_group = set()
                    valid_tiling_group.add((*numel1, 1) if isinstance(numel1, NumelList) else (numel1, 1))
                    valid_tiling_group.add((*numel2, *rnumel2, 1)
                        if isinstance(numel2, NumelList) and isinstance(rnumel2, NumelList) else (numel2, rnumel2, 1))
                    valid_tiling_group.add(numel1)

                    is_reduction_tiling_valid = tuple(
                        self.select_tiling(node1.get_nodes(), numel1).values()
                    ) in valid_tiling_group
                    if not is_reduction_tiling_valid:
                        why("invalid tiling for reduction")
                    return is_reduction_tiling_valid
                return True

            if numel1 != numel2:
                why("nodes numel incompatibility")
            return numel1 == numel2

        if not (node1.is_reduction() and not node2.is_reduction()):
            raise AssertionError
        # swap args to hit the case above
        return self.can_fuse_horizontal(node2, node1)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

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
                # pure_simt_kernel, high dim reduction don't use persitent reduction
                if kernel.is_unified_simt_kernel() and kernel.reduction_dim() != len(kernel.golden_var_list) - 1:
                    kernel.persistent_reduction = False
            # no_loop_axis depends on persistent reduction
            split_tiling.select_no_loop_axis()

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

    @classmethod
    @functools.lru_cache(32)
    def candidate_tilings(cls, node, numel, reduction_numel) -> list[CandidateTiling]:
        """
        The main difference from gpu is default tiling, npu needs non-collapse ranges.
        """
        is_pointwise = reduction_numel == 1

        def assert_true(cond, msg=""):
            if not cond:
                raise AssertionError(msg)

        def tile_ranges(is_pointwise: bool, ranges, rw) -> list[CandidateTiling]:
            assert_true(len(rw.range_vars) == len(ranges), f"{rw.range_vars=} {ranges=}")

            dep_sources = [rw.reads, rw.writes]
            assert_true(all(
                isinstance(dep, (MemoryDep, StarDep))
                for dep in itertools.chain.from_iterable(dep_sources)
            ))
            deps = [
                dep
                for dep in itertools.chain.from_iterable(dep_sources)
                if dep.name not in V.graph.removed_buffers
                and isinstance(dep, MemoryDep)
            ]
            write_names = OrderedSet([dep.name for dep in rw.writes])

            def collapse_ranges(ranges: Sequence[sympy.Expr]) -> sympy.Expr:
                return V.graph.sizevars.simplify(sympy_product(ranges))

            tilings = [
                CandidateTiling(
                    tiling=cls.create_partial_tiling(
                        ranges, is_pointwise
                    ),
                    name="none",
                    score=0,
                )
            ]

            for dep in deps:
                strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
                assert_true(len(strides) == len(ranges))
                try:
                    split = strides.index(1) + 1
                    if split == len(ranges):
                        continue
                    if all(s == 0 for s in strides[split:]):
                        continue

                except ValueError:
                    continue

                tiled_groups = (
                    collapse_ranges(ranges[:split]),
                    collapse_ranges(ranges[split:]),
                )

                # score by number of elements
                score = V.graph.sizevars.size_hint(
                    sympy_product(
                        size for size, stride in zip(ranges, strides) if stride != 0
                    )
                )
                if dep.name in write_names:
                    # ngimel said contiguous writes is more important than reads
                    score *= 2
                if CandidateTiling.is_good_size(tiled_groups[0]):
                    score *= 2
                if CandidateTiling.is_good_size(tiled_groups[1]):
                    score *= 2

                if (
                    V.graph.sizevars.size_hint(
                        score - sympy_product(itertools.chain(ranges, reduction_ranges))
                    )
                    >= 0
                ):
                    tilings.append(
                        CandidateTiling(
                            tiling=cls.create_partial_tiling(
                                [
                                    collapse_ranges(ranges[:split]),
                                    collapse_ranges(ranges[split:]),
                                ],
                                reduction_numel,
                            ),
                            score=score,
                            name=dep.name,
                        )
                    )

            return tilings

        pointwise_ranges, reduction_ranges = node.get_ranges()
        if len(pointwise_ranges) <= 1 and len(reduction_ranges) <= 1:
            return []

        # Tile either pointwise or reduction dims.
        pointwise_ranges, reduction_ranges = node.get_ranges()
        partial_tilings = tile_ranges(
            is_pointwise,
            pointwise_ranges if is_pointwise else reduction_ranges,
            node.pointwise_or_reduction_read_writes(is_pointwise),
        )

        # Fill in the missing ranges.
        full_tilings = [
            CandidateTiling(
                tiling=cls.complete_partial_tiling(
                    tiling.tiling, numel, reduction_numel
                ),
                score=tiling.score,
                name=tiling.name,
            )
            for tiling in partial_tilings
        ]

        return full_tilings