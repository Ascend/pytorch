import collections
import contextlib
import itertools
import functools
import os
from typing import Sequence, List, Any, Union, Iterable
import sympy

import torch
from torch.fx.immutable_collections import immutable_dict
from torch._dynamo.utils import counters, preserve_rng_state
from torch._inductor import metrics
from torch._inductor.codecache import code_hash, PyCodeCache
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._inductor.codegen.simd import DisableReduction, EnableReduction, SIMDKernelFeatures, SIMDKernel
from torch._inductor.codegen.simd import schedule_log, scheduler, WhyNoFuse, TritonTemplateBuffer
from torch._inductor.codegen.simd_kernel_features import NodeScheduleMarker
from torch._inductor.codegen.triton import (
    TritonScheduling, BaseSchedulerNode, config, schedule_log,
    get_fused_kernel_name, get_kernel_category_by_source_code, Placeholder,
    get_kernel_metadata, get_path, IndentedBuffer)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V
from torch._inductor.dependencies import MemoryDep, StarDep
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.simd import CandidateTiling

from .npu_kernel_features import NumelList, NPUKernelFeatures
from .triton import NPUIndexTritonKernel, NPUTritonKernel, NPUTritonKernelWithLoop, flatten
from .triton_combo_kernel import NPUComboKernel
from .. import config as npu_config
from ..lowering import (
    create_fx_from_snodes_by_traced_graph,
    create_compile_kwargs,
    generate_fx_graph_code,
    dump_fx_graph_code
)
from ..fx_passes.utils.schedule_node_utils import is_multi_stream
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

class NPUNoLinearTritonScheduling(TritonScheduling):
    def __init__(self, input_scheduler):
        super().__init__(input_scheduler)
        from ..config import inductor_ascend_linear_mode
        self.kernel_type = NPUTritonKernelWithLoop
        if inductor_ascend_linear_mode == 'no_linear':
            self.kernel_type = NPUTritonKernel

class NPUTritonScheduling(TritonScheduling):
    def __init__(self, input_scheduler):
        super().__init__(input_scheduler)
        self.kernel_type = NPUIndexTritonKernel

    def group_fn(self, sizes):
        groups = list()
        for s in sizes:
            if not s:
                groups.append(1)
            elif isinstance(s, list):
                group = flatten(s)
                groups.append(NumelList(tuple(group)) if isinstance(group, list) else group)
            else:
                groups.append(s)
        return tuple(groups)

    @classmethod
    def create_tiling(
            cls, pw_tiling: Sequence[sympy.Expr], reduction_tiling: Sequence[sympy.Expr]
    ) -> dict[str, sympy.Expr]:
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

    def create_kernel_choices(
            self, kernel_features: SIMDKernelFeatures, kernel_args, kernel_kwargs
    ) -> List[SIMDKernel]:

        if kernel_features.contains_op("scan"):
            kernel_kwargs = dict(kernel_kwargs)
            kernel_kwargs["override_cooperative_reduction"] = False

        return [self.kernel_type(*kernel_args, **kernel_kwargs)]

    def make_ttir_for_check(self, src_code):
        # 1. use triton_ as tmp kernel name
        checker_src_code = src_code.replace(str(Placeholder.KERNEL_NAME),
                                            "triton_")
        # 2. find inductor_meta str
        inductor_meta_str = next(
            (line for line in checker_src_code.splitlines()
             if "inductor_meta" in line),
            None,
        )
        inductor_meta = eval(inductor_meta_str.strip().split("=", 1)[1].rstrip(','))
        # 3. disable cache and limit configs count for precompile
        inductor_meta["force_disable_caches"] = True
        checker_src_code = checker_src_code.replace(
            inductor_meta_str, f"    inductor_meta={inductor_meta},")
        mod = PyCodeCache.load(checker_src_code)
        # 4. precompile for dsl check
        mod.triton_._triton_make_ttir()
        # 5. remove tmp compile cache
        try:
            assert mod.__file__
            os.remove(mod.__file__)
        except FileNotFoundError:
            pass

    # transform indexing before call codegen_node_schedule_with_kernel
    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures, nodes, origin_node=None):
        node_schedule = kernel_features.node_schedule
        tiling = self.select_tiling(
            node_schedule, kernel_features.numel, kernel_features.reduction_numel
        )

        kernels = self.create_kernel_choices(
            kernel_features, [tiling], {"features": kernel_features}
        )

        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        MultiKernel.merge_workspaces_inplace(kernels)
        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
                self.make_ttir_for_check(src_code)

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
                    traced_graph_hash = code_hash(src_code)

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
        if is_multi_stream():
            final_kernel.call_kernel(name=final_kernel.kernel_name, origin_node=origin_node)
        else:
            final_kernel.call_kernel(name=final_kernel.kernel_name, origin_node=None)

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

    def codegen_template(self, template_node, epilogue_nodes, only_gen_src_code=False):
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        kernel, render = template_node.node.make_kernel_render(template_node.node)
        with kernel:
            if not only_gen_src_code:
                for node in [template_node, *epilogue_nodes]:
                    node.mark_run()
            partial_code = render()
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                for node in epilogue_nodes:
                    node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

        if not isinstance(partial_code, str):
            partial_code.finalize_hook("<DEF_KERNEL>")
            partial_code.finalize_hook("<ARGDEFS>", strict=False)
        # finalize must be called after adding epilogue above
        with V.set_kernel_handler(kernel):
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                if isinstance(partial_code, str):
                    src_code = partial_code
                else:
                    partial_code.finalize_hook("<STORE_OUTPUT>")
                    src_code = partial_code.code
            node_schedule = [template_node, *epilogue_nodes]

            if config.benchmark_kernel:
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                grid_args = V.graph.sizevars.size_hints(kernel.call_sizes)
                assert kernel.meta is not None, "meta is None"
                grid = kernel.grid_fn(*grid_args, kernel.meta)

                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb, grid).getvalue()}"
                )

            if only_gen_src_code:
                return src_code
            traced_graph_hash = None
            kernel_name, src_code = self.define_kernel(src_code, node_schedule, kernel, traced_graph_hash)

        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, template_node, template_node.node)

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.scheduler.free_buffers()
        return None

    def codegen_combo_kernel(self, combo_kernel_node):
        subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
        custom_part_algorithm = combo_kernel_node.use_custom_partition_algo
        enable_autotune = combo_kernel_node.enable_autotune
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_part_algorithm
        )

        kernel_code_list = self.generate_combo_kernel_code(
            subkernel_nodes, custom_part_algorithm, enable_autotune, mixed_sizes
        )

        for src_code, kernel, _ in kernel_code_list:
            kernel_name, _ = self.define_kernel(src_code, [combo_kernel_node], kernel, None)
            self.codegen_comment([combo_kernel_node])
            log.debug("ComboKernels: generated kernel %s.", kernel_name)
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.free_buffers_in_scheduler()

    def generate_combo_kernel_code(
        self,
        subkernel_nodes: list[BaseSchedulerNode],
        custom_part_algorithm: bool,
        enable_autotune: bool,
        mixed_sizes: bool,
        only_gen_src_code: bool = False,
    ) -> list[tuple[str, Any, Any]]:

        fused_node_lists = [node.get_nodes() for node in subkernel_nodes]
        subkernel_map, node_schedule_map = {}, {}
        for pn, nodes in zip(subkernel_nodes, fused_node_lists):
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            tiling = self.select_tiling(node_schedule, numel, rnumel)
            node_schedule_map[pn] = node_schedule, tiling, numel, rnumel
            subkernel_map[pn] = NPUComboKernel.create_triton_kernel(
                tiling,
                features=NPUKernelFeatures(node_schedule, numel, rnumel),
                optimize_mask=not mixed_sizes,
            )

        partitions = NPUComboKernel.horizontal_partition(
            nodes=subkernel_nodes,
            triton_scheduling=self,
            custom_algorithm=custom_part_algorithm,
            kernel_map=subkernel_map,
            node_info_map=node_schedule_map,
        )
        log.debug(
            "ComboKernels: %d nodes partitioned into %s groups",
            len(subkernel_nodes),
            [len(p) for p in partitions],
        )
        kernel_code_list = []
        for node_group in partitions:
            fused_node_lists = [node.get_nodes() for node in node_group]
            kernel = NPUComboKernel(
                enable_autotune=enable_autotune,
                mixed_sizes=mixed_sizes,
            )

            for pn, nodes in zip(node_group, fused_node_lists):
                self.codegen_node_schedule_with_kernel(
                    node_schedule_map[pn][0],
                    kernel.create_sub_kernel(subkernel_map[pn]),
                )
                subkernel = subkernel_map[pn]
                node_schedule = node_schedule_map[pn][0]
                if not only_gen_src_code:
                    with V.set_kernel_handler(subkernel):  # type: ignore[call-arg]
                        for node in NodeScheduleMarker.only_nodes(node_schedule):
                            node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            src_code = kernel.codegen_kernel()
            kernel_code_list.append((src_code, kernel, node_group))
        return kernel_code_list

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
                from ..lowering import snodes_to_fx
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
        if is_multi_stream():
            return self.codegen_node_schedule(
                NPUKernelFeatures(node_schedule, numel, rnumel), nodes, node
            )
        else:
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
                    NPUIndexTritonKernel.is_compatible((numel2, rnumel2), n.get_ranges())
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
