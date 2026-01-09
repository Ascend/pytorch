import contextlib
import copy
import itertools
import logging
import math
from types import ModuleType
from typing import cast, Any, Callable, Optional, Sequence, Union

import torch
from torch._inductor import config, ir
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch._inductor.runtime.runtime_utils import green_text, red_text
from torch._inductor.scheduler import (
    SchedulerNode,
    BaseSchedulerNode,
    FusedSchedulerNode,
    Scheduler,
    WhyNoFuse,
    fusion_log,
)
from torch._inductor.ir import ChoiceCaller, MultiTemplateBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .codegen.catlass.catlass_kernel import CATLASSTemplateCaller


def patch_scheduler():
    def patch_multi_template_buffer():

        @contextlib.contextmanager
        def swap_as_caller(self, caller: ChoiceCaller):  # type: ignore[no-untyped-def]
            assert isinstance(
                caller, (
                    torch._inductor.select_algorithm.TritonTemplateCaller,
                    CATLASSTemplateCaller,
                )
            ), type(caller)
            assert self.layout == caller.layout

            render = self.make_kernel_render
            self.make_kernel_render = caller.get_make_kernel_render()
            try:
                yield
            finally:
                self.make_kernel_render = render

        def finalize_as_caller(self, caller: ChoiceCaller) -> None:
            assert isinstance(
                caller, (
                    torch._inductor.select_algorithm.TritonTemplateCaller,
                    CATLASSTemplateCaller,
                )
            ), type(caller)
            assert self.get_size() == caller.layout.size
            assert self.get_stride() == caller.layout.stride
            self.make_kernel_render = caller.get_make_kernel_render()

        MultiTemplateBuffer.swap_as_caller = swap_as_caller
        MultiTemplateBuffer.finalize_as_caller = finalize_as_caller

    patch_multi_template_buffer()

    def fuse_nodes_once(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuse(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = OrderedSet(nodes)
        if fusion_log.isEnabledFor(logging.DEBUG):
            fusion_log.debug("fuse_nodes_once, candidates:")
            for node in fused_nodes:
                fusion_log.debug("  " + node.debug_str_short())  # noqa: G003

        # These are potential fusions which we are async compiling,
        # and which we will benchmark profitability of.
        pending_fusions: dict[
            BaseSchedulerNode,
            tuple[Callable[[], bool], BaseSchedulerNode, BaseSchedulerNode],
        ] = {}

        def fuse_two_nodes(
            node1: BaseSchedulerNode, node2: BaseSchedulerNode
        ) -> BaseSchedulerNode:
            fusion_log.debug("fusing %s with %s", node1.get_name(), node2.get_name())

            device = node1.get_device()
            assert node2.get_device() == device

            # in case the node has been modified
            may_new_node1 = self.get_fused_node(node1)
            may_new_node2 = self.get_fused_node(node2)
            node3 = self.get_backend(device).fuse(may_new_node1, may_new_node2)
            fused_nodes.remove(node1)
            fused_nodes.remove(node2)
            fused_nodes.add(node3)
            self.name_to_fused_node.update(
                {n.get_name(): node3 for n in node3.get_nodes()}
            )
            return node3

        def resolve_pending_fusions(
            node1: BaseSchedulerNode, node2: BaseSchedulerNode
        ) -> None:
            while (
                self.get_fused_node(node1) in pending_fusions
                or self.get_fused_node(node2) in pending_fusions
            ):
                pending_fusion = pending_fusions.get(
                    self.get_fused_node(node1),
                    pending_fusions.get(self.get_fused_node(node2), None),
                )
                assert pending_fusion is not None

                is_speedup, node_key1, node_key2 = pending_fusion
                pending_fusions.pop(node_key1, None)
                pending_fusions.pop(node_key2, None)

                assert self.get_fused_node(node_key1) is node_key1
                assert self.get_fused_node(node_key2) is node_key2

                if not is_speedup() or self.will_fusion_create_cycle(node1, node2):
                    continue

                fuse_two_nodes(node_key1, node_key2)

        for node1, node2 in self.get_possible_fusions(nodes):
            # if either node is in a pending fusion, resolve it.
            # since we iterate on potential fusions based on profitability
            # the first potential fusion should take precedence.
            resolve_pending_fusions(node1, node2)
            node1 = self.get_fused_node(node1)
            node2 = self.get_fused_node(node2)

            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                speedup = self.speedup_by_fusion(node1, node2)
                if callable(speedup):
                    pending_fusions[node1] = (speedup, node1, node2)
                    pending_fusions[node2] = (speedup, node1, node2)
                    continue

                if not speedup:
                    continue

                fuse_two_nodes(node1, node2)

        seen_pair_speedup_fn: OrderedSet[Callable[[], bool]] = OrderedSet()
        for is_speedup_fn, node_key1, node_key2 in pending_fusions.values():
            if is_speedup_fn in seen_pair_speedup_fn:
                continue

            seen_pair_speedup_fn.add(is_speedup_fn)

            assert self.get_fused_node(node_key1) is node_key1
            assert self.get_fused_node(node_key2) is node_key2

            if is_speedup_fn() and not self.will_fusion_create_cycle(
                node_key1, node_key2
            ):
                fuse_two_nodes(node_key1, node_key2)

        nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        nodes = self.topological_sort_schedule(nodes)
        self.prune_redundant_deps(nodes)
        return nodes

    def speedup_by_fusion(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Union[bool, Callable[[], bool]]:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """

        is_multi_template = any(
            n.is_template()
            and isinstance(n.get_template_node(), ir.MultiTemplateBuffer)
            for n in (node1, node2)
        )
        if not config.benchmark_fusion and not is_multi_template:
            return True

        if (
            node1.is_template()
            and not isinstance(node1.get_template_node(), ir.TritonTemplateBuffer)
            or node1.is_foreach()
            or node2.is_foreach()
        ):
            return True

        node_list_1 = node1.get_nodes()
        device = node_list_1[0].get_device()
        assert device

        # don't support benchmark fusion for CPU right now.
        if device.type == "cpu":
            return True

        node_list_2 = node2.get_nodes()
        node_list_fused = list(itertools.chain(node_list_1, node_list_2))

        # We can not accurately benchmark kernel using atomic_add
        # due to how we generate random integer inputs.
        # Skip benchmarking them by allowing fusion.
        if self._any_atomic_add(node_list_fused):
            return True

        from triton.compiler.errors import CompilationError

        why = WhyNoFuse(node1, node2)

        device = node_list_fused[0].get_device()
        assert device is not None

        def log_fusion(ms_fused: float, ms1: float, ms2: float, choice_name="") -> None:
            if fusion_log.isEnabledFor(logging.DEBUG):
                if ms_fused < ms1 + ms2:
                    fusion_log.debug(
                        "choice: [%s], can fuse (benchmark): fusing %s with %s cause %sx speedup, %s",
                        choice_name,
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        green_text(f"{(ms1 + ms2) / ms_fused:.3f}"),
                        f"ms1: {ms1}, ms2: {ms2}, ms_fused:{ms_fused}",
                    )
                else:
                    fusion_log.debug(
                        "choice: [%s], cannot fuse (benchmark): fusing %s with %s cause %sx slowdown, %s",
                        choice_name,
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        red_text(f"{ms_fused / (ms1 + ms2):.3f}"),
                        f"ms1: {ms1}, ms2: {ms2}, ms_fused:{ms_fused}",
                    )

        async_compile = torch._inductor.async_compile.AsyncCompile()

        def replace_operation_buffer(
            orig_node: ir.MultiTemplateBuffer, new_node: ir.OperationBuffer
        ) -> None:
            replaced_buf_name = new_node.get_name()
            orig_buf_name = orig_node.get_name()
            assert isinstance(orig_buf_name, str) and isinstance(replaced_buf_name, str)

            replaced_op_name = new_node.get_operation_name()
            orig_op_name = orig_node.get_operation_name()
            assert isinstance(orig_op_name, str) and isinstance(replaced_op_name, str)

            del V.graph.name_to_buffer[replaced_buf_name]
            new_node.name = orig_buf_name

            del V.graph.name_to_op[replaced_op_name]
            new_node.operation_name = orig_op_name

            orig = V.graph.buffers.index(orig_node)
            V.graph.buffers.remove(new_node)
            V.graph.buffers[orig] = new_node
            V.graph.name_to_buffer[orig_buf_name] = new_node

            orig = V.graph.operations.index(orig_node)
            V.graph.operations.remove(new_node)
            V.graph.operations[orig] = new_node
            V.graph.name_to_op[orig_op_name] = new_node

        def compile_kernel(
            nodes: Sequence[BaseSchedulerNode]
        ) -> tuple[Optional[LambdaFuture], ModuleType]:
            src_code_or_mod = self.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel=True
            )

            if self.get_backend(device)._catlass_scheduling.is_catlass_template(
                nodes[0]
            ):
                return None, src_code_or_mod

            if not async_compile.use_process_pool():
                fut = None
            else:
                mod = PyCodeCache.load(src_code_or_mod)
                fut = async_compile.triton(
                    kernel_name="triton_", source_code=src_code_or_mod
                )
                assert isinstance(fut, LambdaFuture)

            return (fut, mod)

        # After the succesful fusion with Template, we finalize its config.
        # Subsequently we benchmark but dont update. Checking for SchedulerNode, instead of FusedSchedulerNode
        # accomplishes this.
        if is_multi_template and any(
            n.get_template_node() is not None for n in (node1, node2)
        ):
            epilogue_fusion = node1.get_template_node() is not None
            multi_node = (
                node1.get_template_node()
                if epilogue_fusion
                else node2.get_template_node()
            )
            assert isinstance(multi_node, ir.MultiTemplateBuffer)
            choice_timings = multi_node.choice_timings
            _, ms1 = multi_node.get_min_choice()

            # Eagerly compile and benchmark non-template nodes
            ms1_choice, ms1 = multi_node.get_min_choice()

            ms2, path2 = (
                self.benchmark_fused_nodes(node_list_2)
                if epilogue_fusion
                else self.benchmark_fused_nodes(node_list_1)
            )

            # Start compiling choices in parallel
            future_choices: list[tuple[Any, Optional[LambdaFuture], ModuleType]] = []
            template_choices = 0
            for choice, unfused_time in sorted(
                choice_timings.items(), key=lambda x: x[1]
            ):
                if not (
                    isinstance(choice, torch._inductor.ir.TritonTemplateCallerBase)
                    or (
                        isinstance(choice, CATLASSTemplateCaller)
                        and multi_node == node1.get_template_node()
                    )
                ):
                    continue

                # For prologue fusion we check if the underlying template of the choice
                # supports all allowed prologue inputs. If not, we skip this choice in
                # the fusion benchmark.
                # Currently, persistent+TMA Triton template does not due to the TMA-based loads.
                if (
                    not epilogue_fusion
                    and hasattr(choice, "allowed_prologue_inps")
                    and choice.allowed_prologue_inps != multi_node.allowed_prologue_inps
                ):
                    continue

                if unfused_time >= ms1 + ms2:
                    break

                if isinstance(choice, CATLASSTemplateCaller):
                    out_tensorbox = choice.output_node()
                    out_storage = out_tensorbox.data
                    assert isinstance(out_storage, ir.StorageBox)
                    out_buffer = out_storage.data
                    assert isinstance(out_buffer, ir.OperationBuffer)
                    # hack out_buffer's name to judge if can fuse
                    out_buffer.name = multi_node.get_name()

                    # Since current can_fuse does not check CATLASSScheduling.can_fuse
                    # for MultiTemplateBuffer, we hack here to check if can_fuse again
                    if not self.get_backend(
                        device
                    )._catlass_scheduling._can_fuse_epilogue_impl(
                        out_buffer, [], node2
                    ):
                        del out_buffer
                        continue

                template_choices += 1
                if template_choices > config.max_epilogue_benchmarked_choices:
                    break

                with multi_node.swap_as_caller(choice):
                    new_node_list_fused = node_list_fused
                    if isinstance(choice, CATLASSTemplateCaller):
                        # hack for the template node
                        new_node = self.create_scheduler_node(out_buffer)
                        for new_out, old_out in zip(
                            new_node.get_outputs(), node1.get_outputs()
                        ):
                            new_out.users = old_out.users
                        new_node_list_fused = copy.copy(node_list_fused)
                        new_node_list_fused[0] = new_node
                    future_choices.append(
                        (choice, *compile_kernel(new_node_list_fused))
                    )

            if len(future_choices) == 0:
                return False

            def benchmark_when_ready() -> bool:
                min_ms_fused = float("inf")
                ms_fused_choice = None
                ms_fused_mod = None

                new_timings = {}
                # Benchmark each choice after compilation completes
                for choice, future, mod_fused in future_choices:
                    try:
                        if future is not None:
                            future.result()

                    # Ideally we would more narrowly catch Exceptions here but
                    # triton  will unpredictably error with valid prologue fusions
                    except Exception as e:
                        if fusion_log.isEnabledFor(logging.DEBUG):
                            fusion_log.debug(
                                "Exception in compiling %s: %s",
                                "prologue" if not epilogue_fusion else "epilogue",
                                str(e),
                            )
                        continue
                    with multi_node.swap_as_caller(choice):
                        ms_fused, path = self.benchmark_codegened_module(
                            mod_fused, device
                        )
                        new_timings[choice] = ms_fused
                        if ms_fused < min_ms_fused:
                            min_ms_fused = ms_fused
                            ms_fused_choice = choice
                            ms_fused_mod = mod_fused

                if ms_fused_choice:
                    log_fusion(min_ms_fused, ms1, ms2, ms_fused_choice.name)

                if min_ms_fused < (ms1 + ms2) and ms_fused_choice is not None:
                    multi_node.finalize_as_caller(ms_fused_choice)
                    multi_node._choice_timings = new_timings
                    if isinstance(ms_fused_choice, CATLASSTemplateCaller):
                        out_tensorbox = ms_fused_choice.output_node()
                        out_storage = out_tensorbox.data
                        assert isinstance(out_storage, ir.StorageBox)
                        out_buffer = out_storage.data
                        assert isinstance(out_buffer, ir.OperationBuffer)
                        # hack out_buffer's name to judge if can fuse
                        out_buffer.name = multi_node.get_name()
                        replace_operation_buffer(multi_node, out_buffer)
                        # NB: We have created a new scheduler node that replaced the original node,
                        # but the outer loop that called fuse_two_nodes(node1, node2) use the original
                        # node, so we have modified fuse_two_nodes
                        new_scheduler_node = self.create_scheduler_node(out_buffer)
                        idx = self.nodes.index(node1)
                        self.nodes[idx] = new_scheduler_node
                        self.name_to_node[node1.get_name()] = new_scheduler_node
                        self.name_to_fused_node[node1.get_name()] = new_scheduler_node

                        for new_out, old_out in zip(
                            new_scheduler_node.get_outputs(), node1.get_outputs()
                        ):
                            self.name_to_buf[old_out.get_name()] = new_out
                            new_out.users = old_out.users

                        new_scheduler_node.min_order = node1.min_order
                        new_scheduler_node.max_order = node1.max_order
                        new_scheduler_node.last_usage = node1.last_usage
                        # update workspace_size
                        choice.fbmreq = ms_fused_mod.bmreq
                        out_buffer.workspace_size = ms_fused_mod.bmreq.workspace_size
                    return True
                else:
                    return False

            return benchmark_when_ready

        else:
            # Start parallel compilation for all three kernels
            future_and_mod_l1 = compile_kernel(node_list_1)
            future_and_mod_l2 = compile_kernel(node_list_2)
            future_and_mod_l1_fused = compile_kernel(node_list_fused)

            def benchmark_when_ready() -> bool:
                from torch._inductor.runtime.triton_heuristics import (
                    NoTritonConfigsError,
                )

                try:
                    # Wait for all compilations to complete
                    for fut in (
                        future_and_mod_l1[0],
                        future_and_mod_l2[0],
                        future_and_mod_l1_fused[0],
                    ):
                        if fut is not None:
                            fut.result()

                    ms1, path1 = self.benchmark_codegened_module(
                        future_and_mod_l1[1], device
                    )
                    if math.isinf(ms1):
                        why("register spilling of the first kernel")
                        return False

                    ms2, path2 = self.benchmark_codegened_module(
                        future_and_mod_l2[1], device
                    )
                    if math.isinf(ms2):
                        why("register spilling of the second kernel")
                        return False

                    ms_fused, path_fused = self.benchmark_codegened_module(
                        future_and_mod_l1_fused[1], device
                    )
                    if math.isinf(ms_fused):
                        why("register spilling of the fused kernel")
                        return False

                    log_fusion(ms_fused, ms1, ms2)

                    if (
                        is_metric_table_enabled("slow_fusion")
                        and ms_fused >= ms1 + ms2
                        and (path1, path2) not in self.logged_slow_fusion
                    ):
                        self.logged_slow_fusion.add((path1, path2))
                        get_metric_table("slow_fusion").add_row(
                            lambda: {
                                "kernel1_path": path1,
                                "kernel1_latency": ms1,
                                "kernel2_path": path2,
                                "kernel2_latency": ms2,
                                "fused_kernel_path": path_fused,
                                "fused_kernel_latency": ms_fused,
                                "slow_down_ratio": ms_fused / (ms1 + ms2),
                            }
                        )

                    return ms_fused < ms1 + ms2

                except NoTritonConfigsError:
                    return False

                except CompilationError as e:
                    if "Loop-carried variable" in str(e):
                        return True
                    raise

            return benchmark_when_ready

    Scheduler.speedup_by_fusion = speedup_by_fusion
    Scheduler.fuse_nodes_once = fuse_nodes_once
