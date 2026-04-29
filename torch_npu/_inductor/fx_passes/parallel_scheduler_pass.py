import torch
import torch_npu
from torch._inductor import config
from torch._inductor.utils import sympy_product
from torch._inductor.ir import ExternKernelOut, InputBuffer, ReinterpretView
import os
from torch._inductor.virtualized import V
from torch._inductor.utils import device_need_guard
from torch.utils._ordered_set import OrderedSet
import traceback
from ..config import log
import typing
from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode, SchedulerNode, NopKernelSchedulerNode, ForeachKernelSchedulerNode, ExternKernelSchedulerNode
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.simd import SIMDScheduling
from collections import defaultdict
from .parallelism_strategy_framework import ParallelGroupingStrategy
from .utils.fx_pass_level import GroupType
import torch._inductor.ir as ir
from ..codegen.catlass.catlass_kernel import CATLASSTemplateBuffer


def parallel_scheduler():
    original_codegen = torch._inductor.scheduler.Scheduler._codegen
    
    def patched_codegen(self, nodes: list[BaseSchedulerNode]) -> None:
        parallel_strategy = ParallelGroupingStrategy()
        groups = parallel_strategy.execute_strategy(nodes)
        is_parallel = True
        for key, group_nodes in groups.items():
            if len(group_nodes) == 0:
                is_parallel = False
                break
        if len(groups) < 3 or not is_parallel:
            original_codegen(self, nodes)
            return
        if config.check_stack_no_cycles_TESTING_ONLY:
            import torch._dynamo.convert_frame
            stack = traceback.extract_stack()
            seen = OrderedSet()
            for frame in reversed(stack):
                if (
                    frame.name == "_compile_inner"
                    and frame.filename == torch._dynamo.convert_frame.__file__
                ):
                    break
                key = (frame.filename, frame.lineno)

                if key in seen:
                    raise AssertionError(f"Duplicate stack frame {frame.filename}:{frame.lineno}; "
                    "did you add a decorator to one of the functions in this stack "
                    "trace?  If so, try using a context manager instead.")
                seen.add(key)

        self.current_device = None
        for node in nodes:
            try:
                log.debug(
                    "Generating code for node %s with estimated runtime %f",
                    node.get_name(),
                    node.get_estimated_runtime(),
                )
            except Exception:
                log.debug(
                    "Generating code for node %s with estimated runtime 0.0",
                    node.get_name(),
                )

        wrapper = V.graph.wrapper_code
        wrapper.pre_define_buffer = []

        stream_vars = {
            key: f"stream_group_{key.lower()}" 
            for key in list(groups.keys())
        }
        event_vars = {
            key: f"event_group_{key.lower()}" 
            for key in list(groups.keys())
        }
        stream_codegen(wrapper, groups, stream_vars, event_vars)

        node_to_group_id = {}
        for group_id, group in groups.items():
            for node in group:
                node_to_group_id[node.get_name()] = group_id

        current_indent_level = 0
        current_stream = "main_stream"
        current_event = None

        pre_node_multi_stream_flag(nodes, node_to_group_id, stream_vars)

        group_to_buffer = build_buffer_producer_group(nodes, node_to_group_id)

        waited_events = set()
        need_main_stream_event = []
        for node in nodes:
            node_name = node.get_name()
            group_id = node_to_group_id.get(node_name)
            is_main_stream = group_id == GroupType.MAIN.name
            target_stream = "main_stream" if is_main_stream else stream_vars[group_id]
            target_event = None if is_main_stream else event_vars[group_id]

            self.enter_context(node)
            device = node.get_device()
            if device:
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if self.current_device and device_need_guard(self.current_device.type):
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    self.current_device = device
                    if device_need_guard(device.type):
                        if device.index is None:
                            raise AssertionError("device.index is None")
                        V.graph.wrapper_code.codegen_device_guard_enter(device.index)
            self.buffer_names_to_free.update(node.last_usage)
            tab_value = current_indent_level * wrapper.wrapper_call.tabwidth
            
            if target_stream != current_stream:
                if current_stream and current_stream != "main_stream" and current_event:
                    wrapper.writeline(" " * (tab_value) + f"{current_event}.record({current_stream})")
                    need_main_stream_event.append(current_event)

                    while current_indent_level > 0:
                        current_indent_level -= 1

                if target_stream != "main_stream":
                    add_need_pre_buf_define(group_to_buffer, group_id, wrapper)
                    wrapper.writeline(f"with torch_npu.npu.stream({target_stream}):")
                    current_indent_level += 1
                    wrapper.writeline(" " * (current_indent_level * wrapper.wrapper_call.tabwidth) + f"{target_stream}.wait_event(main_event)")

                if target_stream == "main_stream":
                    for event_name in need_main_stream_event:
                        if event_name not in waited_events:
                            intent_tab = 0
                            wrapper.writeline(
                                " " * (intent_tab * wrapper.wrapper_call.tabwidth) +
                                f"{target_stream}.wait_event({event_name})"
                            )
                            waited_events.add(event_name)

                current_stream = target_stream
                current_event = target_event

            tab_value = current_indent_level * wrapper.wrapper_call.tabwidth
            build_multi_stream_buf_intent(node, tab_value, wrapper)

            if node.is_template():
                prologue, template_node, epilogue = node.get_prologue_template_epilogue(list(node.get_nodes()))
                self.get_backend(device).codegen_template(template_node, epilogue, prologue)
            elif node.is_extern():
                node = typing.cast(ExternKernelSchedulerNode, node)
                self.codegen_extern_call(node)
            elif node.is_foreach():
                node = typing.cast(ForeachKernelSchedulerNode, node)
                backend_ = self.get_backend(device)
                if isinstance(backend_, (SIMDScheduling, CUDACombinedScheduling)):
                    backend = backend_
                else:
                    raise AssertionError(f"{type(self)=}")
                backend.codegen_combo_kernel(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_node(node)
            else:
                if not (isinstance(node, NopKernelSchedulerNode)):
                    raise AssertionError("node type is not NopKernelSchedulerNode")
                node.mark_run()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_buffer_names())
            self.completed_operations.update(node.get_operation_names())

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if device is not None and self.get_backend(device).ready_to_flush():
                    self.flush()
        if current_stream != "main_stream" and current_event:
            wrapper.writeline(
                " " * (current_indent_level * wrapper.wrapper_call.tabwidth) +
                f"{current_event}.record({current_stream})"
            )
            wrapper.writeline(
                " " * (current_indent_level * wrapper.wrapper_call.tabwidth) +
                f"# end last stream {current_stream} context"
            )

        while current_indent_level > 0:
            current_indent_level -= 1

        wrapper.writeline("\n# Wait for all parallel streams to complete")
        for key, event_name in event_vars.items():
            if event_name not in waited_events and event_name != event_vars[GroupType.MAIN.name]:
                wrapper.writeline(f"main_stream.wait_event({event_name})")    

        if self.current_device and device_need_guard(self.current_device.type):
            V.graph.wrapper_code.codegen_device_guard_exit()
        self.flush()

    torch._inductor.scheduler.Scheduler._codegen = patched_codegen

    original_codegen_assert = torch._inductor.ir.ExternKernel.codegen_size_asserts
    def patched_codegen_size_asserts(self, wrapper) -> None:
        if hasattr(wrapper, "buffer_define_multi_stream"):
            buffer_define_multi_stream = getattr(wrapper, "buffer_define_multi_stream")
            if config.size_asserts and not V.graph.cpp_wrapper:
                if sympy_product(self.get_size()) == 0:
                    return
                size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
                stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
                multi_stream_intent = ""
                if self.get_name() in buffer_define_multi_stream.keys():
                    multi_stream_intent = " " * buffer_define_multi_stream.get(self.get_name())
                wrapper.writeline(
                    f"{multi_stream_intent}assert_size_stride({self.get_name()}, {size}, {stride})"
                )
        else:
            original_codegen_assert(self, wrapper)
    torch._inductor.ir.ExternKernel.codegen_size_asserts = patched_codegen_size_asserts


def stream_codegen(wrapper, groups, stream_vars, event_vars):
    groups_keys = list(groups.keys())
    wrapper.writeline("import torch_npu")
    wrapper.writeline("main_stream = torch_npu.npu.current_stream()")
    wrapper.writeline("main_event = torch_npu.npu.Event()")
    wrapper.writeline("main_event.record(main_stream)")
    for i in groups_keys:
        if i != GroupType.MAIN.name:
            wrapper.writeline(f"{stream_vars[i]} = torch_npu.npu.Stream()")
            wrapper.writeline(f"{event_vars[i]} = torch_npu.npu.Event()")


def add_need_pre_buf_define(group_to_buffer, group_id, wrapper):
    need_define_buffers = group_to_buffer[group_id]
    for buf in need_define_buffers:
        if isinstance(buf, (ir.ComputedBuffer, ir.ExternKernelOut, CATLASSTemplateBuffer)) and hasattr(buf, "name") and buf.name not in V.graph.removed_buffers:
            line = wrapper.make_buffer_allocation(buf)
            wrapper.writeline(line)
            wrapper.pre_define_buffer.append(buf.name)


def build_multi_stream_buf_intent(node, tab_value, wrapper):
    if not hasattr(node, "multi_stream_intent"):
        setattr(node, "multi_stream_intent", tab_value)
        if hasattr(node, 'get_nodes'):
            nodes = node.get_nodes()
            for n in nodes:
                if not hasattr(n, "multi_stream_intent"):
                    setattr(n, "multi_stream_intent", tab_value)
    
    if getattr(node, "multi_stream_name", None) not in (None, "main_stream.npu_stream"):
        if hasattr(node, "node") and isinstance(node.node, ExternKernelOut) and getattr(node.node, "python_kernel_name", None) is not None:
            args = set()
            inputs = getattr(node.node, "inputs", None)
            if inputs:
                for input in inputs:
                    if isinstance(input, InputBuffer):
                        args.add(input.name)
                    elif isinstance(input, ReinterpretView):
                        args.add(input.get_name())
            data = {
                "python_kernel_name": getattr(node.node, "python_kernel_name"),
                "args": args,
                "multi_stream_intent": tab_value
            }
            wrapper.extern_node_intent_multi_stream.append(data)
        if len(node.last_usage) > 0:
            for buf in node.last_usage:
                wrapper.buffer_args_multi_stream_intent[buf] = tab_value

        node_outputs = getattr(node, "outputs", None)
        if node_outputs:
            for output in node_outputs:
                wrapper.buffer_define_multi_stream[output.node.name] = tab_value
        elif getattr(node, "node", None) is None and hasattr(node, "snodes"):
            snodes = getattr(node, "snodes")
            for sn in snodes:
                sn_outputs = getattr(sn, "outputs", None)
                if sn_outputs:
                    for output in sn_outputs:
                        wrapper.buffer_define_multi_stream[output.node.name] = tab_value


def pre_node_multi_stream_flag(nodes, node_to_group_id, stream_vars):
    for node in nodes:
        node_name = node.get_name()
        group_id = node_to_group_id.get(node_name)

        if group_id == GroupType.MAIN.name:
            target_stream = "main_stream"
        else:
            target_stream = stream_vars[group_id]
        if not hasattr(node, "multi_stream_name"):
            setattr(node, "multi_stream_name", f"{target_stream}.npu_stream")
            if hasattr(node, 'get_nodes'):
                nodes = node.get_nodes()
                for n in nodes:
                    if not hasattr(n, "multi_stream_name"):
                        setattr(n, "multi_stream_name", f"{target_stream}.npu_stream")


def build_buffer_producer_group(nodes, node_to_group_id):
    buffer_producer_group = {}
    all_buffers = []
    for node in nodes:
        gid = node_to_group_id.get(node.get_name())

        outputs = getattr(node, "outputs", None)
        if outputs:
            for out in outputs:
                buffer_producer_group[out.node.name] = gid
                all_buffers.append(out.node)

        if getattr(node, "node", None) is None and hasattr(node, "snodes"):
            for sn in node.snodes:
                sn_outputs = getattr(sn, "outputs", None)
                if sn_outputs:
                    for out in sn_outputs:
                        buffer_producer_group[out.node.name] = gid
                        all_buffers.append(out.node)
    group_to_buffer = defaultdict(list)
    for buf_name, gid in buffer_producer_group.items():
        for buf in all_buffers:
            if hasattr(buf, "name") and buf.name == buf_name:
                group_to_buffer[gid].append(buf)
    return group_to_buffer