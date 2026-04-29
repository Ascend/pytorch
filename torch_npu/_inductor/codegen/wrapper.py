import os
import copy
from typing import Optional, Union
import contextlib
from torch._inductor import config
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SymbolicCallArg,
    pexpr,
)
from torch._inductor.utils import (
    cache_on_self,
)
from torch._inductor.virtualized import V
from torch._inductor.ir import GraphPartitionSignature, TorchBindObject, NoneLayout
from torch._dynamo.utils import counters
from torch._inductor.codegen.common import DeferredLine, WorkspaceArg, IndentedBuffer
from torch._inductor.codegen.wrapper import BufferLike, WrapperLine
from torch._inductor import ir
import torch_npu.npu.aclnn
from ..fx_passes.utils.schedule_node_utils import is_multi_stream


class NPUWrapperCodeGen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        self.buffer_args_multi_stream_intent = {}
        self.buffer_define_multi_stream = {}
        self.extern_node_intent_multi_stream = []
        self.pre_define_buffer = []


    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            return super().create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures)
        return NPUWrapperCodeGen()

    @cache_on_self
    def write_triton_header_once(self) -> None:
        super().write_triton_header_once()
        import_str = f"""
            import torch_npu
            has_initialized = False
        """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.splice(
                "import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics"
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.splice(
                "import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics"
            )

    # generate numel expr for range_tree_node
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        self.writeline(f"{expr} = {pexpr(numel_expr)}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, numel_expr)

    # don't assert
    def codegen_input_size_asserts(self) -> None:
        pass

    def get_next_kernel_suffix(self) -> str:
        iter_val = copy.copy(self._names_iter)
        return f"{next(iter_val)}"

    def write_prefix(self) -> None:
        super().write_prefix()
        if torch_npu.npu.aclnn._use_static_aclnn_kernel:
            with self.prefix.indent():
                self.prefix.writeline('global has_initialized')
                self.prefix.writeline('if not has_initialized:')
            self.prefix.do_indent()
            with self.prefix.indent():
                self.prefix.writeline('from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler')
                self.prefix.writeline('static_kernel_complier = StaticKernelCompiler()')
                self.prefix.writeline('static_kernel_complier.__enter__()')
                self.prefix.writeline('has_initialized = True')
            self.prefix.do_unindent()

    def generate_return(self, output_refs: list[str]) -> None:
        if torch_npu.npu.aclnn._use_static_aclnn_kernel:
            self.wrapper_call.do_unindent()
            with self.wrapper_call.indent():
                self.wrapper_call.writeline('if not has_initialized:')
            self.wrapper_call.do_indent()
            with self.wrapper_call.indent():
                self.wrapper_call.writeline('exc_info=(None, None, None)')
                self.wrapper_call.writeline('static_kernel_complier.__exit__(*exc_info)')
        super().generate_return(output_refs)


    def get_buffer_define_multi_stream_by_name(self, name):
        multi_stream_intent_str = ""
        if name and self.buffer_define_multi_stream is not None and name in self.buffer_define_multi_stream.keys():
            multi_stream_intent_str = " " * self.buffer_define_multi_stream.get(name)
        return multi_stream_intent_str


    def make_buffer_free(self, buffer: Union[BufferLike, TorchBindObject]):
        if is_multi_stream():
            multi_stream_intent = ""
            if hasattr(buffer, "multi_stream_intent"):
                multi_stream_intent = " " * getattr(buffer, "multi_stream_intent")
            elif self.buffer_args_multi_stream_intent is not None and buffer.get_name() in self.buffer_args_multi_stream_intent.keys():
                multi_stream_intent = " " * self.buffer_args_multi_stream_intent.get(buffer.get_name())
            return f"{multi_stream_intent}del {buffer.get_name()}"
        return super().make_buffer_free(buffer)


    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None
    ):
        out = super().make_allocation(name, device, dtype, shape, stride, allocation_shape)
        if is_multi_stream():
            if name in self.pre_define_buffer:
                return ""
            else:
                multi_stream_intent_str = self.get_buffer_define_multi_stream_by_name(name)
                out = f"{multi_stream_intent_str}{out}"
        return out


    def generate_extern_kernel_out(
        self,
        kernel: str,
        out: str,
        out_view: Optional[str],
        args: list[str],
        device: str,
    ) -> None:
        if is_multi_stream():
            multi_stream_intent = ""
            if self.extern_node_intent_multi_stream is not None and len(self.extern_node_intent_multi_stream) > 0:
                for node_intent in self.extern_node_intent_multi_stream:
                    python_kernel_name = node_intent.get("python_kernel_name")
                    kernel_args = list(node_intent.get("args"))
                    stream_intent = node_intent.get("multi_stream_intent")
                    args_str = str(args)
                    is_sub_args = all(arg in args_str for arg in kernel_args)
                    if is_sub_args and python_kernel_name == kernel:
                        multi_stream_intent = " " * stream_intent
            # add debug printer code for triton kernel calls at (jit) inductor level
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
            args.append(f"out={out_view if out_view else out}")
            with debug_printer_manager:
                self.writeline(f"{multi_stream_intent}{kernel}({', '.join(args)})")
        else:
            super().generate_extern_kernel_out(kernel, out, out_view, args, device)


    def generate_extern_kernel_alloc(self, extern_kernel, args):
        if is_multi_stream():
            no_return = isinstance(extern_kernel.layout, NoneLayout)
            output_name = extern_kernel.get_name()
            origin_node = extern_kernel.get_origin_node()
            kernel_name = extern_kernel.get_kernel_name()
            ending = self.ending
            if config.memory_planning and "view_as_complex" in kernel_name:
                ending = f".clone(){ending}"
            multi_stream_intent_str = self.get_buffer_define_multi_stream_by_name(output_name)
            if no_return:
                self.writeline(f"{multi_stream_intent_str}{self.declare}{kernel_name}({', '.join(args)}){ending}")
            else:
                self.writeline(
                    f"{multi_stream_intent_str}{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}"
                )
                if (
                    self.supports_intermediate_hooks
                    and config.generate_intermediate_hooks
                    and origin_node is not None
                ):
                    counters["inductor"]["intermediate_hooks"] += 1
                    self.writeline(
                        f"{multi_stream_intent_str}run_intermediate_hooks({origin_node.name!r}, {output_name})"
                    )
        else:
            super().generate_extern_kernel_alloc(extern_kernel, args)


    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        origin_node=None,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_args=None,
        triton_meta=None,
    ):
        if is_multi_stream():
            """
            Generates kernel call code.

            triton: Defines whether the backend uses Triton for codegen. Otherwise it uses the CUDA language when gpu=True,
                    and C++ when gpu=False.
            """
            device = device or V.graph.get_current_device_or_throw()
            if not (triton or device.type != "cpu"):
                self.writeline(self.wrap_kernel_call(kernel_name, call_args))
                return
            call_args_str = self.prepare_triton_kernel_call(call_args)
            call_args_str = ", ".join(call_args_str)
            if origin_node is not None and hasattr(origin_node, "multi_stream_name"):
                stream_name = getattr(origin_node, "multi_stream_name")
                multi_stream_intent = " " * getattr(origin_node, "multi_stream_intent")
            else:
                multi_stream_intent = ""
                stream_name = PythonWrapperCodegen.write_get_raw_stream(
                    self, device.index, V.graph
                )

            if not triton:
                stream_ptr = f"c_void_p({stream_name})"
                self.writeline(
                    f"{multi_stream_intent}{kernel_name}.{kernel_name}({call_args_str}, {stream_ptr})"
                )
                return

            self.write_triton_header_once()

            if (
                config.triton.autotune_at_compile_time
                and kernel_name not in self.kernel_autotune_names
            ):
                # Create example args for autotune in a separate epilogue
                if not (arg_types is not None and len(call_args) == len(arg_types)):
                    raise AssertionError("call_args and arg_types do not match")
                tensor_args = {}
                all_args = []
                if raw_args is None:
                    # create a dummy raw_args for uniform behavior in the following loop
                    raw_args = [None] * len(call_args)
                else:
                    if not (len(raw_args) == len(call_args)):
                        raise AssertionError("call_args and raw_args do not match")

                for i, (arg, arg_type, raw_arg) in enumerate(
                    zip(call_args, arg_types, raw_args)
                ):
                    key = None
                    if isinstance(arg, str) and "=" in str(arg):
                        # arg may be passed in a kwarg style, and then we need to extract its value
                        key, arg = arg.split("=")
                    from torch import dtype as torch_dtype
                    import re
                    if isinstance(arg_type, torch_dtype):
                        # workspace allocation is already generated by `generate_workspace_allocation()`
                        # in `TritonKernel.call_kernel()`.
                        if re.match(r"^(workspace|semaphore)", arg):
                            arg_str = arg
                            tensor_args[arg] = arg_str
                        elif arg not in tensor_args:
                            arg_str = self.generate_example_arg_value(
                                arg, arg_type, raw_arg, i
                            )
                            tensor_args[arg] = arg_str
                        else:
                            arg_str = tensor_args[arg]
                    else:
                        arg_str = self.generate_example_arg_value(arg, arg_type, raw_arg, i)
                    all_args.append(arg_str if key is None else f"{key}={arg_str}")
                self.kernel_autotune_calls.writeline(
                    f"{multi_stream_intent}{kernel_name}.run({', '.join(all_args)}, stream={stream_name})"
                )
                self.kernel_autotune_calls.writeline(
                    f"{multi_stream_intent}del {', '.join(arg for arg in tensor_args.values())}\n",
                )
                self.kernel_autotune_names.add(kernel_name)
                if V.graph.cpp_wrapper:
                    # For cpp wrapper, no need to continue codegen for the main body
                    return

            # add debug printer code for triton kernel calls at (jit) inductor level
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(call_args, kernel_name, arg_types, None)
            with debug_printer_manager:
                self.writeline(f"{multi_stream_intent}{kernel_name}.run({call_args_str}, stream={stream_name})")
        else:
            super().generate_kernel_call(kernel_name, call_args, device=device, triton=triton, arg_types=arg_types, raw_args=raw_args, triton_meta=triton_meta)


    def codegen_multi_output(self, name, value):
        if is_multi_stream():
            multi_stream_intent = ""
            if self.buffer_define_multi_stream is not None and name in self.buffer_define_multi_stream.keys():
                multi_stream_intent = " " * self.buffer_define_multi_stream.get(name)
            self.writeline(f"{multi_stream_intent}{self.declare}{name} = {value}{self.ending}")
        else:
            super().codegen_multi_output(name, value)


    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        if is_multi_stream():
            if not (old.get_dtype() == new.get_dtype()):
                raise AssertionError("old dtype not equels new dtype")
            old_name = old.get_name()
            new_name = new.get_name()
            del_line = ";"
            if old_name not in V.graph.get_output_names() and delete_old:
                del_line = f"; {self.make_buffer_free(old)}"
            multi_stream_intent_str = self.get_buffer_define_multi_stream_by_name(new_name)
            if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
                return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

            reinterpret_view = self.codegen_reinterpret_view(
                old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
            )
            return f"{multi_stream_intent_str}{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"
        return super().make_buffer_reuse(old, new, delete_old)


    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        if is_multi_stream():
            multi_stream_intent_str = self.get_buffer_define_multi_stream_by_name(new_name)
            return f"{multi_stream_intent_str}{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"
        return super().codegen_exact_buffer_reuse(old_name, new_name, del_line)
    
    
    def codegen_deferred_allocation(self, name: str, view: ir.ReinterpretView) -> None:
        if is_multi_stream():
            multi_stream_intent_str = self.get_buffer_define_multi_stream_by_name(name)
            self.writeline(
                DeferredLine(
                    name,
                    f"{multi_stream_intent_str}{self.declare}{name} = {view.codegen_reference()}{self.ending}  {self.comment} alias",
                )
            )
        else:
            super().codegen_deferred_allocation(name, view)


    def get_node_multi_stream_intent(self, origin_node):
        multi_stream_intent = -1
        if origin_node is not None and hasattr(origin_node, "multi_stream_intent"):
            multi_stream_intent = getattr(origin_node, "multi_stream_intent")
        return multi_stream_intent


    def update_buffer_define_multi_stream(self, name, origin_node):
        multi_stream_intent = self.get_node_multi_stream_intent(origin_node)
        if multi_stream_intent != -1:
            self.buffer_define_multi_stream[name] = multi_stream_intent


    def update_buffer_args_multi_stream_intent(self, name, origin_node):
        multi_stream_intent = self.get_node_multi_stream_intent(origin_node)
        if multi_stream_intent != -1:
            self.buffer_args_multi_stream_intent[name] = multi_stream_intent


    def generate_workspace_allocation(self, ws: WorkspaceArg, origin_node):
        if is_multi_stream():
            self.update_buffer_define_multi_stream(ws.get_name(), origin_node)
        super().generate_workspace_allocation(ws)


    def generate_workspace_deallocation(self, ws: WorkspaceArg, origin_node):
        if is_multi_stream():
            self.update_buffer_args_multi_stream_intent(ws.get_name(), origin_node)
        super().generate_workspace_deallocation(ws)


    def _generate(self, is_inference):
        if is_multi_stream():
            if config.profile_bandwidth:
                self.write_triton_header_once()
            result = IndentedBuffer()
            result.splice(self.imports)
            result.writeline("")
            result.splice(self.header)
            # We do not want the cpp header for intermediate const graph. Headers would be
            # rendered by the main module instead.
            if V.graph.aot_mode and V.graph.cpp_wrapper and V.graph.is_const_graph:
                result = IndentedBuffer()

            # Add subgraph definitions to the result
            result.splice(self.subgraph_definitions)

            with contextlib.ExitStack() as stack:
                stack.enter_context(self.wrapper_call.indent())
                if config.profiler_mark_wrapper_call:
                    self.generate_profiler_mark_wrapper_call(stack)
                if config.profile_bandwidth:
                    self.generate_start_graph()

                # We disable planning during training because it presently increases peak memory consumption.
                if is_inference and config.memory_planning:
                    self.memory_plan()
                else:
                    self.memory_plan_reuse()

                if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                    self.generate_reset_kernel_saved_flags()
                stream_line = -1
                for idx, line in enumerate(self.lines):
                    if isinstance(line, str) and "main_stream = " in line:
                        stream_line = idx
                        break
                
                if stream_line == -1:
                    for line in self.lines:
                        if isinstance(line, WrapperLine):
                            line.codegen(self.wrapper_call)
                        else:
                            self.wrapper_call.writeline(line)
                else:
                    self.handle_cross_stream_del_buf()
                    buffer_define_multi_stream = self.buffer_define_multi_stream
                    for idx, line in enumerate(self.lines):
                        if idx < stream_line:
                            self.buffer_define_multi_stream = {}
                        else:
                            self.buffer_define_multi_stream = buffer_define_multi_stream
                        if isinstance(line, WrapperLine):
                            line.codegen(self.wrapper_call)
                        else:
                            self.wrapper_call.writeline(line)

                output_refs = self.get_output_refs()
                self.mark_output_type()
                if config.triton.debug_sync_graph:
                    self.wrapper_call.writeline(V.graph.device_ops.synchronize())

                if config.profile_bandwidth:
                    self.generate_end_graph()

                if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                    self.generate_save_uncompiled_kernels()

                if config.triton.autotune_at_compile_time:
                    self.generate_and_run_autotune_block()

                # cpp_wrapper currently doesn't support nvtx
                if config.annotate_training and not config.cpp_wrapper:
                    self.wrapper_call.writeline(
                        "nvtx._device_range_end(training_annotation)"
                    )
                self.generate_return(output_refs)

            self.finalize_prefix()
            result.splice(self.prefix)

            wrapper_call_indent = self.get_wrapper_call_indent()

            with result.indent(wrapper_call_indent):
                result.splice(self.wrapper_call)

            self.generate_before_suffix(result)
            result.splice(self.suffix)
            self.generate_after_suffix(result)

            self.generate_end(result)

            self.add_benchmark_harness(result)

            return (
                result.getvaluewithlinemap(),
                self.kernel_declarations.getvaluewithlinemap(),
            )
        return super()._generate(is_inference)
    
    
    def handle_cross_stream_del_buf(self):
        total_lines = len(self.lines)
        sub_streams_line_no = self.get_sub_streams_line_no()
        for idx, line in enumerate(self.lines):
            if len(self.buffer_args_multi_stream_intent) <= 0:
                continue
            for sub_stream_line in sub_streams_line_no:
                keys = list(self.buffer_args_multi_stream_intent.keys())
                tab_value = self.buffer_args_multi_stream_intent[keys[0]]
                if idx > sub_stream_line[0] and idx < sub_stream_line[1] and isinstance(line, WrapperLine) and hasattr(line, "node") and line.node.get_name() not in self.buffer_args_multi_stream_intent.keys():
                    self.buffer_args_multi_stream_intent[line.node.get_name()] = tab_value
            
            n = len(sub_streams_line_no)
            for i in range(1, n):
                prev_end = sub_streams_line_no[i-1][1]
                curr_start = sub_streams_line_no[i][0]
                
                if prev_end < idx < curr_start and isinstance(line, WrapperLine) and hasattr(line, "node") and line.node.get_name() in self.buffer_args_multi_stream_intent.keys():
                    self.buffer_args_multi_stream_intent.pop(line.node.get_name(), None)

            if n > 0:
                last_end = sub_streams_line_no[-1][1]
                if last_end < idx < total_lines and isinstance(line, WrapperLine) and hasattr(line, "node") and line.node.get_name() in self.buffer_args_multi_stream_intent.keys():
                    self.buffer_args_multi_stream_intent.pop(line.node.get_name(), None)
    
    
    def get_sub_streams_line_no(self):
        sub_streams_line_no = []
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            if isinstance(line, str) and "with torch_npu.npu.stream" in line:
                start_idx = i
                stream_name = ""
                if "(" in line:
                    try:
                        stream_name = line.split("(", 1)[1].split(")", 1)[0].strip()
                    except:
                        pass
                end_idx = None
                for j in range(i + 1, len(self.lines)):
                    next_line = self.lines[j]
                    if isinstance(next_line, str):
                        if stream_name and f"record({stream_name})" in next_line:
                            end_idx = j
                            break
                        elif not stream_name and "record(" in next_line:
                            end_idx = j
                            break
                if end_idx is not None:
                    sub_streams_line_no.append([start_idx, end_idx])
                    i = end_idx + 1
                    continue
            i += 1
        return sub_streams_line_no