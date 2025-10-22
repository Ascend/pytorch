import os
import copy
import sympy
import collections
import textwrap

import torch
from sympy import Expr, Integer
from itertools import count
from torch._dynamo.utils import counters

from typing import List, Union, Optional, Tuple, Any, Dict

from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.simd import (
    log,
    OrderedSet,
    EnableReduction,
    DisableReduction,
    SIMDKernel,
    SIMDKernelFeatures,
    MultiKernel,
    code_hash
)
from torch._inductor.codegen.triton import (
    SIMDScheduling,
    FixedTritonConfig,
)

from torch._inductor.ir import IRNode

from torch._inductor.codegen.common import (
    IndentedBuffer,
    Kernel,
)

from torch._inductor.codegen.triton import (
    TritonKernel
)

from torch._inductor.utils import (
    get_fused_kernel_name,
)
from torch._inductor import config, ir, scheduler
from ... import config as anir_config
from torch._inductor.virtualized import V
from ...npu.utils import (
    MLIRProcessor,
    parse_fx_example_inputs,
    fx_graph_op_types,
    npu_cast_to_prim_cast,
    get_fx_graph_code,
    scalarize_tensor_ops_on_scalars,
    to_folder,
    modify_gm_for_acc_comp,
    get_num_call_functions,
    is_fx_dynamic,
    view_to_reshape
)

if anir_config.enable_graph_trace:
    from ...npu.inductor_patch.lowering import (
        merge_fx_graphs,
        map_strings_to_operators
    )

id_iter = count()

class NpuMlirKernel(Kernel):
    def __init__(self, 
                 gm: torch.fx.GraphModule, 
                 snodes: List[scheduler.SchedulerNode], 
                 call_args: List[str],
                 non_contiguous_indices: List[int],
                 num_outputs: List[int]=None,
                 mutated_indices: List[int]=None
                 ):
        super().__init__()
        self._gm = gm
        self._gm_with_prim_cast = self.build_gm_with_prim_cast(gm)
        self._is_dynamic = is_fx_dynamic(self._gm)
        if anir_config.online_acc_comp:
            modify_gm_for_acc_comp(self._gm)
        self._snodes = snodes
        self._call_args = call_args
        self.non_contiguous_indices = non_contiguous_indices
        self.num_outputs = num_outputs
        self.mutated_indices = mutated_indices
        

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )
    
    def build_gm_with_prim_cast(self, gm):
        gm_with_prim_cast = npu_cast_to_prim_cast(gm)
        return gm_with_prim_cast

    def codegen_kernel(self, name=None):
        
        code = IndentedBuffer()

        import torch_mlir
        from ..torch_mlir_patch import stateless_fx_import
        from torch._functorch.aot_autograd import (
            set_model_name,
            get_aot_compilation_context,
        )
        from torch_mlir.compiler_utils import (
            run_pipeline_with_repro_report,
            lower_mlir_module,
        )
        from torch_mlir.compiler_utils import OutputType

        scalarize_tensor_ops_on_scalars(self._gm_with_prim_cast)
        set_model_name(f'MODEL_NAME')
        *_, model_name, nth_graph = get_aot_compilation_context()
        mlir_module = stateless_fx_import(
            self._gm_with_prim_cast, 
            model_name=model_name,
            import_symbolic_shape_expressions=False)
        run_pipeline_with_repro_report(
            mlir_module,
            # f"builtin.module(torch-function-to-torch-backend-pipeline{option_string})",
            f"builtin.module(torch-lower-to-backend-contract)",
            "Lowering TorchFX IR -> Torch Backend IR",
        )

        with mlir_module.context:
            for func in mlir_module.body.operations:
                if isinstance(func, torch_mlir.dialects.func.FuncOp):
                    func.attributes["torch.assume_strict_symbolic_shapes"] = torch_mlir.ir.UnitAttr.get()

        code.splice(f'{str(mlir_module)}')

        return code.getvalue()
    
    def get_call_args(self):
        return self._call_args

    def call_kernel(self, name: str, node: Optional[IRNode] = None):
        wrapper = V.graph.wrapper_code
        call_args = self.get_call_args()
        for call_arg in call_args:
            if call_arg.startswith('_uwu_'):
                expression = map_strings_to_operators(call_arg)
                wrapper.writeline(f'{call_arg} = {expression}')
        if len(call_args) > 0:
            wrapper.generate_kernel_call(
                name,
                call_args,
            )

    def codegen_debug_performance(self, fd):
        from ...npu.utils import generate_compiler_repro_string, generate_fake_inputs
        name_to_example_inputs = parse_fx_example_inputs(self._gm)
        call_args_str = ", ".join(list(name_to_example_inputs.keys()))
        fd.write(
            generate_compiler_repro_string(
                self._gm,
            )
        )
        fd.write("\n")
        fd.write("if __name__ == '__main__':\n")
        fd.write("    from torch._inductor.utils import print_performance\n")
        fd.write(f"    with torch.no_grad():\n"
        )
        fd.write(generate_fake_inputs(name_to_example_inputs))
        fd.write('\n')
        fd.write(
            f"        fn = lambda: mod({call_args_str})\n"
            f"        print_performance(fn, times=10, repeat=10)\n"
        )
    

class NpuTritonKernel(TritonKernel):
    def __init__(self, 
            tiling: Dict[str, sympy.Expr],
            min_elem_per_thread=0,
            optimize_mask=True,
            fixed_config: Optional[FixedTritonConfig] = None,
            **kwargs,
            ):
        super().__init__(
            tiling,
            min_elem_per_thread=min_elem_per_thread,
            optimize_mask=optimize_mask,
            fixed_config = fixed_config,
            **kwargs,
        )

    @staticmethod
    def inductor_meta_common():
        return {}
    
    def call_kernel(self, call_args, name: str):
        wrapper = V.graph.wrapper_code
        for call_arg in call_args:
            if call_arg.startswith('_uwu_'):
                expression = map_strings_to_operators(call_arg)
                wrapper.writeline(f'{call_arg} = {expression}')
        if len(call_args) > 0:
            wrapper.generate_kernel_call(
                name,
                call_args,
            )

def find_common_positions(list1, list2):
    common_elements = set(list1) & set(list2)
    merged_list = list1 + list2
    positions = [index for index, element in enumerate(merged_list) if element in common_elements]
    
    return merged_list, positions

def create_fx_from_snodes_by_traced_graph(snodes: List[scheduler.SchedulerNode], triton_kernel: TritonKernel):
    call_inputs = []
    for snode in snodes:
        snode.node.data.traced_graph.last_node.name = snode.node.get_name()
    if len(snodes) == 1:
        traced_graph = snodes[0].node.data.traced_graph
    else:
        traced_graph = merge_fx_graphs([snode.node.data.traced_graph for snode in snodes])
    inputs = []
    for node in traced_graph.graph.nodes:
        if node.op == 'placeholder':
            call_inputs.append(node.target)
            inputs.append(node.meta['val'])
    non_contiguous_indices = {}
    non_contiguous_indices["inputs"] = [i for i, inp in enumerate(inputs) if torch.is_tensor(inp) and not inp.is_contiguous()]
    num_inputs = len(call_inputs)
    call_outputs = []
    for snode in snodes:
        if snode.has_aliasing_or_mutation():
            for buf in snode.get_outputs():
                if len(buf.get_mutations()):
                    call_outputs.extend(buf.get_mutations())
                elif len(buf.get_aliases()):
                    call_outputs.append(buf.get_name())
        elif snode.node.get_name() not in (V.graph.removed_buffers | V.graph.inplaced_to_remove):
            call_outputs.append(snode.node.get_name())
    num_outputs = len(call_outputs)
    call_args, mutated_indices = find_common_positions(call_inputs, call_outputs)
    outputs = traced_graph.last_node if isinstance(traced_graph.last_node, List) \
        else [traced_graph.last_node]
    outputs = [output for output in outputs if output.name not in (V.graph.removed_buffers | V.graph.inplaced_to_remove)]
    traced_graph.graph.output(tuple(outputs))
    traced_graph.graph.lint()
    orig_module = torch.nn.Module()
    gm = torch.fx.GraphModule(orig_module, traced_graph.graph)
    gm.recompile()
    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)
    with V.graph.fake_mode: 
        gm = make_fx(runnable_gm)(*inputs)
    view_to_reshape(gm)
    non_contiguous_indices["outputs"] = [i + num_inputs for i, call_output in enumerate(call_outputs) \
                                         if not V.graph.try_get_buffer(call_output).layout.is_contiguous()]
    return (gm, call_args, {"num_outputs": num_outputs, 
                            "non_contiguous_indices": non_contiguous_indices, 
                            "mutated_indices": mutated_indices,})


class NpuMlirScheduling(SIMDScheduling):

    kernel_type = NpuTritonKernel

    def __init__(self, scheduler: Scheduler):
        super().__init__(scheduler)
        self.orig_fnode_name_to_fnode = {}

    def define_kernel(self, src_code, mlir_kernel, traced_graph, mode=None):
        if mode is None:
            mode = anir_config._get_compile_mode()
        kernel_key = (src_code, tuple(mlir_kernel.non_contiguous_indices))
        wrapper = V.graph.wrapper_code

        if kernel_key in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[kernel_key]
        else:
            fused_kernel_name = get_fused_kernel_name(mlir_kernel._snodes, config.triton.descriptive_names)
            if mode in ["complete_fallback", "auto_fallback"]:
                fx_graph_suffix = f"{next(id_iter)}"
            else:
                kernel_suffix = V.graph.wrapper_code.next_kernel_suffix()
            kernel_name = "_".join(["mlir", fused_kernel_name, fx_graph_suffix if mode in ["complete_fallback", "auto_fallback"] else kernel_suffix])

            traced_graph_hash = code_hash(traced_graph.print_readable(print_output=False) + kernel_name)
            
            num_call_functions = get_num_call_functions(mlir_kernel._gm)

            if num_call_functions <= 1 or kernel_name in anir_config.force_fallback_kernel_names:
                mode = "complete_fallback"
            wrapper.src_to_kernel[kernel_key] = kernel_name

            if mode in ["auto_fallback", "default"]:
                src_code = src_code.replace("MODEL_NAME", kernel_name)
            
            mlir_processor = MLIRProcessor()
            src_code, kernel_info = mlir_processor.get_named_op_str(src_code, kernel_name, dynamic=mlir_kernel._is_dynamic)
            current_device = V.graph.get_current_device_or_throw()
            
            kernel_meta = {
                'device_str': current_device.type,
                'device_index': current_device.index,
                'num_outputs': mlir_kernel.num_outputs,
                'non_contiguous_indices': mlir_kernel.non_contiguous_indices,
                'dynamic': mlir_kernel._is_dynamic,
                'mutated_indices': mlir_kernel.mutated_indices,
                'traced_graph_cache': anir_config.traced_graph_cache,
                'traced_graph_hash': traced_graph_hash,
                'num_call_functions': num_call_functions,
                **kernel_info
            }

            compile_wrapper = IndentedBuffer()
            if mode == "auto_fallback":
                compile_wrapper.writeline(f"{kernel_name} = async_compile.mlir_auto_fallback({kernel_name!r}, '''")
                compile_wrapper.splice(src_code, strip=True)
                if 'PY_DIR_PATH' in os.environ:
                    kernel_path = os.path.join(os.environ['PY_DIR_PATH'], kernel_name + '.mlir')
                    with open(kernel_path, 'w') as f:
                        f.write(src_code)
                line = f"''', kernel_meta={kernel_meta})"
                compile_wrapper.writeline(line)
                metadata_comment = ''
                wrapper.header.splice(f"\n\n{metadata_comment}{compile_wrapper.getvalue()}")
            elif mode == "complete_fallback":
                compile_wrapper.writeline(f"async_compile.import_fx({kernel_name!r}, kernel_meta={kernel_meta})")
                metadata_comment = f'"""\n{mlir_kernel._gm.print_readable(print_output=False)}\n"""'
                wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)
            elif mode == "default":
                compile_wrapper.writeline(f"async_compile.mlir({kernel_name!r}, '''")
                compile_wrapper.splice(src_code, strip=True)
                compile_wrapper.writeline(
                    f"''', device_str='{V.graph.scheduler.current_device.type}')"
                )
                metadata_comment = ''
                if anir_config.debug:
                    with open(f"{anir_config.debug_dir}/fx_graph_runnable_{kernel_name}.py", 'w') as fd:
                        mlir_kernel.codegen_debug_performance(fd)
                    comment = 'related Nodes: ' + '+'.join(fx_graph_op_types(mlir_kernel._gm))
                    metadata_comment = f"'''\nsource fx graph:\n{comment}\n{mlir_kernel._gm.print_readable(print_output=False)}\n'''"
                wrapper.define_kernel(
                    kernel_name, compile_wrapper.getvalue(), metadata_comment
                )

            num_args = len(mlir_kernel._gm.code.split('forward(', )[1].split(')')[0].split(', ')) - 1

            if mode in ["complete_fallback", "auto_fallback"]:
                dump_path = os.path.join(os.getenv("TORCHINDUCTOR_CACHE_DIR"), anir_config.traced_graph_cache, str(current_device.index), traced_graph_hash)
                if not os.path.exists(dump_path):
                    os.makedirs(dump_path, exist_ok=True)
                    to_folder(mlir_kernel._gm, dump_path, graph_hash=traced_graph_hash, module_name=traced_graph_hash)

            if anir_config.fx_subgraph_dump_path is not None and mode in ["complete_fallback", "auto_fallback"]:
                subgraph_dump_path = os.path.join(anir_config.fx_subgraph_dump_path, str(current_device.index), kernel_name)
                os.makedirs(subgraph_dump_path, exist_ok=True)

                if mode == "complete_fallback":
                    fx_graph_code = get_fx_graph_code(mlir_kernel._gm.code, num_args, runnable=False)
                    runnable_fx_graph_code = get_fx_graph_code(mlir_kernel._gm.code, num_args, runnable=True)
                else:
                    fx_graph_code = get_fx_graph_code(mlir_kernel._gm.code, num_args, runnable=False, kernel_code=compile_wrapper.getvalue(), \
                                                      kernel_name=kernel_name)
                    runnable_fx_graph_code = get_fx_graph_code(mlir_kernel._gm.code, num_args, runnable=True, kernel_code=compile_wrapper.getvalue(), \
                                                      kernel_name=kernel_name)
                with open(os.path.join(subgraph_dump_path, f'{kernel_name}.py'), 'w') as f:
                    f.write(fx_graph_code)
                with open(os.path.join(subgraph_dump_path, f'runnable_{kernel_name}.py'), 'w') as f:
                    f.write(runnable_fx_graph_code)

                if mode == "auto_fallback":
                    with open(os.path.join(subgraph_dump_path, f'{kernel_name}.mlir'), 'w') as f:
                        f.write(src_code)

        return kernel_name

    # transform indexing before call codegen_node_schedule_with_kernel
    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures, nodes):
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
            V.graph.removed_buffers |= kernel.removed_buffers
            V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
            if not anir_config.traced_graph_cache:
                anir_config.traced_graph_cache = "traced_graph_cache"
            os.makedirs(os.path.join(os.getenv("TORCHINDUCTOR_CACHE_DIR"), anir_config.traced_graph_cache), exist_ok=True)
            traced_graph, call_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(nodes, kernel)
            mlir_kernel = NpuMlirKernel(traced_graph, nodes, call_args, **compile_kwargs)
            with V.set_kernel_handler(mlir_kernel):
                src_code = mlir_kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, mlir_kernel, traced_graph)
            log.debug("Generating kernel code with kernel_name: %s", kernel_name)
            kernel.kernel_name = kernel_name
            kernel.code_hash = code_hash(src_code)
        del kernel

        final_kernel: Union[SIMDKernel, MultiKernel]
        if len(kernels) > 1:
            raise RuntimeError("MultiKernel not Implemented!")
        else:
            (final_kernel,) = kernels

        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(call_args, final_kernel.kernel_name)

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernels[0].kernel_name)
            
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


    def codegen_node(self, node: Union[scheduler.FusedSchedulerNode, scheduler.SchedulerNode]):
        """
        Given a set of pre-fused nodes, generate a Mlir kernel.
        """
        nodes: List[scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
        
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        kernel_features = SIMDKernelFeatures(node_schedule, numel, rnumel)
        return self.codegen_node_schedule(kernel_features, nodes)