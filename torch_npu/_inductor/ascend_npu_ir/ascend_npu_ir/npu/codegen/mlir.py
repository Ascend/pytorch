import os
from typing import Optional

import torch
from torch._inductor.scheduler import Scheduler
from torch._inductor.utils import get_kernel_metadata
from torch._inductor.codecache import get_path
from torch._inductor.codegen.simd import code_hash
from torch._inductor import config
from torch._inductor.virtualized import V

from ... import config as anir_config
from ...npu.inductor_patch.lowering import map_strings_to_operators
from ...npu.utils import (
    MLIRProcessor,
    fold_expand,
    fx_graph_op_types,
    get_fx_graph_code,
    npu_cast_to_prim_cast,
    parse_fx_example_inputs,
    scalarize_tensor_ops_on_scalars,
    to_folder,
)
from ...npu.codegen.meta_kernel import NpuMetaKernel, NpuMetaScheduling


class NpuMlirKernel(NpuMetaKernel):
    def build_gm_with_prim_cast(self, gm):
        return npu_cast_to_prim_cast(gm)

    def codegen_kernel(self, name=None):
        scalarize_tensor_ops_on_scalars(self._gm_with_prim_cast)
        return super().codegen_kernel(name=name)

    def call_kernel(self, name: str, node: Optional[torch.fx.Node] = None):
        wrapper = V.graph.wrapper_code
        call_args = self.get_call_args()
        for call_arg in call_args:
            if call_arg.startswith("_uwu_"):
                expression = map_strings_to_operators(call_arg)
                wrapper.writeline(f"{call_arg} = {expression}")
        if len(call_args) > 0:
            wrapper.generate_kernel_call(name, call_args)

    def codegen_debug_performance(self, fd):
        from ...npu.utils import generate_compiler_repro_string, generate_fake_inputs

        name_to_example_inputs = parse_fx_example_inputs(self._gm)
        call_args_str = ", ".join(list(name_to_example_inputs.keys()))
        fd.write(generate_compiler_repro_string(self._gm))
        fd.write("\n")
        fd.write("if __name__ == '__main__':\n")
        fd.write("    from torch._inductor.utils import print_performance\n")
        fd.write("    with torch.no_grad():\n")
        fd.write(generate_fake_inputs(name_to_example_inputs))
        fd.write("\n")
        fd.write(f"        fn = lambda: mod({call_args_str})\n")
        fd.write("        print_performance(fn, times=10, repeat=10)\n")


class NpuMlirScheduling(NpuMetaScheduling):
    meta_kernel_type = NpuMlirKernel

    def __init__(self, scheduler: Scheduler):
        super().__init__(scheduler)

    def _get_compile_api(self) -> str:
        return "mlir_auto_fallback"

    def _get_kernel_prefix(self) -> str:
        return "mlir"

    def _postprocess_src_code(self, src_code, mlir_kernel, kernel_name):
        mlir_processor = MLIRProcessor()
        src_code, kernel_info = mlir_processor.get_named_op_str(
            src_code, kernel_name, dynamic=mlir_kernel._is_dynamic
        )
        return src_code, kernel_info

    def _handle_auto_fallback_mode(self, compile_wrapper, src_code, name, subs_name, meta, wrapper, metadata_comment, mlir_kernel=None):
        _basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")
        compile_wrapper.writeline(f"async_compile.{self._get_compile_api()}({subs_name!r}, '''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline(f"''', kernel_meta={meta})")
        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(mlir_kernel._snodes, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(name, compile_wrapper.getvalue(), metadata_comment)
        if config is not None and getattr(anir_config, "debug", False):
            pass

    def _handle_default_mode(self, compile_wrapper, src_code, name, subs_name, meta, wrapper, metadata_comment, mlir_kernel=None):
        compile_wrapper.writeline(f"async_compile.mlir({name!r}, '''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline(f"''', device_str='{V.graph.scheduler.current_device.type}')")
        metadata_comment = ""
        if anir_config.debug:
            with open(f"{anir_config.debug_dir}/fx_graph_runnable_{name}.py", "w") as fd:
                mlir_kernel.codegen_debug_performance(fd)
            comment = "related Nodes: " + "+".join(fx_graph_op_types(mlir_kernel._gm))
            metadata_comment = f"'''\nsource fx graph:\n{comment}\n{mlir_kernel._gm.print_readable(print_output=False)}\n'''"
        wrapper.define_kernel(name, compile_wrapper.getvalue(), metadata_comment)

    def _dump_fx_graph_for_fallback(self, mlir_kernel, device, graph_hash, kernel_name, compile_code):
        cache_root = os.getenv("TORCHINDUCTOR_CACHE_DIR")
        dump_path = os.path.join(
            cache_root,
            anir_config.traced_graph_cache,
            str(device.index),
            graph_hash,
        )
        if not os.path.exists(dump_path):
            os.makedirs(dump_path, exist_ok=True)
            if anir_config.fallback_folder_expand:
                fold_expand(mlir_kernel._gm)
            to_folder(mlir_kernel._gm, dump_path, graph_hash=graph_hash, module_name=graph_hash)

        if anir_config.fx_subgraph_dump_path is not None:
            subgraph_dump_path = os.path.join(anir_config.fx_subgraph_dump_path, str(device.index), kernel_name)
            os.makedirs(subgraph_dump_path, exist_ok=True)
            num_args = len(mlir_kernel._gm.code.split("forward(",)[1].split(")")[0].split(", ")) - 1
            if "kernel_code" in compile_code:
                pass
            fx_graph_code = get_fx_graph_code(
                mlir_kernel._gm.code,
                num_args,
                runnable=False,
                kernel_code=compile_code,
                kernel_name=kernel_name,
            )
            runnable_fx_graph_code = get_fx_graph_code(
                mlir_kernel._gm.code,
                num_args,
                runnable=True,
                kernel_code=compile_code,
                kernel_name=kernel_name,
            )
            with open(os.path.join(subgraph_dump_path, f"{kernel_name}.py"), "w") as f:
                f.write(fx_graph_code)
            with open(os.path.join(subgraph_dump_path, f"runnable_{kernel_name}.py"), "w") as f:
                f.write(runnable_fx_graph_code)
