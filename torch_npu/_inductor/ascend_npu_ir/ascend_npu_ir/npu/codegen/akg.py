from itertools import count
from typing import Dict, List, Optional

import sympy
from torch._functorch.aot_autograd import (
    get_aot_compilation_context,
    set_model_name,
)
from torch._inductor import config, metrics
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.simd import code_hash
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonKernel,
    TritonScheduling,
)
from torch._inductor.codecache import get_path
from torch._inductor.scheduler import Scheduler
from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.virtualized import V

from ... import config as anir_config
from ...npu.utils import (
    is_fx_dynamic,
    logger,
    modify_gm_for_acc_comp,
    npu_cast_to_prim_cast,
    scalarize_tensor_ops_on_scalars,
)
from ...npu.codegen.mlir import create_fx_from_snodes_by_traced_graph

try:
    from akg.kernel import Kernel as MlirKernel
except ImportError:
    logger.warning("akg is not installed, install it first.")

id_iter = count()


class AkgCompiler:
    def __init__(self, kernel_meta=None):
        self.kernel_name = kernel_meta.get('kernel_name')
        self.kernel = MlirKernel(kernel_meta)

    def compile(self, input_mlir):
        self.kernel.compile(input_mlir)
        
    def run(self, *args, **kwargs):
        self.kernel.run(*args, **kwargs)


class AkgKernel(TritonKernel):
    def __init__(
        self,
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
            fixed_config=fixed_config,
            **kwargs,
        )

    def codegen_kernel(self, Name=None):
        from torch_mlir.compiler_utils import OutputType
        from torch_mlir.fx import stateless_fx_import

        nodes = self.features.node_schedule
        traced_graph, call_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(
            nodes, None
        )

        gm = traced_graph

        gm_with_prim_cast = npu_cast_to_prim_cast(gm)

        is_dynamic = is_fx_dynamic(gm)
        if anir_config.online_acc_comp:
            modify_gm_for_acc_comp(gm)

        code = IndentedBuffer()

        scalarize_tensor_ops_on_scalars(gm_with_prim_cast)

        set_model_name("MODEL_NAME")
        *_, model_name, nth_graph = get_aot_compilation_context()

        mlir_module = stateless_fx_import(
            gm_with_prim_cast,
            output_type=OutputType.LINALG_ON_TENSORS,
            model_name=model_name,
        )

        code.splice(str(mlir_module))
        src_code = code.getvalue()
        return src_code


class AkgScheduling(TritonScheduling):
    kernel_type = AkgKernel

    def __init__(self, sched: Scheduler):
        super().__init__(sched)

    def define_kernel(self, src_code, node_schedule, mode=None):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(
                ["mlir", fused_name, wrapper.next_kernel_suffix(), "auto_fallback"]
            )
            traced_graph, call_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(node_schedule, None)
            is_dynamic = is_fx_dynamic(traced_graph)
            current_device = V.graph.get_current_device_or_throw()
            
            kernel_meta = {
                'device_str': current_device.type,
                'device_index': current_device.index,
                'kernel_name': kernel_name,
                'num_outputs': compile_kwargs.get('num_outputs'),
                'dynamic': is_dynamic,
            }
            wrapper.src_to_kernel[src_code] = kernel_name
            subs_name = kernel_name if config.triton.unique_kernel_names else "mlir_"

            src_code = src_code.replace("MODEL_NAME", kernel_name)
            _basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")
            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.akg_auto_fallback({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(f"''', kernel_meta={kernel_meta})")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            if metrics.is_metric_table_enabled("kernel_metadata"):
                metrics.log_kernel_metadata(kernel_name, kernel_path, src_code)
        
        return kernel_name