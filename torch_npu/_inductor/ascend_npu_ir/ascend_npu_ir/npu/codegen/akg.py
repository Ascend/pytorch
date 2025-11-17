import subprocess
import copy
import textwrap
from itertools import count
from typing import List, Union, Optional, Tuple, Any, Dict
import numpy as np

import torch
import sympy
from sympy import Expr, Integer
from torch._dynamo.utils import counters

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
    TritonScheduling,
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
    get_kernel_metadata,
)

from torch._inductor import config, metrics
from torch._inductor.virtualized import V
from torch._inductor.codecache import get_path

from ... import config as anir_config
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
    view_to_reshape,
    logger
)

try: 
    from akg.kernel import Kernel as MlirKernel
except ImportError as e:
    logger.warning(f"akg is not installed, install it first.")

from ...npu.codegen.mlir import NpuMlirKernel, create_fx_from_snodes_by_traced_graph

if anir_config.enable_graph_trace:
    from ...npu.inductor_patch.lowering import (
        merge_fx_graphs,
        map_strings_to_operators
    )

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
    def __init__(self,
            tiling: Dict[str, sympy.Expr],
            min_elem_per_thread=0,
            optimize_mask=True,
            fixed_config: Optional[FixedTritonConfig] = None,
            **kwargs):
        super().__init__(
            tiling,
            min_elem_per_thread=min_elem_per_thread,
            optimize_mask=optimize_mask,
            fixed_config=fixed_config,
            **kwargs,
        )

    def codegen_kernel(self, Name=None):
        nodes = self.features.node_schedule
        traced_graph, call_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(nodes, None)
        mlir_kernel = NpuMlirKernel(traced_graph, nodes, call_args, **compile_kwargs)
        with V.set_kernel_handler(mlir_kernel):
            src_code = mlir_kernel.codegen_kernel()
        return src_code


class AkgScheduling(TritonScheduling):
    kernel_type = AkgKernel

    def __init__(self, scheduler: Scheduler):
        super().__init__(scheduler)

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
            mlir_processor = MLIRProcessor()
            src_code, kernel_info = mlir_processor.get_named_op_str(src_code, kernel_name, is_dynamic)
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