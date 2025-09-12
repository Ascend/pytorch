import sympy
import functools
import torch
from torch._inductor.virtualized import V
from torch._inductor import config, ir

from typing import List, Optional, Tuple, Union, Callable, Dict
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen, 
    pexpr, 
    cache_on_self,
    SubgraphPythonWrapperCodegen,
    counters,
)
from ... import codecache
from torch._inductor.codegen.common import (
    IndentedBuffer,
)

class NpuMlirWrapperCodeGen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

    @staticmethod
    def create(
        is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        if is_subgraph:
            return SubgraphPythonWrapperCodegen(subgraph_name, parent_wrapper)
        return NpuMlirWrapperCodeGen()

    @cache_on_self
    def write_triton_header_once(self) -> None:
        self.header.splice(
            """
            {}
            """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def write_header(self) -> None:
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import torch_npu
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty_strided
                from {codecache.__name__} import CustomAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.codegen.multi_kernel import MultiKernelCall
                from torch.utils._sympy.functions import FloatTrueDiv
                from torch.utils._sympy.functions import IntTrueDiv

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                async_compile = CustomAsyncCompile()

            """
        )

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        # If it's a NoneLayout then the extern_kernel should essentially be
        # treated as if it doesn't return anything
        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            # view operation fallbacks cause issues since inductor
            # doesn't know the memory is still needed and might reuse it.
            ending = f".clone(){ending}"

        if no_return:
            self.writeline(f"{self.declare}{kernel_name}({', '.join(args)}){ending}")
        else:
            self.writeline(
                f"{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}"
            )
            if kernel_name == 'torch.ops.npu_stream.npu_set_stream.default':
                device_idx = V.graph.scheduler.current_device.index
                name = f'stream{device_idx}'
                self.writeline(f"{name} = get_raw_stream({device_idx})")
            if (
                self.supports_intermediate_hooks
                and config.generate_intermediate_hooks
                and origin_node is not None
            ):
                counters["inductor"]["intermediate_hooks"] += 1
                self.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
                )

    def write_get_raw_stream(self, device_idx: int, graph=None) -> str:
        self.write_triton_header_once()
        name = f"stream{device_idx}"
        self.writeline(f"{name} = get_raw_stream({device_idx})")
        self.header.writeline(
            f"torch_npu.npu.set_device({device_idx})"
        )
        return name
    
    def generate_kernel_call(
        self,
        kernel_name,
        call_args,
        grid=None,
        device_index=None,
        gpu=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        """
        Generates kernel call code.

        cuda: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if gpu:
            call_args_str = ", ".join(pexpr(item) for item in call_args)
            stream_name = self.write_get_raw_stream(
                V.graph.scheduler.current_device.index, V.graph
            )
            self.writeline(
                f"{kernel_name}.run({call_args_str}, stream={stream_name})"
            )
        else:
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
