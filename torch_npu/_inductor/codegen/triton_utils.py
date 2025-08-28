import inspect
from typing import List, Dict, Any

import sympy
import torch
from torch._inductor import ir, config
from torch._inductor.codegen.common import KernelArgType, SizeArg, TensorArg, TMADescriptorArg
from torch._inductor.codegen.triton_utils import signature_to_meta
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, \
    user_defined_triton_kernel_transitive_closure_source_code
from torch._inductor.runtime import triton_heuristics
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._triton import patch_triton_dtype_repr

from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel, gen_npu_triton_ext_imports
from torch_npu._inductor.runtime import NPUDeviceProperties

# wrapper npu 32 bytes align, get and pass unalign info to triton meta
# then autotune choose tiling param and send them to bishengIR
byte_per_numel = {
    torch.float32: 4,  # torch.float32 or torch.float
    torch.float64: 8,  # torch.float64 or torch.double
    torch.float16: 2,  # torch.float16 or torch.half
    torch.bfloat16: 2,  # torch.bfloat16
    torch.int32: 4,  # torch.int32 or torch.int
    torch.int64: 8,  # torch.int64 or torch.long
    torch.int16: 2,  # torch.int16 or torch.short
    torch.int8: 1,  # torch.int8
    torch.uint8: 1,  # torch.uint8
    torch.bool: 1,  # torch.bool
    torch.complex32: 4,  # torch.complex32 (not yet available in PyTorch as of the latest stable release)
    torch.complex64: 8,  # torch.complex64
    torch.complex128: 16  # torch.complex128
}


def get_aligned_numel(dtype):
    if dtype in byte_per_numel:
        return 32 // byte_per_numel[dtype]
    else:
        return 1


def write_npu_triton_header_once(wrapper: PythonWrapperCodegen) -> None:
    import_str = f"""
        import triton
        import triton.language as tl
        from {triton_heuristics.__name__} import (                
            split_scan_grid,
            grid_combo_kernels,
            start_graph,
            end_graph,
            cooperative_reduction_grid,
        )
        from torch_npu._inductor.npu_triton_heuristics import grid
        import torch_npu
        """
    if config.triton.autotune_at_compile_time:
        wrapper.kernel_autotune_calls.splice(import_str)
        wrapper.kernel_autotune_calls.writeline(
            V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        )
    if not V.graph.cpp_wrapper:
        wrapper.imports.splice(import_str, strip=True)
        wrapper.imports.writeline(
            V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        )


def define_user_defined_npu_triton_kernel(
        wrapper: PythonWrapperCodegen,
        kernel,
        configs,
        kwargs,
        restore_value_args,
        reset_to_zero_args
):
    patch_triton_dtype_repr()

    original_name = kernel.__name__

    signature: List[KernelArgType] = []
    constants: Dict[str, Any] = {}
    non_constant_indices = []
    equal_to_1_args: List[str] = []
    for idx, key in enumerate(kernel.arg_names):
        if key not in kwargs:
            continue
        arg = kwargs[key]
        if idx in kernel.constexprs:
            constants[key] = arg
        elif kwargs[key] is None:
            constants[key] = None
        else:
            non_constant_indices.append(idx)
            if isinstance(arg, ir.TMADescriptor):
                signature.append(
                    TMADescriptorArg(
                        name=key,
                    )
                )
            elif isinstance(arg, ir.Buffer):
                signature.append(
                    TensorArg(
                        name=key,
                        buffer=arg.get_name(),
                        dtype=arg.get_dtype(),
                    )
                )
            elif isinstance(arg, ir.ReinterpretView):
                # for ReinterpretView we use the underlying
                # buffer name and note the (possibly non-zero)
                # offset relative to the underlying buffer
                signature.append(
                    TensorArg(
                        name=key,
                        buffer=arg.data.get_name(),
                        dtype=arg.get_dtype(),
                        offset=arg.layout.offset,
                    )
                )
            else:
                signature.append(SizeArg(key, arg))
                if isinstance(
                        arg, (int, sympy.Integer)
                ) and V.graph.sizevars.statically_known_equals(
                    arg, 1  # type: ignore[arg-type]
                ):
                    equal_to_1_args.append(key)
    triton_meta: Dict[str, Any] = {
        "signature": signature_to_meta(
            signature,
            size_dtype=None,  # try to infer based on symints
            indices=non_constant_indices,
            argdefs=kernel.arg_names,
        ),
        "device": NPUDeviceProperties.create(
            V.graph.get_current_device_or_throw()
        ),
        "constants": {
            **constants,
            **dict.fromkeys(equal_to_1_args, 1),
        },
        # special config for NPU, specify compile target
        "mix_mode": "aiv",
    }

    if restore_value_args:
        triton_meta["restore_value"] = tuple(restore_value_args)

    if reset_to_zero_args:
        triton_meta["reset_to_zero"] = tuple(reset_to_zero_args)

    # Distinguish between different functions using function id
    cache_key: List[Any] = [id(kernel.fn)]
    if len(configs) > 0:
        for arg in kwargs.values():
            # We need to key on non tensor arg only in autotune mode
            if not isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                cache_key.append(arg)
    cache_key.append(str(triton_meta))
    cache_key_tuple = tuple(cache_key)

    if cache_key_tuple in wrapper.user_defined_kernel_cache:
        return wrapper.user_defined_kernel_cache[cache_key_tuple]

    name = f"{original_name}_{len(wrapper.user_defined_kernel_cache)}"
    # Add to the cache for the next use
    wrapper.user_defined_kernel_cache[cache_key_tuple] = (name, triton_meta)

    compile_wrapper = IndentedBuffer()
    compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")

    from .triton import gen_common_triton_imports, TritonKernel

    compile_wrapper.splice(gen_common_triton_imports())
    compile_wrapper.splice(gen_npu_triton_ext_imports())

    inductor_meta = {
        "kernel_name": name,
        **NPUIndexTritonKernel.inductor_meta_common(),
    }

    configs = [
        {
            "kwargs": config.kwargs,
        }
        for config in configs
    ]

    compile_wrapper.splice(
        f"""
        @npu_triton_heuristics.npu_user_autotune(
            configs={configs!r},
            triton_meta={triton_meta!r},
            filename=__file__,
            inductor_meta={inductor_meta!r},
            custom_kernel=True,
        )
        @triton.jit
        """
    )
    compile_wrapper.splice(
        user_defined_triton_kernel_transitive_closure_source_code(kernel)
    )

    current_device = V.graph.get_current_device_or_throw()
    compile_wrapper.writeline(f"''', device_str='{current_device.type}')")
    _, lineno = inspect.getsourcelines(kernel.fn)
    srcfile = inspect.getsourcefile(kernel.fn)
    metadata = f"# Original path: {srcfile}:{lineno}"
    wrapper.define_kernel(
        name,
        compile_wrapper.getvalue(),
        metadata,
    )
    return name, triton_meta
