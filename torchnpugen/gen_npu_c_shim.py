"""
NPU C Shim Code Generator for AOTInductor.

Generates c_shim_npu.h and c_shim_npu.cpp for the NPU device,
following the same pattern as PyTorch's torchgen/gen_aoti_c_shim.py
which generates c_shim_cuda.h/cpp and c_shim_cpu.h/cpp.

Integrated into the torchnpugen codegen pipeline.

Static dispatch:
  Following upstream pattern, we use at::npu::* instead of at::* to bypass the dispatcher
  and directly call NPU backend kernels for better performance.
"""

from __future__ import annotations

import difflib
import os
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from torchgen.aoti.fallback_ops import inductor_fallback_ops
from torchgen.context import method_with_native_function
from torchgen.gen import FileManager
from torchgen.gen_aoti_c_shim import (
    gen_declaration_and_definition,
    gen_static_dispatch_backend_call,
    get_fallback_op_name,
)
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.utils import mapMaybe


NPU_DEVICE = "npu"

NPU_DISPATCH_KEYS: Sequence[DispatchKey] = ()


def _get_backend_index_for_npu(
    func: NativeFunction,
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_func_group_dict: dict[OperatorName, NativeFunctionsGroup] = {},
) -> BackendIndex | None:
    for dk in NPU_DISPATCH_KEYS:
        if dk in backend_indices:
            if backend_indices[dk].has_kernel(func) or (
                func.structured_delegate is not None
                and func.structured_delegate in structured_func_group_dict
                and backend_indices[dk].has_kernel(
                    structured_func_group_dict[func.structured_delegate]
                )
            ):
                return backend_indices[dk]
    if DispatchKey.CompositeExplicitAutograd in backend_indices:
        if backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(func):
            return backend_indices[DispatchKey.CompositeExplicitAutograd]
    if DispatchKey.CompositeExplicitAutogradNonFunctional in backend_indices:
        if backend_indices[
            DispatchKey.CompositeExplicitAutogradNonFunctional
        ].has_kernel(func):
            return backend_indices[DispatchKey.CompositeExplicitAutogradNonFunctional]
    if DispatchKey.CompositeImplicitAutograd in backend_indices:
        if backend_indices[DispatchKey.CompositeImplicitAutograd].has_kernel(func):
            return backend_indices[DispatchKey.CompositeImplicitAutograd]
    return None


def _gen_npu_c_shim(
    func: NativeFunction,
    version_info: dict[str, list[str]],
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_func_group_dict: dict[OperatorName, NativeFunctionsGroup],
    header: bool,
) -> str | None:
    backend_index = _get_backend_index_for_npu(
        func, backend_indices, structured_func_group_dict
    )
    if backend_index is None:
        return None

    schema = func.func
    device = NPU_DEVICE
    backend_call = gen_static_dispatch_backend_call(func, backend_index)

    try:
        if header:
            declaration, _ = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return declaration
        else:
            _, definition = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return definition
    except NotImplementedError:
        return None


def _get_dispatch_header_path(
    func: NativeFunction,
    backend_index: BackendIndex | None,
) -> str | None:
    if backend_index is None:
        return None
    dispatch_ns = backend_index.dispatch_key.lower()
    return (
        f"#include <torch_npu/csrc/aten/ops/{func.root_name}_{dispatch_ns}_dispatch.h>"
    )


@dataclass(frozen=True)
class NPUShimGenerator:
    npu_fallback_ops: dict[str, dict[str, list[str]]]
    backend_indices: dict[DispatchKey, BackendIndex]
    structured_func_group_dict: dict[OperatorName, NativeFunctionsGroup]
    header: bool

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> str | None:
        version_info = self.npu_fallback_ops[get_fallback_op_name(func)]
        return _gen_npu_c_shim(
            func,
            version_info,
            self.backend_indices,
            self.structured_func_group_dict,
            self.header,
        )


def gen_npu_c_shim(
    native_functions: Sequence[NativeFunction],
    npu_fallback_ops: dict[str, dict[str, list[str]]],
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_func_group_dict: dict[OperatorName, NativeFunctionsGroup],
    header: bool,
    includes: str = "",
) -> str:
    body = "\n".join(
        list(
            mapMaybe(
                NPUShimGenerator(
                    npu_fallback_ops,
                    backend_indices,
                    structured_func_group_dict,
                    header,
                ),
                native_functions,
            )
        )
    )

    warning = """

// WARNING: THIS FILE IS AUTOGENERATED BY torchnpugen. DO NOT MODIFY BY HAND.
// See torchnpugen/gen_npu_c_shim.py for details"""

    if header:
        return (
            warning
            + textwrap.dedent("""

            #pragma once

            #include <torch_npu/csrc/inductor/aoti_torch/c/shim.h>

            #ifdef __cplusplus
            extern "C" {
            #endif

            """)
            + body
            + textwrap.dedent("""

            #ifdef __cplusplus
            } // extern "C"
            #endif
            """)
        )
    else:
        return (
            warning
            + textwrap.dedent(f"""

            #include <torch_npu/csrc/inductor/aoti_torch/generated/c_shim_{NPU_DEVICE}.h>
            #include <torch_npu/csrc/inductor/aoti_torch/utils.h>

            #ifndef AT_PER_OPERATOR_HEADERS
            #include <torch_npu/csrc/aten/NPUFunctions.h>
            #include <ATen/CompositeExplicitAutogradFunctions.h>
            #include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
            #include <ATen/CompositeImplicitAutogradFunctions.h>
            #else
            """)
            + includes
            + textwrap.dedent("""
            #endif // AT_PER_OPERATOR_HEADERS

            using namespace torch::aot_inductor;

            """)
            + body
        )


def gen_npu_c_shim_files(
    aoti_fm: FileManager,
    native_functions: Sequence[NativeFunction],
    backend_indices: dict[DispatchKey, BackendIndex],
    dispatch_keys: Sequence[DispatchKey],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    update_aoti_c_shim: bool,
) -> None:
    global NPU_DISPATCH_KEYS
    NPU_DISPATCH_KEYS = dispatch_keys

    structured_func_group_dict: dict[OperatorName, NativeFunctionsGroup] = {}
    for func_group in structured_native_functions:
        for func in func_group.functions():
            if func.structured_delegate is not None:
                structured_func_group_dict[func.structured_delegate] = func_group
                break

    fallback_ops_dict = inductor_fallback_ops
    fallbacks = {}
    for func in native_functions:
        op_name = get_fallback_op_name(func)
        if op_name in fallback_ops_dict:
            fallbacks[op_name] = func
    fallback_native_functions = tuple(value for _, value in sorted(fallbacks.items()))

    def headers_for_npu() -> str:
        headers = []
        for func in fallback_native_functions:
            backend_index = _get_backend_index_for_npu(
                func, backend_indices, structured_func_group_dict
            )
            header = _get_dispatch_header_path(func, backend_index)
            if header is not None:
                headers.append(header)
        return "\n".join(sorted(set(headers)))

    header_content = gen_npu_c_shim(
        fallback_native_functions,
        fallback_ops_dict,
        backend_indices,
        structured_func_group_dict,
        header=True,
    )
    cpp_content = gen_npu_c_shim(
        fallback_native_functions,
        fallback_ops_dict,
        backend_indices,
        structured_func_group_dict,
        header=False,
        includes=headers_for_npu(),
    )

    header_filename = f"c_shim_{NPU_DEVICE}.h"
    cpp_filename = f"c_shim_{NPU_DEVICE}.cpp"

    if update_aoti_c_shim:
        aoti_fm.write(header_filename, lambda: header_content)
    else:
        try:
            with open(os.path.join(aoti_fm.install_dir, header_filename)) as old_file:
                old_header = old_file.read()
                if old_header != header_content:
                    diff = "\n".join(
                        difflib.unified_diff(
                            old_header.splitlines(),
                            header_content.splitlines(),
                            fromfile="expected",
                            tofile="actual",
                            lineterm="",
                        )
                    )
                    raise RuntimeError(f"""
The generated AOTInductor C shim header files have unexpectedly changed. This
indicates an AOTInductor fallback operator ABI backward compatibility breakage!!!

1. You added a fallback op to the inductor_fallback_ops list in torchgen/aoti/fallback_ops.py.
If that's the case, run codegen with --update-aoti-c-shim to add a new entry to
existing C shim header files.

2. You added a new default argument to an existing fallback op. This is clearly a BC breaking
change in the AOTInductor land. You need to annotate the new default argument in
torchgen/aoti/fallback_ops.py, and then run codegen with --update-aoti-c-shim to
update the C shim header files by creating different versions of the fallback op.

{diff}
                    """)
        except FileNotFoundError:
            print(f"{os.path.join(aoti_fm.install_dir, header_filename)} not found")

    aoti_fm.write(cpp_filename, lambda: cpp_content)

    print(f"[torchnpugen] Generated {header_filename}")
    print(f"[torchnpugen] Generated {cpp_filename}")
