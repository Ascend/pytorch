"""DVM compiled-kernel adapter for the fxrt fx_wrapper."""
# pylint: disable=import-outside-toplevel,protected-access

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._inductor.codecache import CodeCacheFuture, LambdaFuture, PyCodeCache
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from fxrt.fx_wrapper import CompiledKernelBackend, cpp_mutated_arg_indices

_DVM_KERNEL_NAME_PREFIX = "dvm_"
_DVM_COMPILE_APIS = (
    "async_compile.mlir",
    "async_compile.akg",
    "async_compile.import_fx",
)


def _current_npu_raw_stream():
    """Return the current NPU raw stream when torch-npu exposes one."""

    try:
        from torch_npu._inductor.utils import get_current_raw_stream

        return get_current_raw_stream(torch.npu.current_device())
    except Exception:  # pylint: disable=broad-exception-caught
        return None


class _DvmKernelLauncher:
    """Callable wrapper that launches a torch-npu compiled DVM kernel."""

    def __init__(self, compiled: object, kernel_name: str) -> None:
        self._compiled = compiled
        self._kernel_name = kernel_name

    def __call__(self, *args: object) -> None:
        compiled = self._compiled
        if hasattr(compiled, "run"):
            compiled.run(*args, stream=_current_npu_raw_stream())
        else:
            compiled(*args)


class DvmBackend(CompiledKernelBackend):
    """torch-npu dvm/mlir fused kernels embedded as HOP nodes."""

    def handles_definition(self, defn_line) -> bool:
        name = getattr(defn_line, "kernel_name", "") or ""
        if name.startswith(_DVM_KERNEL_NAME_PREFIX):
            return True
        body = getattr(defn_line, "kernel_body", "") or ""
        return any(api in body for api in _DVM_COMPILE_APIS)

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        while isinstance(kernel, (CodeCacheFuture, LambdaFuture)):
            kernel = kernel.result()
        return _DvmKernelLauncher(kernel, defn_line.kernel_name)

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        arg_types = getattr(call_line, "arg_types", None)
        if not arg_types:
            raise NotImplementedError(
                "DvmBackend needs KernelCallLine.arg_types to know which args "
                "are written. Make torch_npu's NpuMetaKernel.call_kernel pass "
                "arg_types to generate_kernel_call, marking output buffers as "
                "non-const pointers."
            )
        return cpp_mutated_arg_indices(arg_types)


__all__ = ["DvmBackend"]
