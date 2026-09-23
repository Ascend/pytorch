"""AscendC (AutoFuse) compiled-kernel adapter for the fxrt fx_wrapper.

Keeps the ``autofused_*`` kernels that inductor_npu_ext (torchair AutoFuse)
generates alive as HOP nodes in the inductor host graph, so the fx backend can
lower them to FXRT's ``compiled_kernel_mutation`` op.

AutoFuse must own the npu device codegen for its kernels to be generated at all,
and it registers that on import. Import ``inductor_npu_ext`` **before**
``fxrt``: whoever registers last wins.

One more thing the caller owns: torch-npu's inductor backend appends "npu" to
inductor's GPU_TYPES, after which the scheduler refuses to build any npu graph
without a working triton -- which AutoFuse generates no code for, and which is why
it keeps npu out of that list itself. Importing it (directly, or through anything
that pulls torch_npu._inductor) puts npu back, so an environment without a usable
triton has to drop it again by removing "npu" from
``torch._inductor.scheduler.GPU_TYPES`` before compiling. fxrt does not touch
that list on its own: it is a disagreement between two backends over torch
global state, not something to fix behind the caller's back.
"""
# pylint: disable=import-outside-toplevel,protected-access

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future

from torch._inductor.codecache import CodeCacheFuture, LambdaFuture, PyCodeCache
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from fxrt.fx_wrapper import CompiledKernelBackend

# AutoFuse names a fused kernel after its fused ops ("autofused_add_mul_relu_<md5>"),
# prefixed with "unsupported_<ops>_" when the fusion contains ops it cannot lower.
_ASCENDC_KERNEL_NAME_PREFIXES = ("auto", "unsupported_")
_ASCENDC_COMPILE_API = "async_compile_ascendc"


class _AscendcKernelLauncher:
    """Callable wrapper that launches a compiled AutoFuse kernel.

    The kernel takes flat positional args and writes its results through the
    buffers passed at its output positions. It launches on the stream torch
    considers current (its C wrapper resolves a null stream to
    ``c10_npu::getCurrentNPUStream()``).
    """

    def __init__(self, compiled: object, kernel_name: str) -> None:
        self._compiled = compiled
        self._kernel_name = kernel_name

    def __call__(self, *args: object) -> None:
        self._compiled(*args)

    def fxrt_converter_metadata(self) -> dict[str, str]:
        """Expose the shared libraries required by FX Converter runtime v2."""
        kernel_path = getattr(self._compiled, "lib_kernel", None)
        wrapper_path = getattr(self._compiled, "lib_wrapper", None)
        if not isinstance(kernel_path, str) or not kernel_path:
            raise RuntimeError("AscendC compiled kernel does not expose lib_kernel")
        if not isinstance(wrapper_path, str) or not wrapper_path:
            raise RuntimeError("AscendC compiled kernel does not expose lib_wrapper")
        return {
            "backend": "inductor_autofuse",
            "adapter_path": os.path.join(
                os.path.dirname(wrapper_path), "autofuse_adapter.so"
            ),
            "kernel_path": kernel_path,
        }


def _kernel_args(kernel_name: str):
    """Return the ``KernelArgs`` of the AutoFuse kernel generated under this name.

    ``KernelCallLine.arg_types`` carries dtypes for this backend, not the C types
    the cpp_pybinding path exposes, so a written arg cannot be told from a read
    one by its type. inductor_npu_ext keeps every kernel it generated in its
    scheduling backend, and that kernel's ``args`` is the authority on which of
    the call args are output (or in-place) buffers.
    """
    from torch._inductor.virtualized import V

    scheduler = getattr(V.graph, "scheduler", None)
    for backend in getattr(scheduler, "backends", {}).values():
        cache = getattr(backend, "_kernel_cache", None)
        kernel = cache.get(kernel_name) if cache else None
        args = getattr(kernel, "args", None)
        if args is not None:
            return args
    return None


def _written_arg_names(args) -> set:
    """Return the buffer names a kernel writes, as they appear in its call args."""
    written = set(getattr(args, "output_buffers", {}))
    for inplaced in getattr(args, "inplace_buffers", {}).values():
        # An in-place buffer is passed under the last of its aliased names.
        other_names = getattr(inplaced, "other_names", None)
        if other_names:
            written.add(other_names[-1])
    return written


class AscendcBackend(CompiledKernelBackend):
    """inductor_npu_ext (torchair AutoFuse) AscendC fused kernels as HOP nodes."""

    def handles_definition(self, defn_line) -> bool:
        name = getattr(defn_line, "kernel_name", "") or ""
        if name.startswith(_ASCENDC_KERNEL_NAME_PREFIXES):
            return True
        body = getattr(defn_line, "kernel_body", "") or ""
        return _ASCENDC_COMPILE_API in body

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        # The definition body is an async_compile_ascendc(...) call whose import
        # inductor_npu_ext carries in the metadata, so the kernel compiles from the
        # definition line alone.
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        # An async compile returns a plain concurrent.futures.Future subclass, not
        # one of inductor's own future types.
        while isinstance(kernel, (CodeCacheFuture, LambdaFuture, Future)):
            kernel = kernel.result()
        return _AscendcKernelLauncher(kernel, defn_line.kernel_name)

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        """Return the positions of the call args this kernel writes.

        The positions come from the kernel's own argdefs, not from the buffer
        names at this call site. NPUScheduling names a generated kernel after the
        fused graph it came from and reuses it for every structurally identical
        fusion, so a reused kernel is called with a different set of buffers than
        the one its KernelArgs hold: matching call-site names against
        ``output_buffers`` resolves only the first call site and silently leaves
        every later one with no mutated arg at all. Reuse requires the same fused
        graph, which is exactly what makes the argument layout stable.
        """
        args = _kernel_args(call_line.kernel_name)
        if args is None:
            raise NotImplementedError(
                f"AscendcBackend cannot tell which args kernel '{call_line.kernel_name}' "
                f"writes: inductor_npu_ext's scheduling backend no longer exposes the "
                f"kernel it generated (expected it in NPUScheduling._kernel_cache, keyed "
                f"by kernel name, with KernelArgs.output_buffers naming the written args)."
            )
        written = _written_arg_names(args)
        def_call_args = args.python_argdefs()[1]
        if len(def_call_args) != len(call_line.call_args):
            raise NotImplementedError(
                f"AutoFuse kernel '{call_line.kernel_name}' is called with "
                f"{len(call_line.call_args)} args but was defined with "
                f"{len(def_call_args)}; its written args cannot be resolved by position."
            )
        indices = tuple(
            i for i, name in enumerate(def_call_args) if str(name) in written
        )
        if not indices:
            raise NotImplementedError(
                f"AutoFuse kernel '{call_line.kernel_name}' writes none of its call args "
                f"{list(def_call_args)}; expected at least one output buffer among "
                f"{sorted(written)}."
            )
        return indices


__all__ = ["AscendcBackend"]
