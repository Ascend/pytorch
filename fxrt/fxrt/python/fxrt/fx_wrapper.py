# Copyright(c) 2026, the respective contributors
# All rights reserved.
#
# Modifications by Huawei Technologies Co., Ltd. 2025.
#
# This file contains code derived from PyTorch:
# torch/_higher_order_ops/triton_kernel_wrap.py
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
#    Laboratories America and IDIAP Research Institute nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Inductor fx_wrapper integration for fxrt.

Importing this module registers nothing. :func:`register_fx_wrapper` installs
the fx_wrapper codegen for torch-npu and switches inductor to the fx form; the
installed wrapper then routes the generated host FX GraphModule to fxrt by
default. :func:`set_fx_wrapper_backend` and :func:`fx_wrapper_backend` select
another backend for that graph.

The fused-kernel HOP pieces are migrated from the temporary TDC v4 prototype so
torch-npu/dvm fused kernels can survive inside the host FX graph instead of
being rejected by stock ``WrapperFxCodegen``.

For debugging, :func:`set_eager_debug` makes the wrapper run the generated host
FX graph in python eager (``gm.forward``) instead of routing it to the fxrt
runtime, so the two can be diffed on identical inputs.
"""
# pylint: disable=import-outside-toplevel,unused-argument,protected-access,undefined-all-variable,wrong-import-position

from __future__ import annotations

import abc
import contextlib
import dataclasses
import inspect
import os
from collections.abc import Callable
from typing import Any

import sympy

import torch
from torch._inductor import ir
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.codegen.common import FileBackedGraphModule
from torch._inductor.codegen.wrapper import CommentLine, PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import FxConverter, WrapperFxCodegen
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from fxrt import compiled_kernel_hop as _compiled_kernel_hop
from fxrt.fx_simplify import simplify_duplicate_symints

CompiledKernelSideTable = _compiled_kernel_hop.CompiledKernelSideTable
compiled_kernel_side_table = _compiled_kernel_hop.compiled_kernel_side_table
compiled_kernel_wrapper_functional = (
    _compiled_kernel_hop.compiled_kernel_wrapper_functional
)
compiled_kernel_wrapper_mutation = _compiled_kernel_hop.compiled_kernel_wrapper_mutation
launch_compiled_kernel_functional = (
    _compiled_kernel_hop.launch_compiled_kernel_functional
)
launch_compiled_kernel_mutation = _compiled_kernel_hop.launch_compiled_kernel_mutation


_active_fx_backend: "Callable | None" = None
_installed_devices: set[str] = set()

# Debug switch: run the generated host FX graph in python eager (gm.forward)
# instead of handing it to the fxrt runtime. Off by default.
_eager_debug = False


def set_eager_debug(enabled: bool = True) -> bool:
    """Run the host FX graph in python eager instead of the fxrt runtime.

    Set this before the ``torch.compile`` call whose graph you want to compare;
    compiled artifacts are cached, so flipping it after compilation has no
    effect on an already-compiled graph.

    Args:
        enabled: ``True`` to eager-run ``gm.forward``, ``False`` for the normal
            fxrt runtime path.

    Returns:
        The previous value.
    """

    global _eager_debug
    previous = _eager_debug
    _eager_debug = enabled
    return previous


def _default_fx_backend(gm, example_inputs):
    """Run the default fxrt FX backend."""

    from fxrt.fx_backend import backend

    return backend(gm, example_inputs)


def get_fx_wrapper_backend() -> "Callable | None":
    """Return the configured process-wide fx_wrapper backend.

    ``None`` means the wrapper will use ``fxrt.fx_backend.backend``.
    """

    return _active_fx_backend


def set_fx_wrapper_backend(gm_backend: "Callable | None") -> "Callable | None":
    """Set the process-wide backend used by the installed fx_wrapper.

    Args:
        gm_backend: callable ``(gm, example_inputs) -> compiled_callable``.
            Passing ``None`` restores the default fxrt backend.

    Returns:
        The previous backend value.
    """

    global _active_fx_backend
    previous = _active_fx_backend
    _active_fx_backend = gm_backend
    return previous


@contextlib.contextmanager
def fx_wrapper_backend(gm_backend: "Callable | None"):
    """Temporarily set the fx_wrapper backend."""

    previous = set_fx_wrapper_backend(gm_backend)
    try:
        yield
    finally:
        set_fx_wrapper_backend(previous)


def _resolve_backend() -> Callable:
    """Return the configured backend or the default fxrt backend."""

    return _active_fx_backend or _default_fx_backend


class CompiledKernelBackend(abc.ABC):
    """Teaches the FX converter how to preserve one non-Triton kernel kind."""

    @abc.abstractmethod
    def handles_definition(self, defn_line) -> bool:
        """Return true if this backend owns the kernel definition line."""

    @abc.abstractmethod
    def compile_kernel(self, converter: "CompiledKernelFxConverter", defn_line) -> Callable:
        """Compile to a callable taking flat positional args."""

    @abc.abstractmethod
    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        """Return positions in the kernel call args written by the kernel."""


_COMPILED_BACKENDS: list[CompiledKernelBackend] = []


def register_compiled_kernel_backend(backend: CompiledKernelBackend) -> None:
    """Register a compiled-kernel backend."""

    if not any(isinstance(x, type(backend)) for x in _COMPILED_BACKENDS):
        _COMPILED_BACKENDS.append(backend)


def _select_backend(defn_line) -> "CompiledKernelBackend | None":
    """Return the registered backend that owns a compiled-kernel definition."""

    for backend in _COMPILED_BACKENDS:
        if backend.handles_definition(defn_line):
            return backend
    return None


class CompiledKernelFxConverter(FxConverter):
    """FxConverter that routes non-Triton kernels through registered backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernels: dict[str, tuple[int, CompiledKernelBackend]] = {}

    def _generate_kernel_definition(self, line) -> None:
        backend = _select_backend(line)
        if backend is None:
            super()._generate_kernel_definition(line)
            return
        kernel = backend.compile_kernel(self, line)
        idx = compiled_kernel_side_table.add_kernel(kernel)
        self.compiled_kernels[line.kernel_name] = (idx, backend)

    def _generate_kernel_call(self, line) -> None:
        """Emit a compiled-kernel HOP call, or defer to the stock kernel path."""

        entry = self.compiled_kernels.get(line.kernel_name)
        if entry is None:
            super()._generate_kernel_call(line)
            return
        idx, backend = entry
        # _lookup_args resolves SymbolicCallArg to a bare sympy.Expr, not an fx
        # Node, so symbolic args must go through _generate_sym_node (as the
        # Triton call path does) to become real graph edges. Otherwise the
        # symbol is embedded as a repr()'d constant that happens to print
        # identically to the live variable name, and FX's last-use liveness
        # analysis frees that variable long before this call actually runs.
        args = tuple(self._lift_sym_args(a) for a in self._lookup_args(line.call_args))
        self.gm.graph.call_function(
            compiled_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": idx,
                "mutated_arg_indices": tuple(backend.mutated_arg_indices(line)),
                "args": args,
            },
        )

    # -- extern fallback calls -------------------------------------------------
    def _lift_sym_args(self, arg):
        """Recursively replace raw sympy shape args with real fx nodes.

        Stock FxConverter embeds ``kernel.constant_args`` verbatim; when a backend
        falls back view-ish ops (e.g. aten.reshape on NPU) those constants carry
        raw sympy symbols, which then appear in the host gm as bare names (``s0``)
        with no matching variable -- ``NameError: name 's0' is not defined`` at
        run time (it may also silently alias an unrelated same-named variable).
        Every in-scope symbol already has an expr_to_proxy entry (placeholder or
        sym-compute node), so route them through _generate_sym_node, exactly like
        the Triton / compiled-kernel call paths do.
        """
        if isinstance(arg, sympy.Expr):
            return self._generate_sym_node(arg)
        if isinstance(arg, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            # inductor passes shape constants as torch.Sym* wrappers, not bare
            # sympy exprs -- unwrap first (this is the common case in practice).
            return self._generate_sym_node(arg.node.expr)
        if isinstance(arg, (list, tuple)):
            return type(arg)(self._lift_sym_args(a) for a in arg)
        if isinstance(arg, dict):
            return {k: self._lift_sym_args(v) for k, v in arg.items()}
        return arg

    def _generate_extern_kernel_common(self, kernel, out_ir_node) -> None:
        """Mirror FxConverter._generate_extern_kernel_common (torch 2.13), plus
        _lift_sym_args on the call args/kwargs (see docstring above)."""

        if not ir.is_node_sequence(kernel.inputs):
            raise TypeError(
                f"Extern kernel '{kernel}' inputs are not a node sequence: {kernel.inputs!r}"
            )
        tensor_nodes = tuple(self._generate_buffer(arg) for arg in kernel.inputs)
        if hasattr(kernel, "unflatten_args"):
            args, _ = kernel.unflatten_args(tensor_nodes, kernel.constant_args)
        else:
            args = tensor_nodes + tuple(kernel.constant_args)
        args = self._lift_sym_args(args)

        kwargs = {
            k: self._generate_buffer(v) if isinstance(v, ir.IRNode) else v
            for k, v in kernel.kwargs.items()
        }
        kwargs = self._lift_sym_args(kwargs)

        result_buffer = None
        if isinstance(kernel, ir.ExternKernelOut):
            kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            result_buffer = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(f"Unrecognized output layout: {kernel.layout}")

        fx_node = self.gm.graph.call_function(
            kernel.op_overload,
            args=args,
            kwargs=kwargs,
        )

        if result_buffer:
            if "out" in kwargs:
                raise RuntimeError(
                    f"Extern kernel '{kernel}' has both result and out kwarg. Expected only one."
                )
            fx_node.name = result_buffer
            self.buffer_to_node[result_buffer] = fx_node


class FxrtFxWrapper(WrapperFxCodegen):
    """WrapperFxCodegen that sends the host FX graph to the configured backend."""

    def writeline(self, line):
        """Accept a wrapper line, tolerating raw source strings.

        Third-party fusion backends sometimes writeline raw source strings
        straight into the wrapper. The fx converter only lowers structured
        WrapperLine objects, so a bare str aborts conversion. Comments and
        blank lines carry no semantics and are dropped (the converter already
        ignores CommentLine); any other raw source cannot be represented as FX
        and must fail loudly rather than silently miscompile.
        """
        if isinstance(line, str):
            nonblank = [s for s in line.splitlines() if s.strip()]
            if all(s.lstrip().startswith("#") for s in nonblank):
                line = CommentLine(line)
            else:
                raise NotImplementedError(
                    f"fx_wrapper cannot lower raw source written to the wrapper: {line!r}"
                )
        super().writeline(line)

    def _generate(self, is_inference: bool):
        """Generate the wrapper FX graph and compile it with the active backend."""

        self.run_wrapper_ir_passes(is_inference)
        prologue = "\n".join([self.imports.getvalue(), self.header.getvalue()])
        gm = self._make_fx_converter(prologue).generate()
        return FileBackedGraphModule(gm, self.compile_graph(gm)), None

    def _make_fx_converter(self, prologue: str) -> CompiledKernelFxConverter:
        """Build the converter for whichever FxConverter signature torch has.

        torch 2.9's FxConverter takes only lines/prologue and reads the graph
        inputs/outputs off V.graph itself; the graph_inputs/graph_outputs/subgms/
        is_subgraph fields (and get_fx_graph_inputs) only exist from 2.10 on, and
        are required there -- so one call cannot serve both.
        """

        kwargs: dict[str, Any] = {"lines": self.lines, "prologue": prologue}
        if "graph_inputs" in inspect.signature(FxConverter.__init__).parameters:
            kwargs.update(
                graph_inputs=self.get_fx_graph_inputs(),
                graph_outputs=self.get_graph_outputs(),
                subgms=self.subgms,
                is_subgraph=self.is_subgraph,
            )
        return CompiledKernelFxConverter(**kwargs)

    def compile_graph(self, gm):
        """Compile the generated FX graph with the active fxrt backend.

        After :func:`set_eager_debug`, the graph is instead returned as-is for
        python eager execution -- this is stock ``WrapperFxCodegen`` behaviour
        and gives a reference result to diff against the fxrt runtime, on
        the very same host graph (fused kernels still run, via the
        compiled-kernel HOP's CompositeExplicitAutograd impl).
        """

        simplify_duplicate_symints(gm)
        if _eager_debug:
            return gm.forward
        for node in gm.graph.nodes:
            if "example_value" not in node.meta and node.meta.get("val") is not None:
                node.meta["example_value"] = node.meta["val"]
        example_inputs = [
            n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"
        ]
        return _resolve_backend()(gm, example_inputs)


def cpp_mutated_arg_indices(arg_types) -> tuple[int, ...]:
    """A C ABI kernel arg is mutated iff it is a non-const pointer."""

    return tuple(
        i
        for i, t in enumerate(arg_types)
        if isinstance(t, str) and "*" in t and not t.strip().startswith("const")
    )


class CppPybindingBackend(CompiledKernelBackend):
    """Inductor CPU cpp_pybinding kernels."""

    def handles_definition(self, defn_line) -> bool:
        return not getattr(defn_line, "gpu", True)

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()
        if isinstance(kernel, CachingAutotuner):
            raise AssertionError("Triton kernel reached the compiled-kernel backend")
        return kernel

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        return cpp_mutated_arg_indices(call_line.arg_types)


from fxrt.ascendc_fx_wrapper import AscendcBackend
from fxrt.dvm_fx_wrapper import DvmBackend

register_compiled_kernel_backend(CppPybindingBackend())
register_compiled_kernel_backend(DvmBackend())
register_compiled_kernel_backend(AscendcBackend())


def _replace_fx_wrapper_codegen(device_codegen, wrapper_cls):
    """Return a device codegen with the requested fx_wrapper class installed."""

    try:
        return dataclasses.replace(device_codegen, fx_wrapper_codegen=wrapper_cls)
    except (TypeError, AttributeError):
        if not hasattr(device_codegen, "fx_wrapper_codegen"):
            raise
        device_codegen.fx_wrapper_codegen = wrapper_cls
        return device_codegen


def _install_fx_wrapper_codegen(device: str = "npu") -> bool:
    """Install the fxrt fx_wrapper codegen for torch-npu.

    This is idempotent. It imports ``torch_npu._inductor`` when the device has no
    codegen registered yet, so the torch-npu backend has registered one before we
    patch it. Returns ``True`` when the wrapper is installed, otherwise ``False``
    when torch-npu/inductor is not available in the current environment.
    """

    try:
        import torch_npu  # pylint: disable=unused-import,import-outside-toplevel
        from torch._inductor.codegen.common import (
            device_codegens,
            init_backend_registration,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if os.environ.get("FXRT_FX_WRAPPER_DEBUG") == "1":
            print(f"fxrt fx_wrapper install skipped: {exc}")
        return False

    device = torch.device(device).type
    if device not in device_codegens:
        # Only torch-npu's own inductor backend needs this import. Another backend
        # (e.g. AutoFuse) may already own the device codegen, and importing
        # torch_npu._inductor then does more harm than good: it appends the device
        # to inductor's GPU_TYPES, after which the scheduler demands a working
        # triton for every graph on it -- one the fused backend never needed.
        try:
            import torch_npu._inductor  # noqa: F401  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if os.environ.get("FXRT_FX_WRAPPER_DEBUG") == "1":
                print(f"fxrt fx_wrapper install skipped: {exc}")
            return False
    init_backend_registration()
    device_codegen = device_codegens.get(device)
    if device_codegen is None:
        if os.environ.get("FXRT_FX_WRAPPER_DEBUG") == "1":
            print(f"fxrt fx_wrapper install skipped: no backend for {device}")
        return False

    if getattr(device_codegen, "fx_wrapper_codegen", None) is FxrtFxWrapper:
        _installed_devices.add(device)
        return True

    device_codegens[device] = _replace_fx_wrapper_codegen(
        device_codegen, FxrtFxWrapper
    )
    _installed_devices.add(device)
    return True


# Inductor settings the fx form needs: fx_wrapper selects it, and the two assert
# switches would otherwise write raw assert lines into the wrapper, which the fx
# form cannot lower.
_FX_WRAPPER_CONFIG = {
    "fx_wrapper": True,
    "size_asserts": False,
    "alignment_asserts": False,
}


def register_fx_wrapper(device: str = "npu", *, patch_config: bool = True) -> bool:
    """Register the fxrt fx_wrapper codegen and switch inductor to the fx form.

    This is the entry point for callers who want the fxrt runtime to execute what
    inductor produces. Importing fxrt registers nothing on its own, so the takeover
    is always an explicit call. On success the inductor settings the fx form needs
    are applied process-wide: ``fx_wrapper`` is turned on and ``size_asserts`` and
    ``alignment_asserts`` are turned off. Callers can still override them per
    compilation via ``torch._inductor.config.patch``. Pass ``patch_config=False``
    to register the codegen only and apply those settings yourself, for example
    within a ``torch._inductor.config.patch`` scope.

    The backend that runs the generated graph is chosen separately, with
    :func:`set_fx_wrapper_backend` or :func:`fx_wrapper_backend`; by default it
    is the fxrt runtime.

    Returns ``True`` when the codegen is registered, otherwise ``False`` when
    torch-npu/inductor is not available in the current environment.
    """

    if not _install_fx_wrapper_codegen(device):
        return False

    if patch_config:
        import torch._inductor.config as inductor_config

        for name, value in _FX_WRAPPER_CONFIG.items():
            setattr(inductor_config, name, value)
    return True


_EXPORTED_NAMES = (
    "AscendcBackend",
    "CompiledKernelBackend",
    "CompiledKernelFxConverter",
    "CompiledKernelSideTable",
    "CppPybindingBackend",
    "DvmBackend",
    "FxrtFxWrapper",
    "launch_compiled_kernel_functional",
    "launch_compiled_kernel_mutation",
    "compiled_kernel_side_table",
    "compiled_kernel_wrapper_functional",
    "compiled_kernel_wrapper_mutation",
    "fx_wrapper_backend",
    "get_fx_wrapper_backend",
    "register_fx_wrapper",
    "register_compiled_kernel_backend",
    "set_eager_debug",
    "set_fx_wrapper_backend",
)

__all__ = [name for name in _EXPORTED_NAMES if name in globals()]
