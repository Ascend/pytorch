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

"""Compiled-kernel HOP support used by the fxrt fx_wrapper."""
# pylint: disable=unused-argument

from __future__ import annotations

import threading
from collections.abc import Callable

import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree,
)

class CompiledKernelSideTable:
    """Global idx <-> compiled-callable table, mirroring torch's kernel table."""

    def __init__(self) -> None:
        self.id_to_kernel: dict[int, Callable] = {}
        self.kernel_to_id: dict[Callable, int] = {}
        self.lock = threading.Lock()

    def add_kernel(self, kernel: Callable) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]
            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    def get_kernel(self, idx: int) -> Callable:
        if idx not in self.id_to_kernel:
            raise AssertionError(f"Compiled kernel index {idx} not found")
        return self.id_to_kernel[idx]

    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}


compiled_kernel_side_table = CompiledKernelSideTable()


class CompiledKernelWrapperMutation(HigherOrderOperator):
    """HOP that preserves side-effecting compiled kernels in wrapper FX graphs."""

    def __init__(self) -> None:
        super().__init__("compiled_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
    ) -> None:
        return super().__call__(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=args,
        )


class CompiledKernelWrapperFunctional(HigherOrderOperator):
    """Functional HOP variant that returns cloned mutated tensor values."""

    def __init__(self) -> None:
        super().__init__("compiled_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
    ) -> dict[int, Tensor]:
        return super().__call__(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=args,
        )


compiled_kernel_wrapper_mutation = CompiledKernelWrapperMutation()
compiled_kernel_wrapper_functional = CompiledKernelWrapperFunctional()


def get_compiled_kernel(kernel_idx: int) -> Callable:
    """Resolve a compiled kernel by side-table index.

    This is the entry point FXRT's ``compiled_kernel_mutation`` op calls while
    it initializes, so the C++ side never has to reach into the side table object
    itself.
    """

    return compiled_kernel_side_table.get_kernel(int(kernel_idx))


def launch_compiled_kernel_mutation(kernel_idx: int, mutated_arg_indices, args) -> None:
    """Plain Python entry point for FXRT PythonCall fallback.

    The FX backend lowers the mutation HOP to FXRT's ``compiled_kernel_mutation``
    op (see :mod:`fxrt.compiled_kernel_adapter`). This function is the
    fallback for callers that can only go through PythonCall, which resolves
    callables by importing ``module_name`` then ``func_name`` -- something
    ``torch.ops.higher_order.compiled_kernel_wrapper_mutation`` cannot offer,
    being a dispatcher HOP object rather than an importable module path.
    """

    compiled_kernel_side_table.get_kernel(int(kernel_idx))(*tuple(args))


def launch_compiled_kernel_functional(kernel_idx: int, mutated_arg_indices, args):
    """Plain Python entry point for the functional compiled-kernel HOP."""

    return _run_functional(int(kernel_idx), tuple(mutated_arg_indices), tuple(args))


def _trace(proxy_mode, func_overload, node_args):
    """Replay a HOP call while tracing proxy outputs."""

    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def _mutation_dense(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    compiled_kernel_side_table.get_kernel(kernel_idx)(*args)


@compiled_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def _mutation_fake(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    with mode:
        pass


@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _mutation_meta(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    pass


@compiled_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def _mutation_proxy(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    """Trace the mutation HOP into the wrapper FX graph."""

    _trace(
        mode,
        compiled_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        },
    )


@compiled_kernel_wrapper_mutation.py_functionalize_impl
def _mutation_functionalize(
    ctx, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    """Functionalize a mutation HOP by replacing mutated input tensors."""

    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        new_vals = compiled_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=unwrapped_args,
        )
    for i, output_arg in new_vals.items():
        input_arg = args[i]
        ctx.replace(input_arg, output_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)


def _run_functional(kernel_idx, mutated_arg_indices, args):
    """Run a mutation kernel on cloned inputs and return the mutated clones."""

    new_args = list(args)
    clones: dict[int, Tensor] = {}
    for i in mutated_arg_indices:
        clones[i] = clone_preserve_strides(args[i])
        new_args[i] = clones[i]
    compiled_kernel_side_table.get_kernel(kernel_idx)(*new_args)
    return clones


@compiled_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def _functional_dense(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return _run_functional(kernel_idx, mutated_arg_indices, args)


@compiled_kernel_wrapper_functional.py_impl(FakeTensorMode)
def _functional_fake(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    with mode:
        return {i: clone_preserve_strides(args[i]) for i in mutated_arg_indices}


@compiled_kernel_wrapper_functional.py_impl(DispatchKey.Meta)
def _functional_meta(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return {i: clone_preserve_strides(args[i]) for i in mutated_arg_indices}


@compiled_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def _functional_proxy(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return _trace(
        mode,
        compiled_kernel_wrapper_functional,
        {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        },
    )


@compiled_kernel_wrapper_functional.py_functionalize_impl
def _functional_functionalize(
    ctx, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    """Functionalize a functional HOP by wrapping cloned tensor outputs."""

    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        outputs = compiled_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=unwrapped_args,
        )
    return ctx.wrap_tensors(outputs)


for _hop in (compiled_kernel_wrapper_mutation, compiled_kernel_wrapper_functional):
    _hop.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.ADInplaceOrView)
    _hop.fallthrough(DispatchKey.BackendSelect)
    _hop.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.AutogradCUDA)
    _hop.fallthrough(DispatchKey.AutogradCPU)



__all__ = [
    "CompiledKernelSideTable",
    "compiled_kernel_side_table",
    "compiled_kernel_wrapper_functional",
    "compiled_kernel_wrapper_mutation",
    "get_compiled_kernel",
    "launch_compiled_kernel_functional",
    "launch_compiled_kernel_mutation",
]
