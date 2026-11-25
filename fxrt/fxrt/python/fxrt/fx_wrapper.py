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

The Inductor ``fxrt`` option installs the NPU FX wrapper for the duration of
compilation. The wrapper routes the generated host FX GraphModule to FXRT by
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
import ast
import contextlib
import copy
import dataclasses
import inspect
import operator
import os
from collections.abc import Callable
from typing import Any

import sympy

import torch
from torch._inductor import ir
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.codegen.common import FileBackedGraphModule
from torch._inductor.codegen.wrapper import CommentLine, PythonWrapperCodegen, WrapperLine
from torch.utils._sympy.functions import FloorDiv
from torch._inductor.codegen.wrapper_fxir import FxConverter, WrapperFxCodegen, replace_floor_div
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from fxrt import compiled_kernel_hop as _compiled_kernel_hop
from fxrt.view_offset import (
    ViewOffsetSemantics, is_as_strided, set_view_offset_semantics, view_offset_semantics,
)
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


def _lookup_args_takes_raw_args() -> bool:
    """torch 2.15 FxConverter._lookup_args accepts raw_args; 2.13 does not."""
    return "raw_args" in inspect.signature(FxConverter._lookup_args).parameters


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
    """Return the configured process-wide fx_wrapper backend."""

    return _active_fx_backend


def set_fx_wrapper_backend(gm_backend: "Callable | None") -> "Callable | None":
    """Set the process-wide backend used by the installed fx_wrapper."""

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


def _relative_as_strided_metadata(input, size, stride, storage_offset=None):
    """Give node-creation hooks the layout of a wrapper reinterpretation."""
    offset = input.storage_offset() + (0 if storage_offset is None else storage_offset)
    return torch.as_strided(input, size, stride, offset)


def _propagate_host_graph_metadata(gm, mode, inputs):
    """Re-evaluate fake layouts without erasing stamped view semantics."""
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    relative = []
    try:
        for node in gm.graph.nodes:
            if is_as_strided(node) and view_offset_semantics(node) == ViewOffsetSemantics.IMMEDIATE_RELATIVE:
                relative.append((node, node.op, node.target))
                node.op, node.target = "call_function", _relative_as_strided_metadata
        return FakeTensorProp(gm, mode).propagate(*inputs)
    finally:
        for node, op, target in relative:
            node.op, node.target = op, target


class CompiledKernelFxConverter(FxConverter):
    """FxConverter that routes non-Triton kernels through registered backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernels: dict[str, tuple[int, CompiledKernelBackend]] = {}

    def _create_as_strided(self, input_node, size, stride, offset):
        """Attach immediate-buffer offset provenance to wrapper-generated views."""
        # WrapperFxCodegen's node-creation hook executes the target on fake
        # inputs. Public as_strided would compute an absolute fake offset before
        # we can stamp the relative contract, and kernel-arg reconstruction
        # would subsequently freeze that incorrect layout. Let the hook evaluate
        # the relative operation, then retain the existing host-graph spelling.
        if isinstance(offset, sympy.Expr):
            offset = replace_floor_div(offset)
        node = self.gm.graph.call_function(
            _relative_as_strided_metadata,
            args=(input_node, self._generate_sym_nodes(size),
                  self._generate_sym_nodes(stride), self._generate_sym_node(offset)),
        )
        node.target = torch.as_strided
        return set_view_offset_semantics(node, ViewOffsetSemantics.IMMEDIATE_RELATIVE)

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
        args = tuple(
            self._lift_sym_args(a)
            for a in self._lookup_compiled_kernel_args(line.call_args, line.raw_args)
        )
        mutated_arg_indices = getattr(line, "_fxrt_mutated_arg_indices", None)
        if mutated_arg_indices is None:
            mutated_arg_indices = backend.mutated_arg_indices(line)
        self.gm.graph.call_function(
            compiled_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": idx,
                "mutated_arg_indices": tuple(mutated_arg_indices),
                "args": args,
            },
        )

    def _resolve_symbol_by_name(self, name):
        """Return the fx node for a bare symbol name, matching on name only.

        Autofuse kernels carry symbolic shape extents in ``call_args`` as plain
        strings ('s13' rather than a SymbolicCallArg), and such a name is not a
        buffer name unless the symbol is also a graph input. The converter's
        ``expr_to_proxy`` keys are Symbols carrying inductor's assumptions
        (integer/nonnegative), which do not compare equal to a freshly built
        plain Symbol, so the dict lookup has to be done by name.
        """
        for expr, proxy in self.expr_to_proxy.items():
            if isinstance(expr, sympy.Symbol) and expr.name == name:
                return proxy.node
        return None

    def _pre_resolve_symbol_args(self, args):
        """Turn bare symbol names among `args` into fx nodes before upstream sees them.

        Every torch version fails on these names, just differently: 2.13/2.14
        index ``buffer_to_node`` and raise KeyError, 2.15 asserts. Resolving here
        keeps the difference from reaching the caller at all.

        A name that is not a symbol either is left untouched, so the upstream
        converter still reports it in its own terms. Names that do resolve as
        buffers -- a symbol that is also a graph input is both -- stay strings,
        because upstream already maps those correctly and to the same node.
        """
        resolved = list(args)
        for index, arg in enumerate(resolved):
            if not isinstance(arg, str) or arg in self.buffer_to_node:
                continue
            node = self._resolve_symbol_by_name(arg)
            if node is not None:
                resolved[index] = node
        return tuple(resolved)

    def _lookup_compiled_kernel_args(self, call_args, raw_args):
        """Resolve kernel args on torch 2.13 and 2.15.

        2.15 forwards raw_args into FxConverter._lookup_args. 2.13 only maps
        buffer names; a missing name is a view such as ReinterpretView, which
        raw_args still carries, so materialize that node locally.

        Autofuse kernels also put symbolic shape extents in ``call_args`` as
        bare strings ('s13'), which are neither buffer names nor IRNode
        raw_args; those are resolved to their fx nodes first, below.
        """
        call_args = self._pre_resolve_symbol_args(call_args)
        if _lookup_args_takes_raw_args():
            return self._lookup_args(call_args, raw_args)
        if raw_args is None:
            raw_args = (None,) * len(call_args)
        looked_up = []
        for arg, raw_arg in zip(call_args, raw_args):
            if isinstance(arg, str) and arg not in self.buffer_to_node:
                if not isinstance(raw_arg, ir.IRNode):
                    raise AssertionError(f"Unrecognized arg type: {type(raw_arg)}")
                looked_up.append(self._generate_buffer(raw_arg))
            else:
                looked_up.append(self._lookup_args((arg,))[0])
        return tuple(looked_up)

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
        input_edges = tuple(self._generate_buffer(input_buffer) for input_buffer in kernel.inputs)
        if hasattr(kernel, "unflatten_args"):
            call_args, _ = kernel.unflatten_args(input_edges, kernel.constant_args)
        else:
            call_args = input_edges + tuple(kernel.constant_args)
        call_args = self._lift_sym_args(call_args)

        call_kwargs = {
            keyword: self._generate_buffer(value) if isinstance(value, ir.IRNode) else value
            for keyword, value in kernel.kwargs.items()
        }
        call_kwargs = self._lift_sym_args(call_kwargs)

        output_name = None
        if isinstance(kernel, ir.ExternKernelOut):
            call_kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            output_name = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(f"Unrecognized output layout: {kernel.layout}")

        call_node = self.gm.graph.call_function(
            kernel.op_overload,
            args=call_args,
            kwargs=call_kwargs,
        )

        if output_name:
            if "out" in call_kwargs:
                raise RuntimeError(
                    f"Extern kernel '{kernel}' has both result and out kwarg. Expected only one."
                )
            call_node.name = output_name
            self.buffer_to_node[output_name] = call_node


# Torchair's no-epilogue matmul fallback writes a Python statement for the stock
# wrapper. These are the only calls that statement is allowed to contain.
_ATEN_SOURCE_OPS = {
    "mm": torch.ops.aten.mm.default,
    "addmm": torch.ops.aten.addmm.default,
    "bmm": torch.ops.aten.bmm.default,
    "baddbmm": torch.ops.aten.baddbmm.default,
}
_SYMPY_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: lambda left, right: FloorDiv(left, right),
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _aten_op_name(func: ast.AST):
    """Return the aten op name of ``torch.ops.aten.<op>.default``, if that is ``func``."""
    if not isinstance(func, ast.Attribute) or func.attr != "default":
        return None
    op_attr = func.value
    if not isinstance(op_attr, ast.Attribute):
        return None
    aten_attr = op_attr.value
    if not isinstance(aten_attr, ast.Attribute) or aten_attr.attr != "aten":
        return None
    ops_attr = aten_attr.value
    if not isinstance(ops_attr, ast.Attribute) or ops_attr.attr != "ops":
        return None
    if not isinstance(ops_attr.value, ast.Name) or ops_attr.value.id != "torch":
        return None
    return op_attr.attr


def _extern_kernel_name(func: ast.AST):
    """Return the op name of ``extern_kernels.<op>``, if that is ``func``."""
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "extern_kernels":
        return None
    if func.attr not in _ATEN_SOURCE_OPS:
        return None
    return func.attr


def _parse_sym_expr(node: ast.AST):
    """Parse one size/stride/offset expression printed by PythonPrinter."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return ("int", node.value)
    if isinstance(node, ast.Name):
        return ("sym", node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _parse_sym_expr(node.operand)
        if operand is None:
            return None
        return ("neg", operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _SYMPY_BINOPS:
        left = _parse_sym_expr(node.left)
        right = _parse_sym_expr(node.right)
        if left is None or right is None:
            return None
        return (type(node.op), left, right)
    return None


def _parse_sym_tuple(node: ast.AST):
    if not isinstance(node, ast.Tuple):
        return None
    items = tuple(_parse_sym_expr(elt) for elt in node.elts)
    if any(item is None for item in items):
        return None
    return items


def _parse_reinterpret(node: ast.AST):
    if not isinstance(node, ast.Call) or node.keywords or getattr(node, "starargs", None):
        return None
    func = node.func
    if not isinstance(func, ast.Name) or func.id != "reinterpret_tensor":
        return None
    if len(node.args) != 4 or not isinstance(node.args[0], ast.Name):
        return None
    size = _parse_sym_tuple(node.args[1])
    stride = _parse_sym_tuple(node.args[2])
    offset = _parse_sym_expr(node.args[3])
    if size is None or stride is None or offset is None:
        return None
    return ("view", node.args[0].id, size, stride, offset)


def _parse_numeric_keywords(keywords):
    values = {}
    for keyword in keywords:
        if keyword.arg not in ("beta", "alpha"):
            return None
        value = keyword.value
        if not isinstance(value, ast.Constant) or isinstance(value.value, bool):
            return None
        if not isinstance(value.value, (int, float)):
            return None
        values[keyword.arg] = value.value
    return values


def _parse_fallback_value(node: ast.AST):
    """Parse one argument or the whole RHS of a matmul fallback statement."""
    viewed = _parse_reinterpret(node)
    if viewed is not None:
        return viewed
    if not isinstance(node, ast.Call) or any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    op_name = _aten_op_name(node.func)
    if op_name == "reshape":
        if node.keywords or len(node.args) != 2:
            return None
        inner = _parse_fallback_value(node.args[0])
        size = _parse_sym_tuple(node.args[1])
        if inner is None or size is None:
            return None
        return ("reshape", inner, size)
    if op_name not in _ATEN_SOURCE_OPS:
        return None
    kwargs = _parse_numeric_keywords(node.keywords)
    if kwargs is None:
        return None
    args = tuple(_parse_fallback_value(arg) for arg in node.args)
    if any(arg is None for arg in args):
        return None
    return ("call", op_name, args, kwargs)


def _parse_extern_kernel_stmt(node: ast.Call):
    op_name = _extern_kernel_name(node.func)
    if op_name is None or any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    out_name = None
    numeric = []
    for keyword in node.keywords:
        if keyword.arg == "out":
            if not isinstance(keyword.value, ast.Name):
                return None
            out_name = keyword.value.id
            continue
        numeric.append(keyword)
    kwargs = _parse_numeric_keywords(numeric)
    if kwargs is None or out_name is None:
        return None
    args = tuple(_parse_fallback_value(arg) for arg in node.args)
    if any(arg is None for arg in args):
        return None
    return out_name, ("call", op_name, args, kwargs)


def _parse_aten_source_fallback(source: str):
    """Recover a torchair matmul fallback statement, or return None."""
    try:
        tree = ast.parse(source.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    stmt = tree.body[0]
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        expr = _parse_fallback_value(stmt.value)
        if expr is None or expr[0] not in ("call", "reshape"):
            return None
        return stmt.targets[0].id, expr
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return _parse_extern_kernel_stmt(stmt.value)
    return None


def _lookup_size_symbol(converter, name: str):
    for expr in converter.expr_to_proxy:
        if getattr(expr, "name", None) == name:
            return expr
    raise NotImplementedError(f"fx_wrapper has no size symbol {name} for a matmul fallback view")


def _eval_sym_expr(converter, expr):
    kind = expr[0]
    if kind == "int":
        return expr[1]
    if kind == "sym":
        return _lookup_size_symbol(converter, expr[1])
    if kind == "neg":
        return -_eval_sym_expr(converter, expr[1])
    binop = _SYMPY_BINOPS.get(kind)
    if binop is None:
        raise NotImplementedError(f"fx_wrapper cannot evaluate size expression {expr!r}")
    return binop(_eval_sym_expr(converter, expr[1]), _eval_sym_expr(converter, expr[2]))


def _emit_fallback_value(converter, expr):
    kind = expr[0]
    if kind == "view":
        _, name, size, stride, offset = expr
        base = converter.buffer_to_node[name]
        return converter._create_as_strided(
            base,
            tuple(_eval_sym_expr(converter, item) for item in size),
            tuple(_eval_sym_expr(converter, item) for item in stride),
            _eval_sym_expr(converter, offset),
        )
    if kind == "reshape":
        _, inner, size = expr
        dims = [
            converter._generate_sym_node(_eval_sym_expr(converter, item)) for item in size
        ]
        return converter.gm.graph.call_function(
            torch.ops.aten.reshape.default,
            args=(_emit_fallback_value(converter, inner), dims),
        )
    _, op_name, args, kwargs = expr
    emitted = tuple(_emit_fallback_value(converter, arg) for arg in args)
    return converter.gm.graph.call_function(
        _ATEN_SOURCE_OPS[op_name],
        args=emitted,
        kwargs=dict(kwargs),
    )


def _emit_aten_source_fallback(converter, line) -> None:
    parsed = _parse_aten_source_fallback(line.source)
    if parsed is None:
        raise NotImplementedError(
            f"fx_wrapper cannot lower raw source written to the wrapper: {line.source!r}"
        )
    out_name, expr = parsed
    node = _emit_fallback_value(converter, expr)
    if not any(existing.name == out_name for existing in converter.gm.graph.nodes):
        node.name = out_name
    converter.buffer_to_node[out_name] = node


@dataclasses.dataclass
class AtenSourceFallbackLine(WrapperLine):
    """Structured form of a raw matmul fallback statement.

    The Python wrapper executes the statement as source. The fx wrapper rebuilds
    it as aten calls plus ``as_strided`` views, which is what ``reinterpret_tensor``
    lowers to.
    """

    source: str

    def codegen(self, code) -> None:
        code.writeline(self.source)

    def codegen_fx(self, converter):
        def emit(line):
            _emit_aten_source_fallback(converter, line)

        return emit




def _merge_embedding_cat_runs(gm) -> int:
    """Merge contiguous runs of one-hot embedding lookups feeding one aten.cat.

    N separate aten.embedding(W_i, idx_i) dispatches cost ~17us each on the
    host async queue while each kernel runs only ~2.5us on device. Each run is
    rewritten (values preserved exactly, no arithmetic on data):

        stack(idx_0..idx_n)            i64[bs, n]
        + offsets(table row ranges)    i64[bs, n]
        embedding(cat(W_0..W_n), .)    f32[bs, n, d]   (per dim-group)
        flatten(1)                     f32[bs, n*d]
        cat(groups) + index_select(perm)   back to original column order

    Only runs of length >= FXRT_MERGE_EMB_MIN_RUN whose members are
    aten.embedding(weight_ph, indices_ph) with rank-1 indices and small vocab
    are merged. Returns the number of embedding nodes collapsed.
    """
    import os
    from collections import OrderedDict

    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    min_run = int(os.environ.get("FXRT_MERGE_EMB_MIN_RUN", "8"))
    max_vocab = int(os.environ.get("FXRT_MERGE_EMB_MAX_VOCAB", "1024"))

    def emb_info(n):
        if not isinstance(n, Node) or n.op != "call_function":
            return None
        if n.target is not aten.embedding.default:
            return None
        if len(n.args) < 2:
            return None
        w, idx = n.args[0], n.args[1]
        if not (isinstance(w, Node) and w.op == "placeholder"):
            return None
        if not (isinstance(idx, Node) and idx.op == "placeholder"):
            return None
        wm, im = w.meta.get("val"), idx.meta.get("val")
        if wm is None or im is None:
            return None
        if wm.dim() != 2 or im.dim() != 1:
            return None
        vocab, dim = wm.shape[0], wm.shape[1]
        if not isinstance(vocab, int) or not isinstance(dim, int):
            return None
        if vocab > max_vocab:
            return None
        return {"w": w, "idx": idx, "vocab": vocab, "dim": dim,
                "dev": wm.device, "idtype": im.dtype}


    def _dbg(*a):
        if os.environ.get("FXRT_MERGE_TRACE", "0") == "1":
            print("[fxrt_merge_dbg]", *a)

    def view_base(n):
        if (not isinstance(n, Node) or n.op != "call_function"
                or n.target is not torch.as_strided or len(n.args) != 4):
            return None
        base, size, stride, offset = n.args
        if not isinstance(base, Node) or not isinstance(offset, int) or offset != 0:
            _dbg("view_rej offset/base", offset)
            return None
        bm = base.meta.get("val")
        if bm is None or bm.dim() != len(size) or len(stride) != len(size):
            _dbg("view_rej meta", bm is None, len(size))
            return None
        # exact symbolic identity first
        try:
            if tuple(size) == tuple(bm.shape) and tuple(stride) == tuple(bm.stride()):
                return base
        except Exception:
            pass
        # structural identity: contiguous strides w.r.t. size and matching
        # static dims. The leading dim may carry a re-bound duplicate symbol
        # (e.g. s95 vs s0) inserted by fxrt preprocessing; such views are
        # pure aliases of the same storage, so treat them as identity.
        bshape = list(bm.shape)
        if not all(isinstance(s, int) for s in stride):
            _dbg("view_rej sym_stride")
            return None
        contig, acc = [1], 1
        for d in reversed(size[1:]):
            if not isinstance(d, int):
                _dbg("view_rej sym_inner_dim")
                return None
            acc = acc * d
            contig.append(acc)
        contig.reverse()
        if list(stride) != contig:
            _dbg("view_rej stride", stride, contig)
            return None
        for i in range(1, len(size)):
            if not isinstance(bshape[i], int) or bshape[i] != size[i]:
                _dbg("view_rej dim", i, bshape[i], size[i])
                return None
        if (isinstance(size[0], int) and isinstance(bshape[0], int)
                and size[0] != bshape[0]):
            _dbg("view_rej static_leading", size[0], bshape[0])
            return None
        return base

    def slot_info(n, cat_node):
        chain = []
        cur = n
        while True:
            b = view_base(cur)
            if b is None:
                break
            chain.append(cur)
            cur = b
        info = emb_info(cur)
        if info is None:
            _dbg("slot_not_emb", cur.op, cur.target)
            return None
        if len(n.users) != 1 or cat_node not in n.users:
            _dbg("slot_rej users", len(n.users))
            return None
        for v in chain:
            if len(v.users) != 1:
                return None
        if chain and (len(cur.users) != 1 or chain[-1] not in cur.users):
            return None
        info["chain"] = chain
        info["base"] = cur
        return info

    def build_merged(before, members):
        g = gm.graph
        dev = members[0]["dev"]
        groups = OrderedDict()
        for m in members:
            groups.setdefault(m["dim"], []).append(m)
        parts = []
        state = gm.__dict__.setdefault("_fxrt_merge_consts", {"n": 0})

        def const_node(value):
            n = state["n"]
            state["n"] += 1
            name = f"_fxrt_merge_const_{n}"
            gm.register_buffer(name, value, persistent=False)
            return g.get_attr(name)

        for dim, mem in groups.items():
            offs = [0]
            for v in [x["vocab"] for x in mem[:-1]]:
                offs.append(offs[-1] + v)
            offs_t = torch.tensor(offs, dtype=members[0]["idtype"], device=dev)
            with g.inserting_before(before):
                offs_n = const_node(offs_t)
                stacked = g.call_function(aten.stack.default, ([x["idx"] for x in mem], 1))
                m_idx = g.call_function(aten.add.Tensor, (stacked, offs_n))
                flat_w = g.call_function(aten.cat.default, ([x["w"] for x in mem], 0))
                emb = g.call_function(aten.embedding.default, (flat_w, m_idx))
                fl = g.call_function(torch.flatten, (emb, 1))
            parts.append((fl, dim, mem))
        if len(parts) == 1:
            return parts[0][0]
        gpos = {}
        gcum = 0
        for dim, mem in groups.items():
            for j, _m in enumerate(mem):
                gpos[id(_m)] = gcum + j * dim
            gcum += len(mem) * dim
        perm = []
        for m in members:
            base = gpos[id(m)]
            perm.extend(range(base, base + m["dim"]))
        perm_t = torch.tensor(perm, dtype=members[0]["idtype"], device=dev)
        with g.inserting_before(before):
            perm_n = const_node(perm_t)
            catg = g.call_function(aten.cat.default, ([p[0] for p in parts], 1))
            return g.call_function(torch.index_select, (catg, 1, perm_n))

    merged = 0
    dead_nodes = []
    for cat_node in [n for n in gm.graph.nodes
                     if n.op == "call_function" and n.target is aten.cat.default]:
        if cat_node.args and isinstance(cat_node.args[0], (list, tuple)):
            tensors = list(cat_node.args[0])
            dim_arg = cat_node.args[1] if len(cat_node.args) > 1 else 0
        elif cat_node.kwargs.get("tensors"):
            tensors = list(cat_node.kwargs["tensors"])
            dim_arg = cat_node.kwargs.get("dim", 0)
        else:
            continue
        if dim_arg != 1:
            continue
        infos = [slot_info(n, cat_node) for n in tensors]
        runs, i = [], 0
        while i < len(infos):
            if infos[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < len(infos) and infos[j + 1] is not None:
                j += 1
            if j - i + 1 >= min_run:
                runs.append((i, j))
            i = j + 1
        _dbg("cat", len(tensors), "runs", runs)
        if not runs:
            continue
        new_tensors = list(tensors)
        # apply right-to-left so earlier run indices stay valid after the
        # list shrinks
        for a, b in reversed(runs):
            new_tensors[a:b + 1] = [build_merged(cat_node, infos[a:b + 1])]
            merged += b - a + 1
            for k in range(a, b + 1):
                dead_nodes.extend(infos[k]["chain"])
                dead_nodes.append(infos[k]["base"])
        if cat_node.args and isinstance(cat_node.args[0], (list, tuple)):
            cat_node.args = (new_tensors, dim_arg)
        else:
            cat_node.kwargs = dict(cat_node.kwargs, tensors=new_tensors)
    if merged:
        # NB: never graph.eliminate_dead_code() here -- the compiled-kernel HOP
        # mutates args in place and its return value has no users, so generic
        # DCE would delete live AF kernels.
        for d in dead_nodes:
            gm.graph.erase_node(d)
        gm.recompile()
        if os.environ.get("FXRT_MERGE_DEBUG"):
            with open("/tmp/merged_gm_debug.py", "w") as f:
                f.write(gm.code)
            consts = {k: v for k, v in gm.__dict__.items()
                      if k.startswith("_fxrt_merge_const_")}
            torch.save(consts, "/tmp/merged_consts.pt")
    return merged


def _try_merge_embedding_cat_runs(gm) -> int:
    """Apply the embedding merge only when the selected executor can consume it."""
    if os.environ.get("FXRT_MERGE_EMBEDDINGS", "auto").lower() in {"0", "false", "off"}:
        return 0

    executor = os.environ.get("TORCH_NPU_FXRT_EXECUTOR", "default")
    if executor == "GERT":
        # This rewrite adds tensor-valued get_attr producers to add/index_select.
        # The converter currently exports those as unnamed value nodes, which its
        # tensor layout planner cannot resolve to a TensorRef.
        print(
            "[fxrt_fx] skip embedding merge: GERT cannot resolve tensor get_attr producers",
            flush=True,
        )
        return 0
    if executor != "default":
        return 0
    return _merge_embedding_cat_runs(gm)


class FxrtFxWrapper(WrapperFxCodegen):
    """WrapperFxCodegen that sends the host FX graph to the configured backend."""

    # AscendC compiles autofuse_adapter.so only when the active wrapper asks for
    # the FXRT C ABI. The converter loads that library beside wrapper.so.
    requires_autofuse_adapter = True

    def writeline(self, line):
        """Accept a wrapper line, tolerating raw source strings.

        Third-party fusion backends sometimes writeline raw source strings
        straight into the wrapper. The fx converter only lowers structured
        WrapperLine objects, so a bare str aborts conversion. Comments and
        blank lines carry no semantics and are dropped (the converter already
        ignores CommentLine). A no-epilogue matmul fallback is recovered as aten
        calls; any other raw source cannot be represented as FX and must fail
        loudly rather than silently miscompile.
        """
        if isinstance(line, str):
            nonblank = [s for s in line.splitlines() if s.strip()]
            if all(s.lstrip().startswith("#") for s in nonblank):
                line = CommentLine(line)
            elif _parse_aten_source_fallback(line) is not None:
                line = AtenSourceFallbackLine(line)
            else:
                raise NotImplementedError(
                    f"fx_wrapper cannot lower raw source written to the wrapper: {line!r}"
                )
        super().writeline(line)

    def generate_kernel_call(
        self, kernel_name, call_args, *, fxrt_mutated_arg_indices=None, **kwargs
    ):
        """Attach optional backend-neutral output positions to a compiled call."""
        super().generate_kernel_call(kernel_name, call_args, **kwargs)
        if fxrt_mutated_arg_indices is not None:
            self.lines[-1]._fxrt_mutated_arg_indices = tuple(fxrt_mutated_arg_indices)

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

        # Synthetic views in an Inductor host graph use immediate-buffer
        # offsets even when the graph contains no compiled kernel call.
        gm.meta["_fxrt_inductor_host_graph"] = True
        simplify_duplicate_symints(gm)
        merged_embs = _try_merge_embedding_cat_runs(gm)
        if merged_embs:
            print(f"[fxrt_fx] merged {merged_embs} embedding lookups into grouped flat lookups")
            try:
                from torch._subclasses.fake_tensor import FakeTensorMode
                mode = FakeTensorMode(allow_non_fake_inputs=True)
                with mode:
                    _propagate_host_graph_metadata(
                        gm, mode, [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[fxrt_fx] fake re-propagation skipped: {e}")
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


# Inductor settings the fx form needs: fx_wrapper selects it, and the two assert
# switches would otherwise write raw assert lines into the wrapper, which the fx
# form cannot lower.
_FX_WRAPPER_CONFIG = {
    "fx_wrapper": True,
    "size_asserts": False,
    "alignment_asserts": False,
}


def _compile_inductor_with_fxrt(compile_fn, wrapper, gm, inputs):
    """Compile with FX wrappers and use FXRT for the NPU wrapper."""
    from torch._inductor.codegen.common import device_codegens

    # Get current codegen for npu.
    if "npu" not in device_codegens:
        from torch_npu.utils._dynamo import register_inductor_npu

        register_inductor_npu()
    codegen = device_codegens.get("npu")
    if codegen is None:
        raise RuntimeError("FXRT requires an Inductor codegen registered for NPU.")
    previous_wrapper = codegen.fx_wrapper_codegen

    try:
        device_codegens["npu"] = _replace_fx_wrapper_codegen(codegen, FxrtFxWrapper)
        fxrt_wrapper = copy.copy(wrapper)
        fxrt_wrapper.config = {**wrapper.config, **_FX_WRAPPER_CONFIG, "cpp_wrapper": False}
        return compile_fn(fxrt_wrapper, gm, inputs)
    finally:
        # The fallback in _replace_fx_wrapper_codegen can update the object in place.
        if codegen.fx_wrapper_codegen is not previous_wrapper:
            codegen.fx_wrapper_codegen = previous_wrapper
        device_codegens["npu"] = codegen


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
    "register_compiled_kernel_backend",
    "set_eager_debug",
    "set_fx_wrapper_backend",
)

__all__ = [name for name in _EXPORTED_NAMES if name in globals()]
