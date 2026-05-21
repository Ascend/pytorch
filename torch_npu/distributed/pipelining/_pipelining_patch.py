# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import importlib
from typing import Any, Optional

import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten

from torch_npu.distributed.pipelining._pipelining_fx import (
    RealPyTreeInfo,
    inject_pipeline_splits,
    populate_boundary_metadata,
    symbolic_trace_for_pipelining,
)


_PATCHED_ATTR = "_torch_npu_pipelining_patch_applied"
_PATCHED_MODIFY_GRAPH_DEVICE_ATTR = "_torch_npu_modify_graph_device_patch_applied"
_ORIGINAL_PIPELINE = None
_IR_MODULE = None


class _FXExportedProgramShim:
    """Provide the ExportedProgram fields required by Pipe construction."""

    constants: dict[str, Any] = {}

    def __init__(self, graph_module: fx.GraphModule) -> None:
        self._graph_module = graph_module

    def module(self, *args, **kwargs) -> fx.GraphModule:
        return self._graph_module


@contextlib.contextmanager
def _preserve_fx_call_modules_for_from_traced(ir_module):
    """Preserve FX `call_module` submodules during Pipe construction."""

    original_split_module = ir_module.split_module
    original_outline_submodules = ir_module._outline_submodules
    graph_to_submodule: dict[int, fx.GraphModule] = {}

    def split_module_capture(*args, **kwargs):
        """Record split submodules for later lookup."""
        split = original_split_module(*args, **kwargs)
        graph_to_submodule.clear()
        for _, submodule in split.named_children():
            if isinstance(submodule, fx.GraphModule):
                graph_to_submodule[id(submodule.graph)] = submodule
        return split

    def outline_submodules_for_fx(graph):
        """Return existing FX submodules when a split graph keeps module calls."""
        submodule = graph_to_submodule.get(id(graph))
        if submodule is not None and any(
            node.op == "call_module" for node in graph.nodes
        ):
            return submodule
        return original_outline_submodules(graph)

    ir_module.split_module = split_module_capture
    ir_module._outline_submodules = outline_submodules_for_fx
    try:
        yield
    finally:
        ir_module.split_module = original_split_module
        ir_module._outline_submodules = original_outline_submodules


def _from_fx_traced(
    ir_module,
    module: torch.nn.Module,
    traced: fx.GraphModule,
    split_policy,
):
    """Build a native PyTorch Pipe from an FX-traced GraphModule."""

    shim = _FXExportedProgramShim(traced)
    with _preserve_fx_call_modules_for_from_traced(ir_module):
        pipe = ir_module.Pipe._from_traced(
            module,
            shim,
            ir_module.MultiUseParameterConfig.REPLICATE,
            output_loss_value_spec=None,
            split_policy=split_policy,
        )
    _prune_none_root_placeholders(pipe)
    return pipe


def _prune_none_root_placeholders(pipe) -> None:
    """Remove stage placeholders whose value is a constant Python None."""

    if not hasattr(pipe, "get_stage_module") or not hasattr(pipe, "num_stages"):
        return

    for stage_index in range(pipe.num_stages):
        stage_module = pipe.get_stage_module(stage_index)
        if not isinstance(stage_module, fx.GraphModule):
            continue

        graph = stage_module.graph
        changed = False
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        for placeholder in placeholders:
            if placeholder.meta.get("val", "__missing__") is None:
                placeholder.replace_all_uses_with(None)
                graph.erase_node(placeholder)
                changed = True

        if changed:
            graph.lint()
            stage_module.recompile()


def _quiet_modify_graph_op_device(
    gm: fx.GraphModule,
    new_device: torch.device,
) -> None:
    """Update graph-level device kwargs for nested FX graph modules."""

    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and "device" in node.kwargs:
            if node.kwargs["device"] != new_device:
                node.update_kwarg("device", new_device)
                modified = True
        elif node.op == "call_module":
            submod = gm.get_submodule(node.target)
            if isinstance(submod, fx.GraphModule):
                _quiet_modify_graph_op_device(submod, new_device)
                continue

            graph_module = getattr(submod, "graph_module", None)
            if isinstance(graph_module, fx.GraphModule):
                _quiet_modify_graph_op_device(graph_module, new_device)

    if modified:
        gm.recompile()


setattr(_quiet_modify_graph_op_device, _PATCHED_MODIFY_GRAPH_DEVICE_ATTR, True)


def _apply_quiet_graph_device_patch(ir_module) -> None:
    original = getattr(ir_module, "_modify_graph_op_device", None)
    if original is None or getattr(original, _PATCHED_MODIFY_GRAPH_DEVICE_ATTR, False):
        return

    ir_module._modify_graph_op_device = _quiet_modify_graph_op_device


def _pipeline_fx(
    ir_module,
    module: torch.nn.Module,
    mb_args: tuple[Any, ...],
    mb_kwargs: Optional[dict[str, Any]],
    split_spec,
    split_policy,
    atomic_units: Optional[list[str]],
):
    """Trace with FX, inject split markers, and construct a native Pipe."""

    traced = symbolic_trace_for_pipelining(
        module,
        example_args=mb_args,
        example_kwargs=mb_kwargs,
        atomic_units=atomic_units,
    )
    warm, module = populate_boundary_metadata(
        traced,
        mb_args,
        mb_kwargs,
        eager_root=module,
        sync_per_node=True,
        use_no_grad=True,
    )
    _, in_spec = tree_flatten(mb_args)
    _, out_spec = tree_flatten(warm.meta_output)
    traced.graph._codegen.pytree_info = RealPyTreeInfo(
        in_spec=in_spec,
        out_spec=out_spec,
    )

    if split_spec is not None:
        traced = inject_pipeline_splits(
            traced,
            split_spec=split_spec,
            pipe_split_op=ir_module.aten_pipe_split_alias,
        )

    return _from_fx_traced(
        ir_module,
        module=module,
        traced=traced,
        split_policy=split_policy,
    )


def pipeline(
    module: torch.nn.Module,
    mb_args: tuple[Any, ...],
    mb_kwargs: Optional[dict[str, Any]] = None,
    split_spec=None,
    split_policy=None,
    mode: Optional[str] = None,
    atomic_units: Optional[list[str]] = None,
):
    """
    Split a module into pipeline stages with export or FX tracing.

    Args:
        module (torch.nn.Module): Module to split into pipeline stages.
        mb_args (tuple[Any, ...]): Example positional microbatch inputs used
            for tracing and shape propagation.
        mb_kwargs (Optional[dict[str, Any]]): Example keyword microbatch inputs
            used for tracing and shape propagation.
        split_spec: Mapping from module names to split points.
        split_policy: Callable split policy used by PyTorch pipeline.
        mode (Optional[str]): Tracing backend. ``None`` and ``"export"`` use
            PyTorch export tracing. ``"fx"`` uses FX tracing.
        atomic_units (Optional[list[str]]): Module names treated as FX leaf
            modules when ``mode="fx"``.

    Returns:
        Pipe: A pipeline IR object whose stages can be built by
        ``Pipe.build_stage``.

    Example:
        >>> from torch.distributed.pipelining import SplitPoint
        >>> from torch_npu.distributed.pipelining import pipeline
        >>>
        >>> split_spec = {"layers.1": SplitPoint.BEGINNING}
        >>> pipe = pipeline(
        ...     model,
        ...     mb_args=(example_input,),
        ...     split_spec=split_spec,
        ...     mode="fx",
        ...     atomic_units=["layers.0"],
        ... )
        >>> stage = pipe.build_stage(stage_index, device)
    """

    if _ORIGINAL_PIPELINE is None or _IR_MODULE is None:
        _init_pipelining_backend()

    if _ORIGINAL_PIPELINE is None or _IR_MODULE is None:
        raise RuntimeError("torch.distributed.pipelining.pipeline is unavailable")

    if split_spec is not None and split_policy is not None:
        raise ValueError(
            "Cannot specify both `split_spec` and `split_policy`. "
            "Please use only one of them."
        )

    if mode is None or mode == "export":
        if atomic_units is not None:
            raise ValueError("`atomic_units` is only supported when `mode='fx'`.")
        return _ORIGINAL_PIPELINE(
            module=module,
            mb_args=mb_args,
            mb_kwargs=mb_kwargs,
            split_spec=split_spec,
            split_policy=split_policy,
        )

    if mode != "fx":
        raise ValueError("`mode` must be one of None, 'export', or 'fx'.")

    return _pipeline_fx(
        _IR_MODULE,
        module=module,
        mb_args=mb_args,
        mb_kwargs=mb_kwargs,
        split_spec=split_spec,
        split_policy=split_policy,
        atomic_units=atomic_units,
    )


setattr(pipeline, _PATCHED_ATTR, True)


def _init_pipelining_backend() -> None:
    """Initialize native pipelining references used by pipeline."""

    global _ORIGINAL_PIPELINE, _IR_MODULE

    try:
        ir_module = importlib.import_module("torch.distributed.pipelining._IR")
    except ImportError:
        return

    if not hasattr(ir_module, "pipeline"):
        return

    _apply_quiet_graph_device_patch(ir_module)
    _ORIGINAL_PIPELINE = ir_module.pipeline
    _IR_MODULE = ir_module
