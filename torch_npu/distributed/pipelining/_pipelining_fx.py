# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.fx as fx
from torch import nn


_ATOMIC_UNIT_ATTR = "_torch_npu_pipelining_atomic_unit"
_WARNED_META_STRIDE_FALLBACK = False


@dataclass
class RealPyTreeInfo:
    """PyTree structure metadata for FX-traced pipeline graphs."""

    in_spec: Any
    out_spec: Any

    def _replace(self, **kwargs):
        """Mirror namedtuple-style `_replace` used by graph codegen metadata."""

        from dataclasses import replace

        return replace(self, **kwargs)


def _find_tp_group_ranks_from_model(model: nn.Module) -> list[int]:
    """Infer the tensor-parallel ranks that own DTensor state in a model."""

    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None

    if DTensor is None:
        return [0]

    for tensor in list(model.parameters(recurse=True)) + list(
        model.buffers(recurse=True)
    ):
        if isinstance(tensor, DTensor):
            mesh = tensor.device_mesh
            try:
                group = mesh.get_group()
            except TypeError:
                group = mesh.get_group(0)
            return dist.get_process_group_ranks(group)

    return [0]


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move nested tensors to a target device while preserving structure."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _tree_to_meta(value: Any) -> Any:
    """Convert a nested runtime value tree to meta tensors."""

    if isinstance(value, torch.Tensor):
        global _WARNED_META_STRIDE_FALLBACK
        try:
            return torch.empty_strided(
                size=tuple(value.size()),
                stride=tuple(value.stride()),
                dtype=value.dtype,
                device="meta",
                requires_grad=value.requires_grad,
            )
        except Exception:
            if not _WARNED_META_STRIDE_FALLBACK:
                warnings.warn(
                    "Failed to preserve tensor stride when converting runtime "
                    "values to meta tensors. Falling back to contiguous meta "
                    "tensor metadata.",
                    stacklevel=2,
                )
                _WARNED_META_STRIDE_FALLBACK = True
            return torch.empty(
                tuple(value.size()),
                dtype=value.dtype,
                device="meta",
                requires_grad=value.requires_grad,
            )
    if isinstance(value, tuple):
        return tuple(_tree_to_meta(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_meta(item) for key, item in value.items()}
    return value


def _module_has_meta_tensors(module: nn.Module) -> bool:
    """Return whether a module still contains meta parameters or buffers."""

    for param in module.parameters(recurse=True):
        if isinstance(param, torch.Tensor) and param.device.type == "meta":
            return True
    for buffer in module.buffers(recurse=True):
        if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
            return True
    return False


def _materialize_meta_module_(
    module: nn.Module,
    device: torch.device,
) -> nn.Module:
    """Materialize meta parameters and buffers for warmup execution."""

    if not _module_has_meta_tensors(module):
        return module

    module.to_empty(device=device, recurse=True)
    with torch.no_grad():
        for param in module.parameters(recurse=True):
            if param is None:
                continue
            if param.is_floating_point() or param.is_complex():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                param.zero_()
        for buffer in module.buffers(recurse=True):
            if buffer is not None:
                buffer.zero_()
    return module


def _pack_meta_tree(value: Any) -> Any:
    """Pack meta tensors into Python data for distributed object broadcast."""

    if isinstance(value, torch.Tensor):
        return {
            "__kind__": "tensor",
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "dtype": str(value.dtype).split(".", 1)[-1],
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_pack_meta_tree(v) for v in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_pack_meta_tree(v) for v in value]}
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": {key: _pack_meta_tree(item) for key, item in value.items()},
        }
    return {"__kind__": "scalar", "value": value}


def _dtype_from_str(dtype_name: str) -> torch.dtype:
    """Resolve a packed dtype string back to a torch dtype."""

    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.split(".", 1)[1]
    dtype = getattr(torch, dtype_name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise KeyError(f"Unsupported dtype string in packed meta: {dtype_name}")


def _unpack_meta_tree(value: Any) -> Any:
    """Unpack distributed metadata back into meta tensors."""

    kind = value["__kind__"]
    if kind == "tensor":
        return torch.empty_strided(
            size=tuple(value["shape"]),
            stride=tuple(value["stride"]),
            dtype=_dtype_from_str(value["dtype"]),
            device="meta",
            requires_grad=bool(value["requires_grad"]),
        )
    if kind == "tuple":
        return tuple(_unpack_meta_tree(item) for item in value["items"])
    if kind == "list":
        return [_unpack_meta_tree(item) for item in value["items"]]
    if kind == "dict":
        return {
            key: _unpack_meta_tree(item)
            for key, item in value["items"].items()
        }
    return value["value"]


def _pack_graph_meta(graph_module: fx.GraphModule) -> dict[str, Any]:
    """Pack all node `meta['val']` entries from an FX graph."""

    packed = {}
    for node in graph_module.graph.nodes:
        if "val" in node.meta:
            packed[node.name] = _pack_meta_tree(node.meta["val"])
    return packed


def _apply_packed_meta(
    graph_module: fx.GraphModule,
    packed: dict[str, Any],
) -> None:
    """Apply packed graph metadata to matching nodes by name."""

    name_to_node = {node.name: node for node in graph_module.graph.nodes}
    for name, meta_obj in packed.items():
        if name in name_to_node:
            name_to_node[name].meta["val"] = _unpack_meta_tree(meta_obj)


@dataclass
class WarmupMetaResult:
    """Result of eager warmup used to populate FX node metadata."""

    real_output: Any
    meta_output: Any


class _MetadataInterpreter(fx.Interpreter):
    """Run an FX graph once and attach meta tensor values to every node."""

    def run_node(self, node: fx.Node) -> Any:
        result = super().run_node(node)
        node.meta["val"] = _tree_to_meta(result)
        return result


def _warmup_device() -> torch.device:
    """Choose the per-rank device used for metadata warmup."""

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "npu") and torch.npu.is_available():
        device_index = local_rank % torch.npu.device_count()
        torch.npu.set_device(device_index)
        return torch.device(f"npu:{device_index}")
    if torch.cuda.is_available():
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def _seed_placeholder_meta_from_examples(
    graph_module: fx.GraphModule,
    example_args,
    example_kwargs=None,
) -> None:
    """Populate placeholder node metadata from real example inputs."""

    if example_kwargs is None:
        example_kwargs = {}

    signature = inspect.signature(graph_module.forward)
    bound = signature.bind_partial(*example_args, **example_kwargs)
    bound.apply_defaults()
    arg_map = dict(bound.arguments)

    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue

        key = node.target if isinstance(node.target, str) else node.name
        if key in arg_map:
            node.meta["val"] = _tree_to_meta(arg_map[key])
        elif node.name in arg_map:
            node.meta["val"] = _tree_to_meta(arg_map[node.name])


def populate_boundary_metadata(
    traced_gm: fx.GraphModule,
    example_args: tuple[Any, ...] | list[Any],
    example_kwargs: Optional[dict[str, Any]] = None,
    *,
    eager_root: Optional[nn.Module] = None,
    sync_per_node: bool = True,
    clear_existing_meta: bool = True,
    use_no_grad: bool = True,
) -> tuple[WarmupMetaResult, Optional[nn.Module]]:
    """Populate FX node `meta['val']` with eager warmup metadata."""

    if not isinstance(example_args, tuple):
        example_args = tuple(example_args)
    if example_kwargs is None:
        example_kwargs = {}
    if eager_root is None:
        eager_root = traced_gm

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    device = _warmup_device()
    tp_group_ranks = _find_tp_group_ranks_from_model(eager_root)
    should_run_warmup = 0 in tp_group_ranks

    real_out = None
    packed = None

    if clear_existing_meta:
        for node in traced_gm.graph.nodes:
            node.meta.pop("val", None)

    if should_run_warmup:
        if _module_has_meta_tensors(eager_root):
            eager_root = _materialize_meta_module_(eager_root, device)

        local_args = _move_to_device(example_args, device)
        local_kwargs = _move_to_device(example_kwargs, device)

        context = torch.no_grad() if use_no_grad else contextlib.nullcontext()
        _seed_placeholder_meta_from_examples(traced_gm, local_args, local_kwargs)
        with context:
            real_out = _MetadataInterpreter(traced_gm).run(
                *local_args,
                **local_kwargs,
            )

    if sync_per_node and dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        packed = _pack_graph_meta(traced_gm)

    if sync_per_node and dist.is_available() and dist.is_initialized():
        obj_list = [packed]
        dist.broadcast_object_list(obj_list, src=0)
        packed = obj_list[0]

    if packed is not None:
        _apply_packed_meta(traced_gm, packed)

    output_meta = None
    for node in traced_gm.graph.nodes:
        if node.op == "output":
            output_meta = node.meta.get("val", None)
            break

    return WarmupMetaResult(real_output=real_out, meta_output=output_meta), eager_root


def _contains_tensor(value: Any) -> bool:
    """Return whether a nested example input contains any Tensor values."""

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    return False


def _validate_atomic_units(
    module: nn.Module,
    atomic_units: Optional[list[str]],
) -> None:
    """Validate that every atomic unit names an existing submodule."""

    if not atomic_units:
        return

    module_names = dict(module.named_modules())
    missing = [name for name in atomic_units if name not in module_names]
    if missing:
        raise ValueError(
            "atomic_units contains module names that do not exist in the model: "
            + ", ".join(missing)
        )


def _mark_atomic_units(module: nn.Module, atomic_units: Optional[list[str]]) -> None:
    """Mark selected modules as FX leaves for the duration of tracing."""

    if not atomic_units:
        return

    module_names = dict(module.named_modules())
    for name in atomic_units:
        setattr(module_names[name], _ATOMIC_UNIT_ATTR, True)


def _clear_atomic_units(module: nn.Module, atomic_units: Optional[list[str]]) -> None:
    """Remove temporary atomic-unit markers after FX tracing."""

    if not atomic_units:
        return

    module_names = dict(module.named_modules())
    for name in atomic_units:
        submodule = module_names.get(name)
        if submodule is not None and hasattr(submodule, _ATOMIC_UNIT_ATTR):
            delattr(submodule, _ATOMIC_UNIT_ATTR)


def _concrete_args_from_examples(
    module: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build FX concrete_args from non-tensor example inputs."""

    example_kwargs = example_kwargs or {}
    signature = inspect.signature(module.forward)

    try:
        bound = signature.bind_partial(*example_args, **example_kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Failed to bind example inputs to forward signature: {exc}"
        ) from exc

    concrete_args: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name not in bound.arguments:
            if parameter.default is not inspect.Parameter.empty:
                concrete_args[name] = parameter.default
            continue

        value = bound.arguments[name]
        if not _contains_tensor(value):
            concrete_args[name] = value

    return concrete_args


class _FXPipeliningTracer(fx.Tracer):
    """FX tracer for pipeline splitting."""

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        """Decide whether FX should trace through a child module."""

        if getattr(module, _ATOMIC_UNIT_ATTR, False):
            return True
        if isinstance(module, nn.Sequential):
            return False
        if module.__module__.startswith(("torch.nn", "torch.ao.nn")):
            return True
        return super().is_leaf_module(module, module_qualified_name)


def symbolic_trace_for_pipelining(
    module: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: Optional[dict[str, Any]] = None,
    atomic_units: Optional[list[str]] = None,
) -> fx.GraphModule:
    """Trace a module with the pipelining FX policy."""

    _validate_atomic_units(module, atomic_units)
    concrete_args = _concrete_args_from_examples(module, example_args, example_kwargs)

    _mark_atomic_units(module, atomic_units)
    try:
        tracer = _FXPipeliningTracer()
        graph = tracer.trace(module, concrete_args=concrete_args)
        graph_module = fx.GraphModule(module, graph)
    finally:
        _clear_atomic_units(module, atomic_units)

    for attr_name in ("config", "device"):
        if hasattr(module, attr_name):
            setattr(graph_module, attr_name, getattr(module, attr_name))
    graph_module.class_for_deserialization = module.__class__
    return graph_module


def inject_pipeline_splits(
    traced: fx.GraphModule,
    split_spec: dict[Any, Any],
    pipe_split_op: Any,
) -> fx.GraphModule:
    """Insert pipeline split operators into an FX graph."""

    graph = traced.graph

    def matches(node_target: Any, split_target: str) -> bool:
        """Return whether an FX module target is covered by a split target."""

        node_target = str(node_target)
        return node_target == split_target or node_target.startswith(split_target + ".")

    for split_target, split_point in split_spec.items():
        split_target = str(split_target)
        matched_nodes = [
            node
            for node in graph.nodes
            if node.op == "call_module" and matches(node.target, split_target)
        ]
        if not matched_nodes:
            raise ValueError(
                f"split_spec target '{split_target}' did not match any "
                "FX call_module node"
            )

        if getattr(split_point, "name", None) == "BEGINNING":
            with graph.inserting_before(matched_nodes[0]):
                graph.call_function(pipe_split_op, args=())
        elif getattr(split_point, "name", None) == "END":
            with graph.inserting_after(matched_nodes[-1]):
                graph.call_function(pipe_split_op, args=())
        else:
            raise ValueError(
                f"Unsupported split point for target '{split_target}': "
                f"{split_point}"
            )

    graph.lint()
    traced.recompile()
    return traced
