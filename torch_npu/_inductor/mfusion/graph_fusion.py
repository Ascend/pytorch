import logging
import os
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch._inductor import config as inductor_config
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.simd import code_hash
from torch._inductor.codegen.wrapper import pexpr, PythonWrapperCodegen
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch_npu._inductor.dvm.fx_pass import (
    expand_dvm_mm_to_explicit_transpose_for_inductor,
)
from torch_npu._inductor.dvm.graph_build import DvmCodegenInterpreter
from torch_npu._inductor.mfusion import subgraph_registry
from torch_npu._inductor.mfusion.decomp import patch_decomp as patch_mfusion_decomp


# NOTE: Do not import fx_mlir_converter (torch_mlir) at module import time.
# test_public_bindings.TestPublicBindings.test_modules_can_be_imported requires
# torch_npu._inductor.mfusion to import without optional torch_mlir / mfusion native deps.

logger = logging.getLogger(__name__)

_mfusion_inductor_decomp_patches_applied = False


def _ensure_mfusion_inductor_decomp_patches() -> None:
    """Apply MFusion Inductor decomposition tweaks at most once per process.

    ``mfusion_graph_fusion`` is invoked for every ``torch.compile`` graph; MFusion
    decomposition patching must be idempotent.
    """
    global _mfusion_inductor_decomp_patches_applied
    if _mfusion_inductor_decomp_patches_applied:
        return
    try:
        patch_mfusion_decomp()
    except Exception:
        logger.exception("Failed to apply MFusion Inductor decomposition patches.")
        raise
    _mfusion_inductor_decomp_patches_applied = True


def _configure_mfusion_logger() -> None:
    """
    Optional logger level override.

    By default we do not override parent logging configuration.
    Set MFUSION_LOG_LEVEL (e.g. DEBUG/INFO/WARNING) to force a level.
    """
    level_name = os.getenv("MFUSION_LOG_LEVEL")
    if not level_name:
        return
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.warning("Invalid MFUSION_LOG_LEVEL=%s; ignored", level_name)


_configure_mfusion_logger()


def _mfusion_debug_print_enabled() -> bool:
    return os.getenv("MFUSION_GRAPH_FUSION_DEBUG_PRINT", "0") == "1"


def _mfusion_gm_num_outputs(gm: GraphModule) -> int:
    for n in gm.graph.nodes:
        if n.op == "output":
            return len(pytree.tree_leaves(n.args[0]))
    return 1


def _mfusion_has_matmul(gm: GraphModule) -> bool:
    aten = torch.ops.aten
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in (aten.mm.default, aten.bmm.default, aten.addmm.default):
            return True
    return False


def _mfusion_meta_summary(obj: Any, max_list_items: int = 8) -> Any:
    try:
        if isinstance(obj, torch.Tensor):
            # Works for FakeTensor as well (subclass of Tensor).
            return {
                "type": type(obj).__name__,
                "dtype": str(getattr(obj, "dtype", None)),
                "device": str(getattr(obj, "device", None)),
                "shape": list(getattr(obj, "shape", ())),
                "stride": list(obj.stride()) if hasattr(obj, "stride") else None,
                "requires_grad": bool(getattr(obj, "requires_grad", False)),
            }
        if isinstance(obj, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            return {"type": type(obj).__name__, "repr": str(obj)}
        if isinstance(obj, (int, float, bool, str)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            items = [
                _mfusion_meta_summary(x, max_list_items=max_list_items)
                for x in obj[:max_list_items]
            ]
            if len(obj) > max_list_items:
                items.append(f"...(+{len(obj) - max_list_items} more)")
            return items
        if isinstance(obj, dict):
            # Avoid overly verbose dumps.
            out = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_list_items:
                    out["..."] = f"+{len(obj) - max_list_items} more keys"
                    break
                out[str(k)] = _mfusion_meta_summary(v, max_list_items=max_list_items)
            return out
        return {"type": type(obj).__name__, "repr": repr(obj)[:300]}
    except Exception as exc:
        return {"type": type(obj).__name__, "error": repr(exc)}


def _build_fake_inputs_from_meta(meta_vals: list[Any]) -> list[Any] | None:
    if not all(v is not None for v in meta_vals):
        raise ValueError("invalid meta_vals: all values must be non-None")
    fake_mode: FakeTensorMode | None = None
    for v in meta_vals:
        if hasattr(v, "fake_mode"):
            fake_mode = v.fake_mode
            break
    if fake_mode is None:
        fake_mode = FakeTensorMode()
    args: list[Any] = []
    for v in meta_vals:
        if isinstance(v, torch.Tensor):
            args.append(fake_mode.from_tensor(v))
        else:
            args.append(v)
    return args


def _mfusion_safe_repr(obj: Any, max_len: int = 240) -> str:
    try:
        text = repr(obj)
    except Exception as exc:
        text = f"<repr_failed: {type(obj).__name__}: {exc!r}>"
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def _extract_subgraph_name_from_arg(arg: Any, gm: GraphModule | None) -> str | None:
    if isinstance(arg, str):
        return arg
    if (
        isinstance(arg, Node)
        and arg.op == "get_attr"
        and gm is not None
        and isinstance(arg.target, str)
    ):
        return getattr(gm, arg.target, None)
    return None


def _get_subgraph_name_from_node(node: Node) -> str:
    if not node.args:
        raise RuntimeError(
            "mfusion custom_op missing subgraph_name arg. "
            f"node_name={node.name}, node_op={node.op}, "
            f"node_target={_mfusion_safe_repr(getattr(node, 'target', None))}, args_len=0"
        )

    raw_subgraph_name_arg = node.args[-1]
    gm = node.graph.owning_module
    subgraph_name = _extract_subgraph_name_from_arg(raw_subgraph_name_arg, gm)
    if not isinstance(subgraph_name, str) or not subgraph_name.strip():
        details = [
            f"node_name={node.name}",
            f"node_op={node.op}",
            f"node_target={_mfusion_safe_repr(getattr(node, 'target', None))}",
            f"args_len={len(node.args)}",
            f"arg_type={type(raw_subgraph_name_arg).__name__}",
            f"arg_value={_mfusion_safe_repr(raw_subgraph_name_arg)}",
            f"resolved_type={type(subgraph_name).__name__}",
            f"resolved_value={_mfusion_safe_repr(subgraph_name)}",
        ]

        if isinstance(raw_subgraph_name_arg, str):
            if not raw_subgraph_name_arg.strip():
                details.append("arg_string_is_blank=True")
        elif isinstance(raw_subgraph_name_arg, Node):
            details.extend(
                [
                    f"arg_node_name={raw_subgraph_name_arg.name}",
                    f"arg_node_op={raw_subgraph_name_arg.op}",
                    f"arg_node_target={_mfusion_safe_repr(getattr(raw_subgraph_name_arg, 'target', None))}",
                ]
            )
            if raw_subgraph_name_arg.op == "get_attr":
                details.append(f"owning_module_present={gm is not None}")
                if gm is not None:
                    if isinstance(raw_subgraph_name_arg.target, str):
                        has_attr = hasattr(gm, raw_subgraph_name_arg.target)
                        details.append(f"gm_has_attr={has_attr}")
                        if has_attr:
                            attr_val = getattr(gm, raw_subgraph_name_arg.target)
                            details.append(
                                f"gm_attr_value={_mfusion_safe_repr(attr_val)}"
                            )
                    else:
                        details.append("arg_get_attr_target_is_not_str=True")
            else:
                details.append("arg_node_is_not_get_attr=True")

        raise RuntimeError(
            "mfusion custom_op subgraph_name resolution failed. "
            "Expected last arg to be a non-empty str, or a get_attr node that resolves to one. "
            + ", ".join(details)
        )
    return subgraph_name


def _get_mfusion_payload_from_node(node: Node) -> subgraph_registry.Payload:
    subgraph_name = _get_subgraph_name_from_node(node)
    return subgraph_registry.get(subgraph_name)


def _propagate_subgraph_meta_from_main(gm: GraphModule) -> None:
    from torch_npu._inductor.mfusion.fx_mlir_converter.fx_exporter import (
        fake_tensor_propagate_mfusion_subgraph,
    )

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        op_name = getattr(target, "_name", None)
        if not (isinstance(op_name, str) and op_name.startswith("mfusion::")):
            continue
        payload = _get_mfusion_payload_from_node(node)
        sub_gm = payload.fx_gm

        fake_inputs: list[Any] = []
        for arg in node.args[:-1]:
            if isinstance(arg, Node):
                fake_inputs.append(arg.meta.get("val"))
            else:
                fake_inputs.append(arg)
        # fake_inputs = _build_fake_inputs_from_meta(fake_inputs)
        if fake_inputs is None:
            raise RuntimeError("invalid fake_inputs: expected list, got None")
        fake_tensor_propagate_mfusion_subgraph(sub_gm, fake_inputs)


class _MFusionMetaSnapshot(NamedTuple):
    placeholders: list[dict[str, Any]]
    meta_by_name: dict[str, dict[str, Any]]


def _mfusion_print_banner(title: str) -> None:
    if not _mfusion_debug_print_enabled():
        return
    print("=" * 100)
    print(title)
    print("=" * 100)


def _mfusion_print_fx_graph(gm: GraphModule, title: str) -> None:
    if not _mfusion_debug_print_enabled():
        return
    _mfusion_print_banner(title)
    gm.print_readable()
    gm.graph.print_tabular()
    print("=" * 100)


def _mfusion_print_mlir(mlir_text: str, title: str) -> None:
    if not _mfusion_debug_print_enabled():
        return
    _mfusion_print_banner(title)
    print(mlir_text)
    print("=" * 100)


def _mfusion_log_saved_metas(
    gm: GraphModule,
    old_placeholders: list[dict[str, Any]],
    old_meta_by_name: dict[str, dict[str, Any]],
) -> None:
    logger.debug(
        "[mfusion_graph_fusion][debug] saved placeholder metas (count=%s):",
        len(old_placeholders),
    )
    for i, ph in enumerate(old_placeholders):
        logger.debug(
            "[mfusion_graph_fusion][debug]   saved placeholder idx=%s name=%s val=%s unbacked_bindings=%s",
            i,
            ph.get("name"),
            _mfusion_meta_summary(ph.get("val")),
            _mfusion_meta_summary(ph.get("unbacked_bindings")),
        )

    # Only log nodes we actually saved in old_meta_by_name (i.e., had "val").
    nodes_with_meta = [n for n in gm.graph.nodes if n.name in old_meta_by_name]
    logger.debug(
        "[mfusion_graph_fusion][debug] saved node metas (count=%s):",
        len(nodes_with_meta),
    )
    for n in nodes_with_meta:
        logger.debug(
            "[mfusion_graph_fusion][debug]   saved node name=%s op=%s target=%s meta_keys=%s val=%s unbacked_bindings=%s",
            n.name,
            n.op,
            str(getattr(n, "target", None)),
            sorted(n.meta.keys()),
            _mfusion_meta_summary(n.meta.get("val", None)),
            _mfusion_meta_summary(n.meta.get("unbacked_bindings", None)),
        )


def _mfusion_snapshot_metas(
    gm: GraphModule, *, debug_meta: bool
) -> _MFusionMetaSnapshot:
    """
    Snapshot `node.meta["val"]` (and optional `unbacked_bindings`) before FX<->MLIR
    roundtrip. Placeholder names may change after roundtrip, so we store placeholder
    metas by position (list order) and store other node metas by name as best-effort.
    """
    old_placeholders: list[dict[str, Any]] = []
    old_meta_by_name: dict[str, dict[str, Any]] = {}

    for n in gm.graph.nodes:
        if "val" in n.meta:
            meta = {
                "val": n.meta["val"],
                "unbacked_bindings": n.meta.get("unbacked_bindings", None),
            }
            old_meta_by_name[n.name] = meta
            if n.op == "placeholder":
                old_placeholders.append({"name": n.name, **meta})
        else:
            logger.debug(
                "[mfusion_graph_fusion][debug] node name=%s op=%s has no val meta",
                n.name,
                n.op,
            )

    if debug_meta:
        _mfusion_log_saved_metas(gm, old_placeholders, old_meta_by_name)

    return _MFusionMetaSnapshot(
        placeholders=old_placeholders, meta_by_name=old_meta_by_name
    )


def _mfusion_restore_metas_after_roundtrip(
    gm: GraphModule, *, snapshot: _MFusionMetaSnapshot, debug_meta: bool
) -> None:
    # Restore placeholder metas by position (names change: a/b/c -> arg0/arg1/arg2).
    new_placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, n in enumerate(new_placeholders):
        if i >= len(snapshot.placeholders):
            break
        if n.meta.get("val") is not None:
            logger.debug(
                "[mfusion_graph_fusion][debug] placeholder idx=%s new_name=%s already has val=%s",
                i,
                n.name,
                _mfusion_meta_summary(n.meta.get("val")),
            )
            continue
        meta = snapshot.placeholders[i]
        n.meta["val"] = meta["val"]
        if meta.get("unbacked_bindings") is not None:
            n.meta["unbacked_bindings"] = meta["unbacked_bindings"]
        if debug_meta:
            logger.debug(
                "[mfusion_graph_fusion][debug] restored placeholder idx=%s old_name=%s -> new_name=%s val=%s unbacked_bindings=%s",
                i,
                meta.get("name"),
                n.name,
                _mfusion_meta_summary(meta.get("val")),
                _mfusion_meta_summary(meta.get("unbacked_bindings")),
            )

    # Restore any remaining metas by name (get_attr etc.), best-effort.
    for n in gm.graph.nodes:
        if "val" in n.meta:
            logger.debug(
                "[mfusion_graph_fusion][debug] node name=%s op=%s already has val=%s",
                n.name,
                n.op,
                _mfusion_meta_summary(n.meta.get("val")),
            )
            continue
        meta = snapshot.meta_by_name.get(n.name)
        if not meta:
            logger.debug(
                "[mfusion_graph_fusion][debug] node name=%s op=%s has no meta to restore",
                n.name,
                n.op,
            )
            continue
        n.meta["val"] = meta["val"]
        if meta.get("unbacked_bindings") is not None:
            n.meta["unbacked_bindings"] = meta["unbacked_bindings"]
        if debug_meta:
            logger.debug(
                "[mfusion_graph_fusion][debug] restored non-placeholder name=%s op=%s target=%s val=%s unbacked_bindings=%s",
                n.name,
                n.op,
                str(getattr(n, "target", None)),
                _mfusion_meta_summary(meta.get("val")),
                _mfusion_meta_summary(meta.get("unbacked_bindings")),
            )


def _mfusion_refresh_faketensor_metas(gm: GraphModule) -> None:
    # Recompute faketensor meta for call_function nodes in the new graph.
    # This must run under Inductor fake mode (V.fake_mode is set by Inductor).
    try:
        from torch._inductor.fx_utils import FakeTensorUpdater

        updater = FakeTensorUpdater(gm.graph)
        # Force processing all nodes in the reconstructed graph.
        updater.processed_hashes.clear()
        updater.incremental_update()
    except Exception:
        logger.debug(
            "mfusion graph_fusion failed to refresh node.meta['val']",
            exc_info=True,
        )


def _is_mfusion_op(fallback_kernel) -> bool:
    logger.debug("is_mfusion_op: %s", fallback_kernel.op_overload._name)
    return fallback_kernel.op_overload._name.startswith("mfusion::")


def _ensure_graph_module(orig_graph) -> GraphModule | None:
    if isinstance(orig_graph, GraphModule):
        return orig_graph
    if isinstance(orig_graph, Graph):
        return GraphModule(torch.nn.Module(), orig_graph)
    return None


def _get_mfusion_orig_graph(fallback_kernel) -> GraphModule | None:
    origin_node = fallback_kernel.get_origin_node()
    if origin_node is None:
        raise RuntimeError("mfusion custom_op has no origin node")
    payload = _get_mfusion_payload_from_node(origin_node)
    return _ensure_graph_module(payload.fx_gm)


def _get_mfusion_mlir(fallback_kernel) -> str | None:
    origin_node = fallback_kernel.get_origin_node()
    if origin_node is None:
        return None
    payload = _get_mfusion_payload_from_node(origin_node)
    return payload.mlir


def _get_mfusion_payload_from_fallback(fallback_kernel) -> subgraph_registry.Payload:
    origin_node = fallback_kernel.get_origin_node()
    if origin_node is None:
        raise RuntimeError("mfusion custom_op has no origin node")
    return _get_mfusion_payload_from_node(origin_node)


def _layout_get_size(layout):
    if hasattr(layout, "size"):
        return layout.size
    if hasattr(layout, "get_size"):
        return layout.get_size()
    return None


def _layout_get_stride(layout):
    if hasattr(layout, "stride"):
        return layout.stride
    if hasattr(layout, "get_stride"):
        return layout.get_stride()
    return None


def _layout_get_dtype(layout):
    if hasattr(layout, "dtype"):
        return layout.dtype
    if hasattr(layout, "get_dtype"):
        return layout.get_dtype()
    return None


def _is_none_layout(layout) -> bool:
    return layout is None or layout.__class__.__name__ == "NoneLayout"


def _normalize_output_layouts(layout) -> list:
    if _is_none_layout(layout):
        return []
    if isinstance(layout, (list, tuple)):
        return list(layout)
    layouts = getattr(layout, "layouts", None)
    if layouts is not None:
        return list(layouts)
    return [layout]


def _format_sympy_list(items) -> str:
    return "[" + ", ".join(pexpr(item) for item in items) + "]"


def _graph_output_has_none(gm: GraphModule) -> bool:
    """True if the graph output tuple contains any None (e.g. backward grad_outputs)."""
    for node in gm.graph.nodes:
        if node.op == "output":
            args = node.args[0]
            if not isinstance(args, (list, tuple)):
                args = [args]
            return any(a is None for a in args)
    return False


def mfusion_graph_fusion(graph: Graph) -> None:
    # Ensure Inductor decomposition table excludes rms_norm (decomps_to_exclude_npu) so
    # that graphs entering MFusion still contain aten.rms_norm when model uses F.rms_norm.
    _ensure_mfusion_inductor_decomp_patches()

    from torch_mlir import ir as mlir_ir
    from torch_mlir.dialects import torch as torch_d

    from torch_npu._inductor.mfusion.fx_mlir_converter.fx_exporter import (
        export_mlir_module_to_fx,
        fake_tensor_propagate_mfusion_subgraph,
    )
    from torch_npu._inductor.mfusion.fx_mlir_converter.fx_importer import (
        import_mlir_module_from_fx,
    )

    gm: GraphModule = graph.owning_module
    # NOTE [Inductor meta['val'] invariants]
    # Inductor expects FX nodes to carry FakeTensor metadata in `node.meta["val"]`
    # (placeholders/get_attr/call_function, etc.). Post-grad passes run with a
    # FakeTensorUpdater that can incrementally refresh this metadata *as long as*
    # we mutate the existing graph. The FX<->MLIR roundtrip below reconstructs a
    # new graph and would drop all `meta["val"]`, causing Inductor to later fail
    # with KeyError: 'val'. We snapshot input/const metas and then re-propagate.
    # Snapshot metas from the *current* graph (pre-roundtrip).
    # NOTE: placeholder names/targets may change after roundtrip, so we must
    # restore placeholder metas by position, not by name.

    # Record original output structure when output contains None (restore after roundtrip so backward returns correct count)
    original_output_args = None
    valid_positions = None
    output_node_pre = next((n for n in gm.graph.nodes if n.op == "output"), None)
    if output_node_pre:
        args_pre = output_node_pre.args[0]
        args_pre = list(args_pre) if isinstance(args_pre, (list, tuple)) else [args_pre]
        if any(a is None for a in args_pre):
            original_output_args = args_pre
            valid_positions = [i for i, a in enumerate(args_pre) if a is not None]

    # 0.Extract inputs for metadata restoration
    example_inputs = [
        node.meta.get("val") for node in graph.nodes if node.op == "placeholder"
    ]
    _mfusion_print_fx_graph(gm, "# Before fx -> torch-mlir, FX Graph:")

    # 1.fx -> torch-mlir
    mlir_module = import_mlir_module_from_fx(gm)
    mlir_str = str(mlir_module)
    _mfusion_print_mlir(mlir_str, "# After fx -> torch-mlir, MLIR Module:")

    # 2.graph fusion (no longer skip when output has None; SSA dominance fixed in convert-mfuse-to-torch)
    skip_mfusion = os.getenv("MFUSION_SKIP_FUSION", "0") == "1"
    fuse_and_optimize = None
    if not skip_mfusion:
        try:
            from mfusion.torch.inductor import fuse_and_optimize as _fuse_and_optimize

            fuse_and_optimize = _fuse_and_optimize
        except ModuleNotFoundError:
            logger.warning(
                "mfusion Python package not found (install AKG mfusion or set PYTHONPATH); "
                "skipping fuse_and_optimize and using pre-fusion MLIR."
            )

    if skip_mfusion or fuse_and_optimize is None:
        if skip_mfusion:
            logger.warning("SKIP mfusion (MFUSION_SKIP_FUSION=1)...")
        fused_mlir_str = mlir_str
    else:
        fused_mlir_str = fuse_and_optimize(mlir_str)
        if fused_mlir_str is None:
            raise RuntimeError("mfusion fuse output must not be None")
        if not isinstance(fused_mlir_str, str):
            raise TypeError(
                f"mfusion fuse output must be a str, got {type(fused_mlir_str).__name__}"
            )

    # 3.torch-mlir -> fx
    with mlir_ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        torch_d.register_dialect(ctx)
        out_module = mlir_ir.Module.parse(fused_mlir_str)
    out_gm = export_mlir_module_to_fx(out_module)

    if out_gm is None:
        raise RuntimeError("mfusion export_mlir_module_to_fx returned None")
    if out_gm is gm:
        raise RuntimeError(
            "mfusion export_mlir_module_to_fx must return a new GraphModule, "
            "got the same instance as the input graph"
        )
    gm.graph = out_gm.graph
    # Restore original output count by padding None at positions that were None (AOT backward expects fixed count)
    if original_output_args is not None and valid_positions is not None:
        out_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
        if out_node:
            current = out_node.args[0]
            current = list(current) if isinstance(current, (list, tuple)) else [current]
            if len(current) == len(valid_positions) and len(original_output_args) > len(
                valid_positions
            ):
                new_output_list = [None] * len(original_output_args)
                for j, i in enumerate(valid_positions):
                    new_output_list[i] = current[j]
                out_node.args = (tuple(new_output_list),)
    _mfusion_print_fx_graph(gm, "# After torch-mlir -> fx, FX Graph:")

    # 4.restore meta info
    # v1:
    # _mfusion_restore_metas_after_roundtrip(gm, snapshot=snapshot, debug_meta=debug_meta)
    # _mfusion_refresh_faketensor_metas(gm)
    # v2:
    # Restore metadata using FakeTensorProp. MLIR roundtrip can emit aten.mm + dvm_trans_* (fused
    # transpose semantics with storage shapes); use fake_tensor_propagate_mfusion_subgraph so mm is
    # patched during propagation (see fx_exporter.patch_dvm_mm_targets_for_fake_eval).
    if example_inputs and all(x is not None for x in example_inputs):
        mode = None
        for x in example_inputs:
            if hasattr(x, "fake_mode"):
                mode = x.fake_mode
                break

        if mode is None:
            try:
                from torch._subclasses.fake_tensor import get_fake_mode

                mode = get_fake_mode()
            except ImportError:
                pass

        fake_tensor_propagate_mfusion_subgraph(gm, example_inputs, fake_mode=mode)
        # dvm_trans_* keeps storage shapes for fused aclnn semantics; Inductor mm lowering needs
        # explicit transpose + mm so mm_args inner dims match. Re-propagate FakeTensor after expand.
        if expand_dvm_mm_to_explicit_transpose_for_inductor(gm):
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp

            if example_inputs and all(x is not None for x in example_inputs):
                if mode is not None:
                    with mode:
                        FakeTensorProp(gm, mode=mode).propagate(*example_inputs)
                else:
                    FakeTensorProp(gm).propagate(*example_inputs)
    _propagate_subgraph_meta_from_main(gm)
    gm.recompile()
    _mfusion_print_fx_graph(gm, "# After restore meta, FX Graph:")


def _emit_mfusion_dvm_codegen(
    self, fallback_kernel, args, sub_gm: GraphModule, payload
) -> None:
    # payload is already passed in from caller
    sub_gm._mfusion = True
    ktype = "mix" if _mfusion_has_matmul(sub_gm) else "vector"
    cg = DvmCodegenInterpreter(sub_gm, ktype=ktype, is_dynamic=payload.is_dynamic)
    cg.run()
    kernel_name = f"mfusion_dvm_{self.next_kernel_suffix()}"
    cg.append_mfusion_kernel_profiling_metadata(
        kernel_name, _mfusion_gm_num_outputs(sub_gm)
    )
    code = cg.code.getvalue().replace(cg.KERNEL_NAME_PLACEHOLDER, kernel_name)
    self.header.splice(code)

    logger.debug("dvm codegen: %s", code)

    buf_name = fallback_kernel.get_name()
    args_list = list(args[:-1])
    if len(args_list) != len(cg.cont_flag_input):
        raise RuntimeError(
            "mfusion dvm codegen arg mismatch: "
            f"{len(args_list)} vs {len(cg.cont_flag_input)}"
        )
    for i, no_trans in enumerate(cg.cont_flag_input):
        if not no_trans:
            args_list[i] += ".contiguous()"
    for i, trans in enumerate(cg.need_trans_input):
        if trans:
            args_list[i] += ".mT"
    self.writeline(f"{buf_name} = {kernel_name}({', '.join(args_list)})")
    self.add_import_once("from torch_npu._inductor import dvm")


def _define_mfusion_akg_kernel(self, mlir: str, num_outputs: int, is_dynamic: bool):
    # Ref: ascend_npu_ir/.../npu/codegen/akg.py AkgCompiler
    # Ref: ascend_npu_ir/.../codecache.py akg_auto_fallback (uses AkgCompiler.compile)
    src_key = code_hash(mlir.strip())
    kernel_name = self.src_to_kernel.get(src_key, None)
    if kernel_name is not None:
        # Kernel already generated, return the name directly
        return kernel_name

    kernel_name = f"mfusion_akg_{src_key}"
    self.src_to_kernel[src_key] = kernel_name

    # Ref: ascend_npu_ir/.../npu/codegen/akg.py AkgScheduling.define_kernel
    # kernel_meta uses current device + dynamic flag
    current_device = V.graph.get_current_device_or_throw()
    kernel_meta = {
        "device_str": current_device.type,
        "device_index": current_device.index,
        "kernel_name": kernel_name,
        "num_outputs": num_outputs,
        "dynamic": is_dynamic,
    }
    # Ref: ascend_npu_ir/.../npu/codegen/wrapper.py write_header
    # ensure AkgCompiler import once (non-mlir-backend wrapper)
    self.add_import_once(
        "from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.akg "
        "import AkgCompiler"
    )
    # Ref: ascend_npu_ir/.../codecache.py akg_auto_fallback
    # compile MLIR via AkgCompiler.compile
    compile_wrapper = IndentedBuffer()
    compile_wrapper.writeline(f"{kernel_name} = AkgCompiler(kernel_meta={kernel_meta})")
    compile_wrapper.writeline(f"{kernel_name}.compile('''")
    compile_wrapper.splice(mlir, strip=True)
    compile_wrapper.writeline("''')")
    self.header.splice(compile_wrapper.getvalue())
    return kernel_name


def _emit_mfusion_akg_codegen(
    self, fallback_kernel, args, sub_gm: GraphModule, payload
) -> None:
    # payload is already passed in from caller
    mlir = payload.mlir
    if not mlir:
        raise RuntimeError("mfusion custom_op has no mlir string input")

    # Ref: torch._inductor.codegen.wrapper generate_extern_kernel_alloc
    # outputs are inferred from fallback kernel layout
    output_layouts = _normalize_output_layouts(fallback_kernel.layout)
    if not output_layouts:
        raise RuntimeError("mfusion custom_op has no tensor outputs")

    kernel_name = _define_mfusion_akg_kernel(
        self, mlir, num_outputs=len(output_layouts), is_dynamic=payload.is_dynamic
    )

    suffix = self.next_kernel_suffix()
    # Ref: ascend_npu_ir/.../npu/codegen/wrapper.py generate_kernel_call
    # Match MLIR backend kernel calling convention:
    #   kernel.run(*inputs, *outputs, stream=streamX)
    #
    # In non-mlir-backend, wrapper may be `PythonWrapperCodegen` / `NPUWrapperCodeGen`,
    # which may not have `write_get_raw_stream`. So we:
    #   - Ensure `get_raw_stream` import exists via `write_triton_header_once()`
    #   - Emit a per-call stream var via `get_raw_stream(device_index)`
    #   - Set device context once in header to match MLIR wrapper behavior.
    try:
        device_index = V.graph.scheduler.current_device.index  # type: ignore[union-attr]
    except Exception:
        device_index = V.graph.get_current_device_or_throw().index
    device_index = device_index if device_index is not None else 0

    # Ref: ascend_npu_ir/.../npu/codegen/wrapper.py write_triton_header_once
    # Ref: ascend_npu_ir/.../npu/codegen/wrapper.py write_get_raw_stream
    # Ensure `get_raw_stream` is importable in generated wrapper code.
    # NPU MLIR wrapper does this in `write_triton_header_once()` / `write_get_raw_stream()`,
    # but mfusion can run under the default (non-mlir-backend) wrapper as well.
    self.write_triton_header_once()
    self.add_import_once(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))
    self.add_import_once("import torch")
    self.add_import_once("import torch_npu")

    # Ref: ascend_npu_ir/.../npu/codegen/wrapper.py write_get_raw_stream
    # mirror set_device in header (dedup per device)
    if not hasattr(self, "_mfusion_set_device_done"):
        self._mfusion_set_device_done = set()
    if device_index not in self._mfusion_set_device_done:
        self.header.writeline(f"torch_npu.npu.set_device({device_index})")
        self._mfusion_set_device_done.add(device_index)

    stream_var = f"_mfusion_stream_{suffix}"
    self.writeline(f"{stream_var} = get_raw_stream({device_index})")
    device_expr = f"torch.device('npu', {device_index})"

    # Ref: torch._inductor.codegen.wrapper / memory_planning output allocation
    # allocate outputs explicitly for fallback path
    out_names: list[str] = []
    for i, layout in enumerate(output_layouts):
        size = _layout_get_size(layout)
        stride = _layout_get_stride(layout)
        dtype = _layout_get_dtype(layout)
        if size is None or stride is None or dtype is None:
            raise RuntimeError(
                "mfusion custom_op output layout missing size/stride/dtype"
            )
        size_expr = _format_sympy_list(size)
        stride_expr = _format_sympy_list(stride)
        out_name = f"mfusion_out_{suffix}_{i}"
        out_names.append(out_name)
        self.writeline(
            f"{out_name} = torch.empty_strided({size_expr}, {stride_expr}, "
            f"device={device_expr}, dtype={dtype!r})"
        )

    args_list = list(args[:-1])
    call_args = ", ".join([*args_list, *out_names])
    self.writeline(f"{kernel_name}.run({call_args}, stream={stream_var})")

    buf_name = fallback_kernel.get_name()
    if len(out_names) == 1:
        self.writeline(f"{buf_name} = {out_names[0]}")
    else:
        self.writeline(f"{buf_name} = ({', '.join(out_names)})")


def _mfusion_generate_fallback_kernel(self, fallback_kernel, args) -> None:
    logger.debug("generate fallback kernel: %s", fallback_kernel)
    if not _is_mfusion_op(fallback_kernel):
        return MFusionPatch._orig_generate_fallback_kernel(self, fallback_kernel, args)

    logger.debug("generate dvm/akg kernel: %s", fallback_kernel)

    # Get subgraph_name directly from args[-1], bypassing fallback_kernel.get_origin_node()
    if not args or len(args) == 0:
        raise RuntimeError(
            "mfusion fallback_kernel args is empty. "
            f"fallback_kernel={fallback_kernel.python_kernel_name}. "
            f"This indicates a bug in mfusion custom_op creation."
        )

    for i, arg in enumerate(args):
        logger.debug(
            "arg %d: %s, type: %s, repr: %s", i, arg, type(arg), _mfusion_safe_repr(arg)
        )
    subgraph_name = args[-1]

    # Handle case where subgraph_name may have been repr'd by inductor codegen
    # e.g., "'main_fused_0_'" or '"main_fused_0_"' instead of "main_fused_0_"
    if isinstance(subgraph_name, str):
        # Strip matching outer quotes (single or double)
        if (subgraph_name.startswith("'") and subgraph_name.endswith("'")) or (
            subgraph_name.startswith('"') and subgraph_name.endswith('"')
        ):
            original = subgraph_name
            subgraph_name = subgraph_name[1:-1]
            logger.debug(
                "Stripped outer quotes from subgraph_name: %r -> %r",
                original,
                subgraph_name,
            )

    if not isinstance(subgraph_name, str):
        raise RuntimeError(
            f"mfusion subgraph_name must be a string, but got {type(subgraph_name).__name__}: "
            f"{repr(subgraph_name)}. "
            f"args length={len(args)}, args={args}. "
            f"fallback_kernel={fallback_kernel.python_kernel_name}. "
            f"The last argument should be subgraph_name (str). "
            f"Please check mfusion custom_op definition."
        )

    if not subgraph_name.strip():
        raise RuntimeError(
            f"mfusion subgraph_name is empty or whitespace-only. "
            f"subgraph_name={repr(subgraph_name)}. "
            f"args={args}. "
            f"fallback_kernel={fallback_kernel.python_kernel_name}."
        )

    try:
        payload = subgraph_registry.get(subgraph_name)
    except Exception as exc:
        subgraph_registry.print_registry()
        raise RuntimeError(
            f"Failed to get mfusion payload for subgraph_name='{subgraph_name}'. "
            f"Error: {exc}. "
            f"fallback_kernel={fallback_kernel.python_kernel_name}. "
            f"Please verify that '{subgraph_name}' is registered in subgraph_registry."
        ) from exc

    try:
        sub_gm = _ensure_graph_module(payload.fx_gm)
        if sub_gm is None:
            raise RuntimeError(
                f"mfusion custom_op has no fx_gm. "
                f"subgraph_name='{subgraph_name}', "
                f"fallback_kernel={fallback_kernel.python_kernel_name}."
            )

        use_akg = os.getenv("TORCHINDUCTOR_USE_AKG", "0") == "1"
        if use_akg:
            _emit_mfusion_akg_codegen(self, fallback_kernel, args, sub_gm, payload)
        else:
            _emit_mfusion_dvm_codegen(self, fallback_kernel, args, sub_gm, payload)
    finally:
        try:
            subgraph_registry.pop(subgraph_name)
        except KeyError:
            logger.warning(
                "mfusion registry cleanup skipped: subgraph_name '%s' already cleared",
                subgraph_name,
            )


class MFusionPatch:
    _enabled = False
    _orig_generate_fallback_kernel = None
    _orig_post_grad_custom_post_pass = None

    @staticmethod
    def enable() -> None:
        if not MFusionPatch._enabled:
            _ensure_mfusion_inductor_decomp_patches()
            MFusionPatch._orig_generate_fallback_kernel = (
                PythonWrapperCodegen.generate_fallback_kernel
            )
            MFusionPatch._orig_post_grad_custom_post_pass = (
                inductor_config.post_grad_custom_post_pass
            )
            PythonWrapperCodegen.generate_fallback_kernel = (
                _mfusion_generate_fallback_kernel
            )
            inductor_config.post_grad_custom_post_pass = mfusion_graph_fusion
            MFusionPatch._enabled = True

    @staticmethod
    def disable() -> None:
        if not MFusionPatch._enabled:
            return
        PythonWrapperCodegen.generate_fallback_kernel = (
            MFusionPatch._orig_generate_fallback_kernel
        )
        inductor_config.post_grad_custom_post_pass = (
            MFusionPatch._orig_post_grad_custom_post_pass
        )
        MFusionPatch._enabled = False

    def __enter__(self) -> "MFusionPatch":
        MFusionPatch.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        MFusionPatch.disable()
        return False
