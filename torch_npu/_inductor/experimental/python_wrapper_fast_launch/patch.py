from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

from torch_npu._inductor import config as npu_config


_PATCHED = False
_WRAPPER_PATCHED_ATTR = "_npu_fast_launch_patched"
_TRITON_PATCHED_ATTR = "_npu_fast_launch_patched"


def _resolve_wrapper_module():
    return import_module("torch_npu._inductor.codegen.wrapper")


def _resolve_triton_heuristics_module():
    return import_module("torch_npu._inductor.runtime.triton_heuristics")


def _original_method(owner: Any, stash_name: str, method_name: str) -> Any:
    return getattr(owner, stash_name, getattr(owner, method_name))


def _build_wrapper_patch_operations(wrapper_module: Any) -> list[tuple[Any, str, Any]]:
    wrapper_cls = wrapper_module.NPUPythonWrapperCodeGen
    if getattr(wrapper_cls, _WRAPPER_PATCHED_ATTR, False):
        return []

    original_init = _original_method(
        wrapper_cls,
        "_npu_fast_launch_original_init",
        "__init__",
    )
    original_write_triton_header_once = _original_method(
        wrapper_cls,
        "_npu_fast_launch_original_write_triton_header_once",
        "write_triton_header_once",
    )
    original_generate_kernel_call = _original_method(
        wrapper_cls,
        "_npu_fast_launch_original_generate_kernel_call",
        "generate_kernel_call",
    )
    emitter_cls = import_module(
        "torch_npu._inductor.experimental.python_wrapper_fast_launch.wrapper_codegen"
    ).FastLaunchWrapperEmitter

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._npu_fast_launch = emitter_cls(self)

    @wraps(original_write_triton_header_once)
    def patched_write_triton_header_once(self):
        result = original_write_triton_header_once(self)
        if getattr(self, "_npu_fast_launch_header_imported", False):
            return result
        self._npu_fast_launch_header_imported = True
        if (
            not wrapper_module.V.graph.cpp_wrapper
            and not wrapper_module.config.triton.autotune_at_compile_time
        ):
            self.imports.writeline(
                "from torch_npu._inductor.experimental."
                "python_wrapper_fast_launch.bind import "
                "bind_python_wrapper_kernel_fast"
            )
        return result

    @wraps(original_generate_kernel_call)
    def patched_generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        origin_node=None,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        original_fxnode_name=None,
    ):
        use_fast_launch = bool(
            triton
            and not wrapper_module.V.graph.cpp_wrapper
            and not wrapper_module._is_codegen_graph_partition_subgraph(self)
            and not wrapper_module.config.triton.autotune_at_compile_time
        )
        if not use_fast_launch:
            return original_generate_kernel_call(
                self,
                kernel_name,
                call_args,
                origin_node,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
                original_fxnode_name=original_fxnode_name,
            )

        use_multi_stream = wrapper_module.is_multi_stream()
        graph = wrapper_module.V.graph
        device = device or graph.get_current_device_or_throw()
        call_args_str = ", ".join(self.prepare_triton_kernel_call(call_args))
        if (
            use_multi_stream
            and origin_node is not None
            and hasattr(origin_node, "multi_stream_name")
        ):
            stream_name = origin_node.multi_stream_name
            multi_stream_indent = " " * origin_node.multi_stream_intent
        else:
            multi_stream_indent = ""
            stream_name = wrapper_module.PythonWrapperCodegen.write_get_raw_stream(
                self,
                device.index,
                graph,
            )

        self.write_triton_header_once()

        debug_printer_manager = graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            arg_types,
            None,
        )
        with debug_printer_manager:
            self._npu_fast_launch.emit_triton_call(
                kernel_name=kernel_name,
                call_args=call_args,
                triton_meta=triton_meta,
                call_args_str=call_args_str,
                stream_name=stream_name,
                indent=multi_stream_indent,
            )
        return None

    return [
        (
            wrapper_cls,
            "_npu_fast_launch_original_init",
            original_init,
        ),
        (
            wrapper_cls,
            "_npu_fast_launch_original_write_triton_header_once",
            original_write_triton_header_once,
        ),
        (
            wrapper_cls,
            "_npu_fast_launch_original_generate_kernel_call",
            original_generate_kernel_call,
        ),
        (wrapper_cls, "__init__", patched_init),
        (
            wrapper_cls,
            "write_triton_header_once",
            patched_write_triton_header_once,
        ),
        (
            wrapper_cls,
            "generate_kernel_call",
            patched_generate_kernel_call,
        ),
        (wrapper_cls, _WRAPPER_PATCHED_ATTR, True),
    ]


def _build_none_args(triton_module: Any, compile_meta: dict[str, Any], fn: Any):
    ordered_set = triton_module.OrderedSet
    known_constants = ordered_set(
        arg for index, arg in enumerate(fn.arg_names) if index in fn.constexprs
    )
    none_args = ordered_set(
        name
        for name, value in compile_meta["constants"].items()
        if value is None and name not in known_constants
    )
    return none_args.difference(ordered_set(compile_meta["signature"].keys()))


def _launcher_args_from_compile_result(
    triton_module: Any,
    compile_result: Any,
    fn: Any,
    none_args: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cfg = compile_result.config
    compile_meta = compile_result.compile_meta
    inductor_meta = compile_result.inductor_meta
    if triton_module.triton_version_uses_attrs_dict():
        runtime_block_names = tuple(inductor_meta.get("runtime_block_arg_names", ()))
        runtime_block_set = set(runtime_block_names)
        cfg_dict = triton_module.config_to_dict(cfg)
        replacements = {}
        for index in fn.constexprs:
            arg = fn.arg_names[index]
            if arg in runtime_block_set:
                continue
            if arg in compile_meta["constants"]:
                replacements[arg] = str(compile_meta["constants"][arg])
            elif arg in cfg_dict:
                replacements[arg] = str(cfg_dict[arg])

        def_args = [
            arg
            for index, arg in enumerate(fn.arg_names)
            if index not in fn.constexprs or arg in runtime_block_set
        ]
        call_args = [
            replacements.get(arg, arg)
            for arg in fn.arg_names
            if arg in compile_meta["signature"]
        ]
    else:
        call_args = [
            arg
            for index, arg in enumerate(fn.arg_names)
            if index not in fn.constexprs and arg not in none_args
        ]
        cfg_dict = triton_module.config_to_dict(cfg)
        def_args = triton_module.filter_launcher_def_args(
            fn.arg_names,
            cfg_dict,
            none_args,
            tuple(inductor_meta.get("runtime_block_arg_names", ())),
        )

    if "extra_launcher_args" in inductor_meta:
        def_args = [*def_args, *inductor_meta["extra_launcher_args"]]

    return tuple(str(arg) for arg in def_args), tuple(str(arg) for arg in call_args)


def _build_grid_expr(
    triton_module: Any,
    compile_result: Any,
    fn: Any,
) -> Any:
    cfg = compile_result.config
    inductor_meta = compile_result.inductor_meta
    numels = [arg for arg in fn.arg_names if "_numel" in arg]
    runtime_block_names = tuple(inductor_meta.get("runtime_block_arg_names", ()))
    linear_mode = inductor_meta.get(
        "inductor_ascend_linear_mode",
        "no_linear",
    )
    if linear_mode == "no_linear" and not runtime_block_names:
        return triton_module.GridExpr.from_meta(inductor_meta, cfg)
    return triton_module.GridExprNpu.from_meta_and_set_numel(
        inductor_meta,
        cfg,
        numels,
    )


def _install_grid_resolver(
    launcher: Any,
    grid: Any,
    def_args: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    scope = getattr(launcher, "__globals__", None)
    if not isinstance(scope, dict):
        raise RuntimeError("launcher globals are unavailable")
    lines = [
        f"def _npu_fast_launch_get_grid({', '.join(def_args)}):",
        *[f"    {line}" for line in grid.prefix],
        f"    grid_0 = {grid.x_grid}",
        f"    grid_1 = {grid.y_grid}",
        f"    grid_2 = {grid.z_grid}",
        "    return grid_0, grid_1, grid_2",
    ]
    exec("\n".join(lines), scope)
    return scope["_npu_fast_launch_get_grid"], scope


def _build_triton_patch_operations(
    triton_module: Any,
) -> list[tuple[Any, str, Any]]:
    compile_cls = triton_module.TritonCompileResultNpu
    if getattr(compile_cls, _TRITON_PATCHED_ATTR, False):
        return []

    original_make_launcher = _original_method(
        compile_cls,
        "_npu_fast_launch_original_make_launcher",
        "make_launcher",
    )
    attach_metadata = import_module(
        "torch_npu._inductor.experimental.python_wrapper_fast_launch.launcher"
    ).attach_python_wrapper_launcher_metadata

    @wraps(original_make_launcher)
    def patched_make_launcher(self):
        launcher = original_make_launcher(self)
        if self.inductor_meta.get("group_enabled", False):
            return launcher
        try:
            compile_meta = self.compile_meta
            binary = self.kernel
            fn = binary.src.fn
            none_args = _build_none_args(
                triton_module,
                compile_meta,
                fn,
            )
            def_args, _ = _launcher_args_from_compile_result(
                triton_module,
                self,
                fn,
                none_args,
            )
            grid = _build_grid_expr(triton_module, self, fn)
            get_grid, scope = _install_grid_resolver(
                launcher,
                grid,
                def_args,
            )
            attach_metadata(
                launcher,
                kernel_name=(
                    self.inductor_meta.get("kernel_name")
                    or getattr(fn, "__name__", "")
                    or "triton_kernel"
                ),
                kernel_stub=scope["function"],
                kernel_stub_owner=binary,
                get_grid=get_grid,
                grid=grid,
                def_args=def_args,
                compile_meta=compile_meta,
                binary=binary,
                launcher_enter=scope.get("launch_enter_hook"),
                launcher_exit=scope.get("launch_exit_hook"),
            )
        except Exception:
            return launcher
        return launcher

    return [
        (
            compile_cls,
            "_npu_fast_launch_original_make_launcher",
            original_make_launcher,
        ),
        (compile_cls, "make_launcher", patched_make_launcher),
        (compile_cls, _TRITON_PATCHED_ATTR, True),
    ]


def _apply_patch_operations(operations: list[tuple[Any, str, Any]]) -> None:
    applied = []
    try:
        for owner, name, value in operations:
            existed = hasattr(owner, name)
            previous = getattr(owner, name, None)
            setattr(owner, name, value)
            applied.append((owner, name, existed, previous))
    except Exception:
        for owner, name, existed, previous in reversed(applied):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        raise


def patch_fast_launch() -> bool:
    global _PATCHED
    if not npu_config.enable_fast_launch:
        return False
    if _PATCHED:
        return True

    wrapper_module = _resolve_wrapper_module()
    triton_module = _resolve_triton_heuristics_module()
    operations = [
        *_build_wrapper_patch_operations(wrapper_module),
        *_build_triton_patch_operations(triton_module),
    ]
    _apply_patch_operations(operations)
    _PATCHED = True
    return True


__all__ = ["patch_fast_launch"]
