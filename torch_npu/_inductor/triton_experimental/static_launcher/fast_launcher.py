# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU construction adapter for the community global-runner protocol."""

import logging
import types


log = logging.getLogger(__name__)


def build_fast_launcher(launcher):
    if not isinstance(launcher, types.FunctionType):
        return None
    if not getattr(launcher, "_is_static", False):
        return None
    # Regenerate bound runners only when a static codegen recipe is available.
    bound_runner = (
        "runner" in launcher.__code__.co_varnames
        or "runner" in launcher.__code__.co_freevars
    )
    global_runner = "runner" in launcher.__code__.co_names and not bound_runner
    spec = getattr(launcher, "_npu_fast_static_codegen", None)
    if not global_runner and spec is None:
        return None
    from torch_npu import _C

    try:
        runner = launcher.__globals__.get("runner")
        kernel = getattr(launcher, "_static_kernel_owner", None)
        if kernel is None:
            kernel = runner.__self__
        loaded = kernel.loaded_kernel
        if loaded is None:
            return None
        fast_type = getattr(_C, "_FastNpuLauncher", None)
        if fast_type is not None:
            fast_runner = fast_type(loaded)
            kind = "cpp_vectorcall"
            if not global_runner:
                optimized = _regenerate_launcher(
                    launcher, spec, fast_runner=fast_runner,
                )
            else:
                globals_copy = {**launcher.__globals__, "runner": fast_runner}
                optimized = types.FunctionType(
                    launcher.__code__, globals_copy, launcher.__name__,
                    launcher.__defaults__, launcher.__closure__,
                )
                optimized.__kwdefaults__ = (
                    dict(launcher.__kwdefaults__) if launcher.__kwdefaults__ else None
                )
        else:
            # Use the static pybind entry when the installed binary lacks C++ vectorcall.
            static_type = kernel._C_impl()
            if static_type is None or spec is None or not hasattr(static_type, "_launch_kernel"):
                return None
            optimized = _regenerate_launcher(
                launcher, spec, launch_kernel=static_type._launch_kernel,
                loaded_kernel=loaded,
            )
            kind = "python_inline_static"
        optimized.__dict__.update(launcher.__dict__)
        optimized._static_kernel_owner = kernel
        optimized._is_fast_static = True
        optimized._npu_fast_launcher_kind = kind
        return optimized
    except (AttributeError, KeyError, TypeError, ValueError):
        log.debug("NPU fast launcher is unavailable", exc_info=True)
        return None
    except Exception:
        log.warning("Unexpected error building NPU fast launcher", exc_info=True)
        return None


def _regenerate_launcher(launcher, spec, *, fast_runner=None,
                         launch_kernel=None, loaded_kernel=None):
    """Recreate an NPU static launcher from its verified grid/argument recipe."""
    from ..launcher_codegen import _gen_launcher_code

    def_args, runtime_args, grid_lines = spec
    scope = {**launcher.__globals__}
    if fast_runner is not None:
        scope["fast_runner"] = fast_runner
        call_args = ", ".join(("grid_0", "grid_1", "grid_2", "stream", *runtime_args))
        call_lines = [f"    fast_runner({call_args})"]
    else:
        scope.update(launch_kernel=launch_kernel, loaded_kernel=loaded_kernel)
        args_text = ", ".join(runtime_args)
        if len(runtime_args) == 1:
            args_text += ","
        call_lines = [
            "    launch_kernel(loaded_kernel, grid_0, grid_1, grid_2, "
            f"stream, ({args_text}))"
        ]
    return _gen_launcher_code(
        scope, def_args, (), grid_lines,
        bound_names=("fast_runner", "launch_kernel", "loaded_kernel",
                     "_grid_cache", "max", "min"),
        call_lines=call_lines,
    )
