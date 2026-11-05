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
    # Legacy NPU generators bind runner as a default; replacing globals there
    # is ineffective. Only accept the new static generator's protocol.
    if "runner" in launcher.__code__.co_varnames or "runner" in launcher.__code__.co_freevars:
        return None
    try:
        from torch_npu._C import _FastNpuLauncher
    except ImportError:
        return None

    try:
        kernel = launcher.__globals__["runner"].__self__
        loaded = kernel.loaded_kernel
        if loaded is None:
            return None
        fast_runner = _FastNpuLauncher(loaded)
        globals_copy = {**launcher.__globals__, "runner": fast_runner}
        optimized = types.FunctionType(
            launcher.__code__, globals_copy, launcher.__name__,
            launcher.__defaults__, launcher.__closure__,
        )
        optimized.__kwdefaults__ = (
            dict(launcher.__kwdefaults__) if launcher.__kwdefaults__ else None
        )
        optimized.__dict__.update(launcher.__dict__)
        optimized._static_kernel_owner = kernel
        optimized._is_fast_static = True
        return optimized
    except (AttributeError, KeyError, TypeError, ValueError):
        log.debug("NPU fast launcher is unavailable", exc_info=True)
        return None
    except Exception:
        log.warning("Unexpected error building NPU fast launcher", exc_info=True)
        return None
