# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Test-only helpers; importing this module does not initialize the NPU."""

import gc
import os
import tempfile
from contextlib import ExitStack, contextmanager
from unittest import mock


@contextmanager
def isolated_fast_config(enabled=True):
    import torch
    from torch._inductor import config
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.utils import clear_caches
    from torch_npu._inductor import config as npu_config

    with ExitStack() as stack:
        root = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(mock.patch.dict(os.environ, {
            "TORCHINDUCTOR_CACHE_DIR": os.path.join(root, "inductor"),
            "TRITON_CACHE_DIR": os.path.join(root, "triton"),
            "TRITON_INTERPRET": "0",
            "TRITON_DEVICE_PRINT": "false",
            "TRITON_REGISTER_TENSOR_MSPROF": "false",
            "TRITON_FAST_PTR": "false",
        }))
        stack.enter_context(config.patch({
            "compile_threads": 1,
            "cpp_wrapper": False,
            "triton.cudagraphs": False,
            "triton.store_cubin": False,
            "use_static_triton_launcher": True,
            "strict_static_triton_launcher": True,
            "use_fast_triton_launcher": enabled,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "autotune_remote_cache": False,
            "coordinate_descent_tuning": False,
        }))
        stack.enter_context(mock.patch.object(npu_config, "enable_fast_launch", False))
        torch._dynamo.reset()
        PyCodeCache.cache_clear()
        clear_caches()
        try:
            yield
        finally:
            # Finish queued work before cache cleanup can release kernel owners.
            if torch.npu.is_initialized():
                torch.npu.synchronize()
            torch._dynamo.reset()
            PyCodeCache.cache_clear()
            clear_caches()
            gc.collect()


def require_fast_type():
    import torch_npu

    fast_type = getattr(torch_npu._C, "_FastNpuLauncher", None)
    if fast_type is None:
        raise AssertionError(
            "_FastNpuLauncher is missing: build the fast implementation; "
            "a static fallback is not a passing fast test"
        )
    return fast_type


def unwrap_launcher(launcher):
    return getattr(launcher, "_npu_original_launcher", launcher)


def require_fast_runner(launcher):
    inner = unwrap_launcher(launcher)
    runner = getattr(inner, "__globals__", {}).get("runner")
    if not isinstance(runner, require_fast_type()):
        raise AssertionError("cached launcher is not backed by _FastNpuLauncher")
    if getattr(inner, "_static_kernel_owner", None) is None:
        raise AssertionError("fast launcher did not retain its static kernel owner")
    return runner


@contextmanager
def observe_fast_builds():
    """Retain actual build results and count calls, without faking the runner."""
    from torch_npu._inductor.triton_experimental.npu_triton_heuristics import (
        NPUCachingAutotuner,
    )

    original = NPUCachingAutotuner._build_fast_launcher
    records = []

    def build(tuner, baseline):
        optimized = original(tuner, baseline)
        record = {"tuner": tuner, "baseline": baseline, "fast": optimized, "calls": 0}
        records.append(record)
        if optimized is None:
            return None
        require_fast_runner(optimized)

        def counted(*args, **kwargs):
            record["calls"] += 1
            return optimized(*args, **kwargs)

        counted.__dict__.update(optimized.__dict__)
        return counted

    with mock.patch.object(NPUCachingAutotuner, "_build_fast_launcher", build):
        yield records
