# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Runtime NPU device-capability probes for the triton_experimental backend.

These are NOT static config (see config.py) -- they query the *live* NPU via
``torch.npu.get_device_properties`` / ``get_soc_version`` and are cached with
``functools.lru_cache``. They live outside config.py on purpose:
``install_config_module`` cannot host an ``lru_cache`` wrapper -- its config
walker rejects the ``functools._lru_cache_wrapper`` type -- and, more to the
point, hardware probes are not tunable configuration.
"""

import functools as _functools

import torch_npu


def _soc_version_no_init():
    """Driver-level read-only soc lookup that must not initialize the device.

    Verified to return the same value from an uninitialized process and from
    a forked child of an initialized one.  The public get_soc_version()
    triggers _lazy_init(), which at import time (these probes are reached
    from module-level calls in config.py / npu_triton_heuristics.py, also
    inside Inductor forked compile workers) raises "Cannot re-initialize NPU
    in forked subprocess" and kills every >=2-kernel compilation.
    """
    return torch_npu._C._npu_get_soc_version()


@_functools.lru_cache(maxsize=None)
def get_npu_vector_core_count() -> int:
    """AI-vector core count for the current NPU (kernel total_thread, dispatch
    grid, multi_processor_count). Prefer prop.vector_core_num, else 2*cube_core_num
    for Ascend910B / 910_9391+; falls back to 40 (Ascend910B3)."""
    try:
        import torch
        prop = torch.npu.get_device_properties(torch.npu.current_device())
        vector = getattr(prop, "vector_core_num", None)
        if vector:
            return int(vector)
        cube = getattr(prop, "cube_core_num", None)
        try:
            soc = _soc_version_no_init()
        except Exception:
            soc = None
        if cube and soc is not None and (220 <= soc < 240 or soc >= 250):
            return int(cube) * 2
        mpc = getattr(prop, "multi_processor_count", None)
        if mpc:
            return int(mpc)
    except Exception:
        pass
    return 40


@_functools.lru_cache(maxsize=None)
def get_npu_ub_size_bytes() -> int:
    """Unified Buffer capacity in bytes: 192 KiB on Ascend910B-class, 256 KiB on
    Ascend950 (soc >= 260). Note this threshold (260) differs from is_a5's 250 --
    a 910_95 at soc 250..259 still has a 192 KiB UB. Falls back to 192 KiB."""
    try:
        soc = _soc_version_no_init()
        if soc is not None and soc >= 260:  # Ascend950
            return 256 * 1024
    except Exception:
        pass
    return 192 * 1024


@_functools.lru_cache(maxsize=None)
def is_a5() -> bool:
    """True on Ascend A5 (910_95, soc >= 250). A5 schedules programs across cores
    itself, so it uses one-program-per-tile instead of the intra-core group-dispatch
    loop (see group_dispatch). Auto-detected by soc; override with
    ``config.force_is_a5``."""
    from . import config as ncfg
    if ncfg.force_is_a5 is not None:
        return ncfg.force_is_a5
    try:
        soc = _soc_version_no_init()
        return soc is not None and soc >= 250
    except Exception:
        return False
