# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU values for the native ``torch._dynamo.package.SystemInfo``.

A Dynamo compile package is only installed when
``SystemInfo.check_compatibility`` accepts the saving environment. Native
``current()`` knows CUDA and XPU only, so on NPU it records no chip name and no
toolkit version, and every environment check passes: a package built on one
Ascend product loads on another.

The native comparison rules are already the right ones, so this module keeps the
native dataclass and fills its two existing fields instead of adding a private
one:

``gpu_name``
    Chip name of the saving device, for example ``Ascend910B3``.
``toolkit_version``
    CANN version, for example ``9.2.0``. Native reads this from
    ``torch.version.<device_type>``, and ``torch.version.npu`` does not exist, so
    the value has to come from ``torch_npu``.

``npu`` is added to ``CHECK_GPUS`` so the native branch that compares the two
fields runs at all.

The producer needs the same values as the consumer. ``_DynamoCacheEntry`` gets
its default ``SystemInfo`` from a dataclass ``default_factory`` captured as a
closure cell when the class was created, so replacing ``SystemInfo.current``
alone only affects the consumer side. :func:`.patches.cache_entry` therefore
stores :func:`current` on the entry it creates.
"""
import logging
import platform

log = logging.getLogger(__name__)
DEVICE_TYPE = "npu"
_MARKER = "_torch_npu_system_info"


def cann_version():
    """CANN version, or None when torch_npu cannot report one."""
    try:
        from torch_npu.npu.utils import get_cann_version

        version = get_cann_version("CANN")
    except Exception:
        log.debug("CANN version unavailable", exc_info=True)
        return None
    if not version or str(version).lower() in ("unknown", "none"):
        return None
    return str(version)


def chip_name(index=0):
    """Name of the current Ascend product, or None when NPU is unusable."""
    import torch

    if not getattr(torch, DEVICE_TYPE).is_available():
        return None
    try:
        return torch.npu.get_device_name(index) or None
    except Exception:
        log.debug("NPU chip name unavailable", exc_info=True)
        return None


def current(cls):
    """NPU-aware replacement for ``SystemInfo.current``.

    Mirrors the native body so that a save and a load in the same environment
    produce equal fields, including ``triton_version``.
    """
    import torch
    from torch.utils._triton import get_triton_version

    gpu_name = toolkit_version = None
    for device_type in cls.CHECK_GPUS:
        if not getattr(torch, device_type).is_available():
            continue
        try:
            if device_type == DEVICE_TYPE:
                gpu_name, toolkit_version = chip_name(), cann_version()
            else:
                gpu_name = getattr(torch, device_type).get_device_name()
                toolkit_version = getattr(torch.version, device_type)
        except Exception:
            log.debug("System info collection failed for %s", device_type, exc_info=True)
            gpu_name = toolkit_version = None
        if gpu_name is not None:
            break
    return cls(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        toolkit_version=toolkit_version,
        triton_version=get_triton_version((0, 0)),
        gpu_name=gpu_name,
    )


def install(packages):
    """Make the native package check aware of NPU. Idempotent and cheap."""
    info = packages.SystemInfo
    if DEVICE_TYPE not in info.CHECK_GPUS:
        info.CHECK_GPUS = (*info.CHECK_GPUS, DEVICE_TYPE)
    original = getattr(info.current, "__func__", info.current)
    if getattr(original, _MARKER, False):
        return
    setattr(current, _MARKER, True)
    info.current = classmethod(current)
