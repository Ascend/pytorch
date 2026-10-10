"""Script-side controls for eager DVM lazy fusion."""

import contextlib

import torch_npu

__all__ = ["set_enable", "set_disable", "enabled", "disabled",
           "set_dump_enable", "set_dump_disable", "dump_enabled", "dump_disabled"]


def set_enable() -> None:
    r"""Enable the process-wide script-side gate for eager DVM fusion.

    Requires ``TORCH_NPU_LAZY_FUSION`` to be enabled before importing
    torch_npu. This gate cannot override the environment master switch.
    The setting persists until changed; unlike a context manager, this
    function does not automatically restore the previous state.

    A state change flushes the pending graph without waiting for device
    execution to finish. Call at region boundaries, without concurrent
    graph producers. Context managers restore their entry state on exit,
    including when this setter is called inside the context.
    """
    torch_npu._C._lazy_fusion._set_disabled(False)


def set_disable() -> None:
    r"""Disable eager DVM fusion until the script-side gate is changed again.

    Uses the same process-wide gate and graph boundaries as :func:`set_enable`.
    Context managers still restore their entry state on exit.
    """
    torch_npu._C._lazy_fusion._set_disabled(True)


@contextlib.contextmanager
def enabled(enabled: bool = True):
    r"""Temporarily enable or disable eager DVM fusion for a code region.

    The ``TORCH_NPU_LAZY_FUSION`` environment variable still controls whether
    fusion is enabled globally. This context only provides a script-side
    override, and ``enabled=True`` cannot enable fusion when the environment
    variable disabled it.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    previous_disabled = torch_npu._C._lazy_fusion._set_disabled(not enabled)
    try:
        yield
    finally:
        torch_npu._C._lazy_fusion._set_disabled(previous_disabled)


def disabled():
    """Return a context manager that disables eager DVM fusion."""
    return enabled(False)


def set_dump_enable() -> None:
    r"""Enable eager DVM text dumps for subsequent fusion graphs.

    Requires ``dump_as_text`` in ``TORCH_NPU_LAZY_FUSION`` before importing
    torch_npu. This process-wide script-side gate defaults to True, preserving
    full dumping when only the environment flag is used. Call
    :func:`set_dump_disable` before model execution to capture only selected
    regions. The setting persists until changed.

    A state change flushes the pending graph. Already queued graphs retain
    their dump state, including when codegen runs on the task-queue thread.
    Call at iteration/region boundaries, without concurrent graph producers.
    """
    torch_npu._C._lazy_fusion._set_dump_enabled(True)


def set_dump_disable() -> None:
    r"""Disable eager DVM text dumps until the script-side gate changes again.

    Uses the same graph boundaries as :func:`set_dump_enable`. Already queued
    graphs retain their dump state; contexts restore their entry state on exit.
    """
    torch_npu._C._lazy_fusion._set_dump_enabled(False)


@contextlib.contextmanager
def dump_enabled(enabled: bool = True):
    r"""Temporarily control eager DVM text dumping for a code region.

    Uses the same environment master switch and graph boundaries as
    :func:`set_dump_enable`. Restores the previous script-side state on
    exit, including nested contexts and exceptions. No step number is needed.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    previous_enabled = torch_npu._C._lazy_fusion._set_dump_enabled(enabled)
    try:
        yield
    finally:
        torch_npu._C._lazy_fusion._set_dump_enabled(previous_enabled)


def dump_disabled():
    """Temporarily disable text dumping, restoring the entry state on exit.

    Equivalent to ``dump_enabled(False)``; supports nesting and exception exits.
    Does not disable fusion itself.
    """
    return dump_enabled(False)
