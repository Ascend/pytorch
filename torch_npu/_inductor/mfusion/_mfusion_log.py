# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MFusion debug logging: default Python logging drops INFO unless a handler exists."""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def mfusion_env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True")


def mfusion_debug_enabled() -> bool:
    return mfusion_env_truthy("TORCH_NPU_MFUSION_DEBUG")


def mfusion_trace_enabled() -> bool:
    """stderr phase markers (always flushed). Use to see where compile hangs."""
    return mfusion_env_truthy("TORCH_NPU_MFUSION_TRACE") or mfusion_debug_enabled()


def mfusion_stderr(msg: str) -> None:
    if mfusion_trace_enabled():
        print(f"[mfusion][trace] {msg}", file=sys.stderr, flush=True)


def configure_mfusion_logging() -> None:
    """Attach a stderr handler so logger.info under torch_npu._inductor.mfusion is visible."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not (mfusion_debug_enabled() or mfusion_trace_enabled()):
        return
    _CONFIGURED = True
    pkg = logging.getLogger("torch_npu._inductor.mfusion")
    pkg.setLevel(logging.DEBUG if mfusion_debug_enabled() else logging.INFO)
    if not pkg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        pkg.addHandler(h)
    # Do not double-print via root (often WARNING-only).
    pkg.propagate = False
