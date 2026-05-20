# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Avoid RecursionError in torch._inductor.exc.OperatorIssue.operator_str.

When lowering fails, Inductor formats ``LoweringException`` with ``operator_str(target, args, kwargs)``.
Each ``arg`` is turned into a string via ``f"{arg}"``. For large graphs whose ``output`` tuple lists
hundreds of ``permute``/custom nodes, IR objects may stringify with deep recursion and mask the real
exception with ``RecursionError``.

Applied from ``torch_npu._inductor`` on import; disable with
``TORCH_NPU_SAFE_INDUCTOR_ERROR_STR=0``.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any


_APPLIED = False

_MAX_DEPTH = 8
_MAX_SEQ = 32
_MAX_STR = 512


def _safe_repr_arg(x: Any, depth: int = 0) -> str:
    if depth > _MAX_DEPTH:
        return "<?>"
    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        try:
            s = repr(x)
        except Exception as exc:
            return f"<repr failed: {exc!r}>"
        return s if len(s) <= _MAX_STR else s[: _MAX_STR - 3] + "..."

    if isinstance(x, (list, tuple)):
        n = len(x)
        if n > _MAX_SEQ:
            head = ", ".join(_safe_repr_arg(t, depth + 1) for t in x[:4])
            return f"{type(x).__name__}(len={n}, head=[{head}], ...)"
        inner = ", ".join(_safe_repr_arg(t, depth + 1) for t in x)
        return f"({inner})" if isinstance(x, tuple) else f"[{inner}]"

    if isinstance(x, dict):
        if len(x) > _MAX_SEQ:
            return f"dict(len={len(x)})"
        parts = []
        for i, (k, v) in enumerate(x.items()):
            if i >= _MAX_SEQ:
                parts.append("...")
                break
            parts.append(
                f"{_safe_repr_arg(k, depth + 1)}: {_safe_repr_arg(v, depth + 1)}"
            )
        return "{" + ", ".join(parts) + "}"

    cls = type(x)
    mod = getattr(cls, "__module__", "") or ""
    name = cls.__name__
    # Inductor IR / TensorBox: never call str()/repr() — can recurse through the whole graph.
    if "torch._inductor" in mod or mod.endswith("inductor.ir"):
        return f"{name}(id=0x{id(x):x})"

    try:
        s = repr(x)
    except Exception as exc:
        return f"<{name} repr failed: {exc!r}>"
    if len(s) > _MAX_STR:
        return s[: _MAX_STR - 3] + "..."
    return s


def _safe_operator_str(target: Any, args: Any, kwargs: Any) -> str:
    lines = [f"target: {target}"] + [
        f"args[{i}]: {_safe_repr_arg(arg)}" for i, arg in enumerate(args)
    ]
    if kwargs:
        lines.append(f"kwargs: {_safe_repr_arg(kwargs)}")
    return textwrap.indent("\n".join(lines), " ")


def apply_safe_operator_str_patch() -> None:
    """Replace ``OperatorIssue.operator_str`` with depth-bounded formatting."""
    global _APPLIED
    if _APPLIED:
        return
    from torch._inductor.exc import OperatorIssue

    OperatorIssue.operator_str = staticmethod(_safe_operator_str)  # type: ignore[assignment]
    _APPLIED = True


def apply_safe_operator_str_patch_if_enabled() -> None:
    if os.environ.get("TORCH_NPU_SAFE_INDUCTOR_ERROR_STR", "1").strip() not in (
        "0",
        "false",
        "False",
    ):
        apply_safe_operator_str_patch()
