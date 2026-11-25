"""Framework for converter-only graph rewrites.

A *graph rewrite* is a ``GraphModule -> None`` pass run by the converter graph builder
before FX nodes lower to IR (``apply_converter_graph_rewrites``). Writers register
rewrites with ``register_converter_graph_rewrite``; later registrations run after
earlier ones.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from torch.fx import GraphModule

GraphRewrite = Callable[[GraphModule], None]
_REWRITES: List[Tuple[str, GraphRewrite]] = []


def register_converter_graph_rewrite(rewrite: GraphRewrite, name: Optional[str] = None) -> GraphRewrite:
    """Append a named rewrite. Later registrations run after earlier ones.

    Names make the registration order auditable and reject silent duplicates.
    """
    rewrite_name = name or getattr(rewrite, "__name__", None) or repr(rewrite)
    if any(existing == rewrite_name for existing, _ in _REWRITES):
        raise ValueError(f"converter graph rewrite '{rewrite_name}' is already registered")
    _REWRITES.append((rewrite_name, rewrite))
    return rewrite


def apply_converter_graph_rewrites(gm: GraphModule) -> None:
    """Run every registered rewrite. Caller is the converter graph builder."""
    for _, rewrite in _REWRITES:
        rewrite(gm)
