"""Host-gm simplify passes for the fxrt fx_wrapper, run from
``InferrtFxWrapper.compile_graph``.

simplify_duplicate_symints
--------------------------
The host gm can carry one symbolic value as two graph objects: a symint
placeholder plus a redundant `aten.sym_size.int` node, each feeding a different
set of ops. Running upstream
`torch._inductor.fx_passes.dedupe_symint_uses.dedupe_symints` on the host gm
collapses them onto the placeholder.

Gate: on by default; set FX_SYM_DEDUP=0 to disable.
"""
from __future__ import annotations

import os

import torch

from fxrt._logging import get_logger

_log = get_logger(__name__)


def _enabled() -> bool:
    return os.environ.get("FX_SYM_DEDUP", "1") != "0"


def simplify_duplicate_symints(gm: torch.fx.GraphModule) -> int:
    """Dedup symint nodes on the host gm by delegating to upstream
    `dedupe_symints`. Returns the number of nodes it erased (0 if disabled, the
    upstream pass is unavailable, or there was nothing to collapse)."""
    if not _enabled():
        return 0

    try:
        # pylint: disable=import-outside-toplevel
        from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints
    except Exception:  # pylint: disable=broad-exception-caught
        # private module moved/renamed across torch versions
        _log.warning("[fx_simplify] upstream dedupe_symints unavailable", exc_info=True)
        return 0

    before = len(gm.graph.nodes)
    dedupe_symints(gm.graph)
    removed = before - len(gm.graph.nodes)

    if removed:
        gm.graph.lint()
        gm.recompile()
        _log.info(
            "[fx_simplify] simplify_duplicate_symints (dedupe_symints): "
            "erased %d duplicate symint node(s)", removed
        )
    return removed
