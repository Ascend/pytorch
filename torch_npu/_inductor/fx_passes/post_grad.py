import functools

import torch.utils._pytree as pytree
from torch._inductor.fx_passes import post_grad
from torch._inductor.pattern_matcher import LoweringPatternEntry


def _is_npu_match(match):
    return any(
        getattr(getattr(value, "device", None), "type", None) == "npu"
        for node in pytree.tree_leaves((match.args, match.kwargs))
        for value in pytree.tree_leaves(getattr(node, "meta", {}).get("val"))
    )


def _npu_aware_extra_check(src_check):
    @functools.wraps(src_check)
    def extra_check(match):
        return not _is_npu_match(match) and src_check(match)

    return extra_check


def patch_pattern_mm_plus_mm():
    # Keep the shared pattern registered for other devices. NPU does not yet
    # support this lowering, so only narrow its applicability predicate.
    seen_entries = set()
    for entries in post_grad.pass_patterns[1].patterns.values():
        for entry in entries:
            if (
                id(entry) not in seen_entries
                and isinstance(entry, LoweringPatternEntry)
                and entry.handler is post_grad.mm_plus_mm
            ):
                seen_entries.add(id(entry))
                entry.extra_check = _npu_aware_extra_check(entry.extra_check)
