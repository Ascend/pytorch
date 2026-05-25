"""FA v3 graph partition pass via Inductor's PatternMatcherPass + POST_GRAD_PATTERNS.

Identifies FA v3 forward/backward FX nodes that hit the dropout-on-TND path
(incompatible with ACLgraph capture) and rewrites their ``target`` to the
proxy ops registered in :mod:`._proxy_ops`. Those proxies carry
``Tag.cudagraph_unsafe``, so inductor's scheduler partitions them out of the
captured graph and falls back to eager.

Wired up via :func:`register_fav3_partition_pass`, called from
``torch_npu/_inductor/__init__.py``.
"""

__all__ = ["register_fav3_partition_pass"]

import torch
from torch._inductor import config as inductor_config
from torch._inductor.fx_passes.post_grad import POST_GRAD_PATTERNS
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.operator_schemas import normalize_function

from . import _proxy_ops


_PASS_KEY = "fav3_partition"

# Module-level singleton -- Python module cache guarantees one-time registration
# of the rules below. Do NOT importlib.reload this module (would re-append
# handlers to _FAV3_PMP.patterns and raise on re-define in _proxy_ops).
_FAV3_PMP = PatternMatcherPass(pass_name=_PASS_KEY)


def _make_check_and_handler(target):
    proxy = _proxy_ops.PROXY_TARGETS[target]

    def _check(match):
        # Early-out: when ACLgraph isn't going to capture, scheduler skips
        # partitioning entirely (scheduler.py: "partition includes all ops
        # when cudagraphs is disabled"); rewriting only adds a dispatcher hop.
        if not inductor_config.triton.cudagraphs:
            return False

        node = match.nodes[0]
        normalized = normalize_function(
            node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is None:
            return False
        _, kwargs = normalized

        keep_prob = kwargs.get("keep_prob", 1.0)
        input_layout = kwargs.get("input_layout")
        if not isinstance(input_layout, str):
            return False
        if not isinstance(keep_prob, (int, float)):
            return False
        return input_layout.upper() == "TND" and float(keep_prob) < 1.0

    def _handler(match, *args, **kwargs):
        node = match.nodes[0]
        node.target = proxy

    return _check, _handler


def _register_rule_for(target):
    check, handler = _make_check_and_handler(target)
    register_graph_pattern(
        CallFunctionVarArgs(target),
        extra_check=check,
        pass_dict=_FAV3_PMP,
    )(handler)


# Module-level rule registration. Both fwd and bwd targets feed into the same
# _FAV3_PMP instance; inductor will dispatch on whichever matches per-graph.
_register_rule_for(torch.ops.npu.npu_fusion_attention_v3.default)
_register_rule_for(torch.ops.npu.npu_fusion_attention_grad_v3.default)


def register_fav3_partition_pass() -> None:
    """Mount the PMP into inductor's post-grad fusion dispatch.

    Called once from ``torch_npu/_inductor/__init__.py`` alongside the other
    inductor extension hooks. Idempotent: dict assignment is overwrite-safe;
    setdefault preserves any user-supplied value.
    """
    POST_GRAD_PATTERNS[_PASS_KEY] = _FAV3_PMP
    inductor_config.post_grad_fusion_options.setdefault(_PASS_KEY, {})
