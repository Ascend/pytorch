"""Symbolic-shape primitives for dynamic-shape graph optimization.

Provides device-independent, guard-free symbolic reasoning helpers for the graph
passes under ``ascend_custom_passes``, so optimizations that used to fire only on
static shapes (integer dims) also work on dynamic shapes (``torch.SymInt`` dims).

Design principles:
- Three-valued logic: a symbolic comparison is statically-true / statically-false
  / undecidable; optimize only when statically-true.
- Guard-free: rely on ``statically_known_true`` / ``bound_sympy`` style static
  reasoning, never adding guards to ``ShapeEnv`` nor changing recompile bounds.
- Pure-static inputs fall back to the original semantics (zero static regression).
"""

import os
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch

from ...config import log


try:
    from torch.fx.experimental.symbolic_shapes import statically_known_true as _skt
except Exception:  # pragma: no cover - fallback for older torch
    _skt = None


Number = Union[int, "torch.SymInt"]


# ---------------------------------------------------------------------------
# Global switch
# ---------------------------------------------------------------------------
def dynamic_fx_pass_enabled() -> bool:
    """Master switch for dynamic-shape optimization (on by default).

    When set to 0/false/off, passes fall back to the legacy static behavior.
    """
    return os.environ.get("NPU_INDUCTOR_DYNAMIC_FX_PASS", "1").lower() not in (
        "0",
        "false",
        "off",
    )


# ---------------------------------------------------------------------------
# Three-valued symbolic predicates
# ---------------------------------------------------------------------------
def statically_true(expr) -> bool:
    """Return True only if a bool/SymBool expression is provably true; never adds a guard.

    Returns False for statically-false and undecidable cases. When the switch is
    off, only python bools are honored (symbols are treated as undecidable) so the
    whole layer degrades to static behavior. Prefers the official
    ``statically_known_true``; falls back to ShapeEnv static evaluation (also
    guard-free) on older torch.
    """
    if isinstance(expr, bool):
        return expr
    if not dynamic_fx_pass_enabled():
        return False
    if _skt is not None:
        try:
            return bool(_skt(expr))
        except Exception:
            pass
    return _static_eval_symbool(expr)


def _static_eval_symbool(expr) -> bool:
    """Fallback: statically evaluate a SymBool via its sympy expr + ShapeEnv, guard-free."""
    node = getattr(expr, "node", None)
    sym_expr = getattr(node, "expr", None)
    if sym_expr is None:
        return False
    try:
        import sympy

        if sym_expr == sympy.true:
            return True
        shape_env = getattr(node, "shape_env", None)
        evaluator = getattr(shape_env, "_maybe_evaluate_static", None)
        if evaluator is None:
            return False
        result = evaluator(sym_expr)
        return result is not None and result == sympy.true
    except Exception:
        return False


def statically_known_eq(a: Number, b: Number) -> bool:
    """Whether a == b provably holds."""
    try:
        return statically_true(a == b)
    except Exception:
        return False


def statically_known_geq(a: Number, b: Number) -> bool:
    """Whether a >= b provably holds."""
    try:
        return statically_true(a >= b)
    except Exception:
        return False


def statically_known_gt(a: Number, b: Number) -> bool:
    """Whether a > b provably holds."""
    try:
        return statically_true(a > b)
    except Exception:
        return False


def statically_known_leq(a: Number, b: Number) -> bool:
    """Whether a <= b provably holds."""
    try:
        return statically_true(a <= b)
    except Exception:
        return False


def is_statically_one(d: Number) -> bool:
    """Whether a dim is provably 1."""
    return statically_known_eq(d, 1)


def has_free_symbols(shape) -> bool:
    """Whether a shape contains any symbolic dim."""
    if shape is None:
        return False
    return any(isinstance(d, torch.SymInt) for d in shape)


def shapes_statically_equal(s1, s2) -> bool:
    """Whether two shapes have equal rank and provably equal dims."""
    if s1 is None or s2 is None:
        return False
    if len(s1) != len(s2):
        return False
    return all(statically_known_eq(a, b) for a, b in zip(s1, s2))


# ---------------------------------------------------------------------------
# Symbol-aware shape reading
# ---------------------------------------------------------------------------
def get_symbolic_shape(node):
    """Read a node's shape keeping symbolic dims; None when no shape info.

    Equivalent to ``get_binary_fold_result.get_node_shape(allow_symbolic=True)``,
    offered for call sites that do not import that module.
    """
    if not isinstance(node, torch.fx.Node):
        return None
    meta = node.meta
    val = meta.get("val", None)
    if val is not None and hasattr(val, "shape"):
        return val.shape
    example = meta.get("example_value", None)
    if example is not None and hasattr(example, "shape"):
        return example.shape
    tensor_meta = meta.get("tensor_meta", None)
    if tensor_meta is not None and hasattr(tensor_meta, "shape"):
        return tensor_meta.shape
    return None


# ---------------------------------------------------------------------------
# Size-argument normalization
# ---------------------------------------------------------------------------
def resolve_size_arg(arg) -> Optional[Number]:
    """Normalize one size argument to int / SymInt; None when unresolvable.

    Handles four forms: python int, SymInt, ``sym_size``-like nodes (meta is
    SymInt/int), and PRE-graph derived nodes whose ``example_value`` is SymInt/int.
    """
    if isinstance(arg, bool):
        return None
    if isinstance(arg, int):
        return arg
    # Switch off: do not resolve symbolic sources (legacy "static int only").
    if not dynamic_fx_pass_enabled():
        return None
    if isinstance(arg, torch.SymInt):
        return arg
    if isinstance(arg, torch.fx.Node):
        for key in ("val", "example_value"):
            v = arg.meta.get(key, None)
            if isinstance(v, torch.SymInt):
                return v
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def resolve_size_list(args) -> Optional[List[Number]]:
    """Normalize a size list element-wise; None if any element is unresolvable."""
    if not isinstance(args, (list, tuple)):
        return None
    resolved = []
    for a in args:
        r = resolve_size_arg(a)
        if r is None:
            return None
        resolved.append(r)
    return resolved


# ---------------------------------------------------------------------------
# Symbolic shape materialization
# ---------------------------------------------------------------------------
def _sym_key(value) -> Optional[str]:
    """Canonical string key for a size value (ints use their value), for equivalence."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, torch.SymInt):
        try:
            return f"sym:{value.node.expr}"
        except Exception:
            return None
    return None


def _iter_anchor_nodes(anchors):
    if anchors is None:
        return
    if isinstance(anchors, torch.fx.Node):
        yield anchors
        return
    for a in anchors:
        if isinstance(a, torch.fx.Node):
            yield a


def materialize_shape(graph, shape, anchors) -> Optional[list]:
    """Turn a (possibly symbolic) target shape into a node-writable arg list.

    - int dims are kept as-is;
    - SymInt dims: insert ``aten.sym_size.int`` referencing the same-valued dim of
      some anchor (an anchor is an input of the rewritten op, so it dominates the
      insertion point -> dominance-safe);
    - returns None if any SymInt dim has no source (rewrite is abandoned).

    Caller must manage the insertion point via ``graph.inserting_before(node)``.
    """
    # Switch off: refuse to emit sym_size for symbolic dims -> degrade to legacy
    # static behavior (callers then skip the rewrite).
    if not dynamic_fx_pass_enabled() and has_free_symbols(shape):
        return None
    result = []
    created = {}
    for dim in shape:
        if isinstance(dim, bool):
            return None
        if isinstance(dim, int):
            result.append(dim)
            continue
        if not isinstance(dim, torch.SymInt):
            return None
        key = _sym_key(dim)
        if key is None:
            return None
        if key in created:
            result.append(created[key])
            continue
        node = _make_sym_size(graph, dim, key, anchors)
        if node is None:
            return None
        created[key] = node
        result.append(node)
    return result


def _make_sym_size(graph, sym, key, anchors):
    """Build a sym_size node for symbolic dim ``sym`` from a same-valued anchor dim."""
    for anchor in _iter_anchor_nodes(anchors):
        val = anchor.meta.get("val", None)
        if val is None or not hasattr(val, "shape"):
            continue
        for dim_idx, s in enumerate(val.shape):
            if isinstance(s, torch.SymInt) and _sym_key(s) == key:
                size_node = graph.call_function(
                    torch.ops.aten.sym_size.int, args=(anchor, dim_idx)
                )
                size_node.meta["val"] = s
                return size_node
    return None


# ---------------------------------------------------------------------------
# Symbolic value-range analysis
# ---------------------------------------------------------------------------
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def symbolic_value_range(v) -> Optional[Tuple]:
    """Provable compile-time range [lo, hi] of a value (ints give (v, v)); None if unknown.

    Relies on ShapeEnv.bound_sympy (available on newer torch); returns None if absent.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, torch.SymInt):
        if not dynamic_fx_pass_enabled():
            return None
        try:
            node = v.node
            bound = getattr(node.shape_env, "bound_sympy", None)
            if bound is None:
                return None
            vr = bound(node.expr)
            return (vr.lower, vr.upper)
        except Exception:
            return None
    return None


def statically_fits_int32(*vals) -> bool:
    """Whether all given values provably lie within the int32 range.

    Implemented with three-valued comparisons (no bound_sympy dependency); a
    symbolic value counts as fitting only when provably bounded.
    """
    if not vals:
        return False
    for v in vals:
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            if v < _INT32_MIN or v > _INT32_MAX:
                return False
        elif isinstance(v, torch.SymInt):
            if not (
                statically_known_leq(v, _INT32_MAX)
                and statically_known_geq(v, _INT32_MIN)
            ):
                return False
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Symbol-aware fake meta propagation
# ---------------------------------------------------------------------------
def _shape_env_from_val(val):
    if isinstance(val, torch.SymInt):
        try:
            return val.node.shape_env
        except Exception:
            return None
    fake_mode = getattr(val, "fake_mode", None)
    return getattr(fake_mode, "shape_env", None)


def get_shape_env(graph):
    """Resolve the ShapeEnv from any fake-carrying node in the graph; None if none."""
    for n in graph.nodes:
        for key in ("val", "example_value"):
            se = _shape_env_from_val(n.meta.get(key, None))
            if se is not None:
                return se
    return None


def get_fake_mode(graph):
    """Resolve the FakeTensorMode from any FakeTensor node in the graph; None if none."""
    for n in graph.nodes:
        val = n.meta.get("val", None)
        fake_mode = getattr(val, "fake_mode", None)
        if fake_mode is not None:
            return fake_mode
    return None


def _resolve_fake(arg):
    if isinstance(arg, torch.fx.Node):
        return arg.meta.get("val", arg)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_resolve_fake(x) for x in arg)
    return arg


def refresh_fake_meta(node, fake_mode) -> bool:
    """Recompute node.meta['val'] from current args/kwargs under fake_mode; keep as-is on failure."""
    if fake_mode is None:
        return False
    try:
        with fake_mode:
            new_val = node.target(
                *[_resolve_fake(a) for a in node.args],
                **{k: _resolve_fake(v) for k, v in node.kwargs.items()},
            )
        node.meta["val"] = new_val
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lightweight hit/skip stats (debug-level only, avoids log spam)
# ---------------------------------------------------------------------------
_STATS = defaultdict(lambda: {"hit": 0, "skip": 0})


def note_hit(pass_name: str, n: int = 1) -> None:
    _STATS[pass_name]["hit"] += n


def note_skip(pass_name: str, n: int = 1) -> None:
    _STATS[pass_name]["skip"] += n


def dump_stats(pass_name: str) -> None:
    """Emit a single hit/skip stat line for a pass (debug level)."""
    stat = _STATS.get(pass_name)
    if stat and (stat["hit"] or stat["skip"]):
        log.debug(
            "[dynamic_fx] %s: hit=%d skip_undecidable=%d",
            pass_name,
            stat["hit"],
            stat["skip"],
        )


def reset_stats() -> None:
    """Clear stats, mainly for tests."""
    _STATS.clear()
