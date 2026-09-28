# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Provable nonnegative-clamp skip: the proof engine.

Given a masked load whose index can go negative at the origin, decide
whether the effective mask PROVABLY implies index >= 0 on kept lanes.
A proven skip is a mathematical no-op (max(idx,0) == idx on kept lanes;
masked-out lanes are never dereferenced). Everything else — unproven,
prover error, missing structural source — keeps the clamp (the historical
95c9094da7 behavior). See config.nonnegative_clamp_mode.

Imported by codegen/triton.py at the load-index emission site.
"""

import ast
import logging

import sympy

from torch._dynamo.utils import counters
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch._inductor.codegen.triton import IndexingOptions

from .. import config as ncfg

log = logging.getLogger("torch._inductor")

def _npu_body_defs(kernel):
    """name -> RHS ast node for ``name = rhs`` lines already emitted."""
    lines_ref = getattr(getattr(kernel, "compute", None), "get_lines_ref", None)
    if lines_ref is None:
        return {}
    lines = lines_ref()
    tag = (len(lines), str(lines[-1]) if lines else "")
    cached = getattr(kernel, "_npu_ast_defs_cache", None)
    if cached is not None and cached[0] == tag:
        return cached[1]
    defs = {}
    for line in lines:
        text = str(line).strip()
        if not text or text.startswith(("#", "@")):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # loop/if headers and continuations are not assignments
        stmt = tree.body[0] if len(tree.body) == 1 else None
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            defs.setdefault(stmt.targets[0].id, stmt.value)
    kernel._npu_ast_defs_cache = (tag, defs)
    return defs


def _npu_contains_arange(node):
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "arange"
        for n in ast.walk(node)
    )


def _npu_ast_eval(node, defs, axis_names, depth=0):
    """Evaluate an emitted-DSL expression node to ('axis', name) | ('const', k) | None.

    Covers the grammar mask predicates use: integer literals, ``tl.full(shape,
    k, dtype)`` constants, ``x.to(dtype)`` value-preserving casts,
    ``tl.broadcast_to(x, shape)`` unwraps, axis names (arange-derived names map
    to their base axis), and named references recursing into their definitions.
    Anything else yields None (no information, caller stays conservative).

    ``axis_names`` is the axis-symbol set from the kernel's range tree — the
    SINGLE structural identity source (upstream-maintained symbol->axis
    mapping). A name not in the set is not an axis, full stop: no naming
    convention, no guessing. Empty/missing set -> nothing is an axis -> the
    caller declines (sound).
    """
    if depth > 8:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return ("const", node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _npu_ast_eval(node.operand, defs, axis_names, depth + 1)
        if v is not None and v[0] == "const":
            return ("const", -v[1])
        return None
    if isinstance(node, ast.Name):
        if node.id in axis_names:
            return ("axis", node.id)
        if node.id in defs:
            rhs = defs[node.id]
            # ``<pfx>index = <pfx>offset + tl.arange(...)`` -> base axis var
            # (longest axis name prefixing the derived name).
            if _npu_contains_arange(rhs):
                cands = [a for a in axis_names if node.id.startswith(a)]
                if cands:
                    return ("axis", max(cands, key=len))
            return _npu_ast_eval(rhs, defs, axis_names, depth + 1)
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr == "full":  # tl.full(shape, value, dtype)
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, int)
            ):
                return ("const", node.args[1].value)
            return None
        if attr == "broadcast_to" and node.args:
            return _npu_ast_eval(node.args[0], defs, axis_names, depth)
        if attr == "to":  # (x).to(dtype)
            return _npu_ast_eval(node.func.value, defs, axis_names, depth)
    return None


def _npu_mask_conj_atoms(node, defs, atoms, axis_names, depth=0):
    """Accumulate ``axis >= k`` lower-bound atoms from a mask conjunction.

    ``a & b`` recurses on both sides; bare names recurse into their
    definitions (nested sub-masks like ``tmp5 = tmp2 & tmp4``); only ``>=``
    of an axis var against a constant contributes. Upper-bound atoms
    (``a < b``) and unknown operators contribute nothing — sound: no lower
    bound is inferred from them.
    """
    if depth > 8:
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        _npu_mask_conj_atoms(node.left, defs, atoms, axis_names, depth + 1)
        _npu_mask_conj_atoms(node.right, defs, atoms, axis_names, depth + 1)
        return
    if isinstance(node, ast.Name) and node.id in defs:
        _npu_mask_conj_atoms(defs[node.id], defs, atoms, axis_names, depth + 1)
        return
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.GtE)
    ):
        lhs = _npu_ast_eval(node.left, defs, axis_names)
        rhs = _npu_ast_eval(node.comparators[0], defs, axis_names)
        if (
            lhs is not None and lhs[0] == "axis"
            and rhs is not None and rhs[0] == "const"
        ):
            atoms[lhs[1]] = max(atoms.get(lhs[1], rhs[1]), rhs[1])


def _npu_mask_lower_atoms(mask_name, defs, axis_names):
    """Best-effort {axis_var: lower_bound} implied by a mask definition."""
    atoms = {}
    node = defs.get(mask_name)
    if node is not None:
        _npu_mask_conj_atoms(node, defs, atoms, axis_names)
    return atoms


def _npu_index_provably_nonnegative(expr, lb, axis_names):
    """True iff ``expr >= 0`` on every lane, given axis lower bounds ``lb``.

    Sound only for affine exprs with nonneg coefficients on axis vars; anything
    else (TMP/indirect symbols, negative coefficients, non-atomic terms,
    symbols outside the axis set) fails — the caller keeps the clamp.
    """
    try:
        coeffs = sympy.expand(expr).as_coefficients_dict()
    except (AttributeError, TypeError, ValueError):
        return False
    total = 0
    for term, c in coeffs.items():
        if term == 1:
            total += int(c)
            continue
        if not term.is_Symbol:
            return False
        if symbol_is_type(term, SymT.TMP):
            return False
        if str(term) not in axis_names:
            return False
        if c < 0:
            return False
        total += int(c) * lb.get(str(term), 0)
    return total >= 0


def _masked_index_needs_nonnegative_clamp(indexing, kernel=None):
    # See config.nonnegative_clamp_mode ("provable" | "on" | "log" | "off")
    # and config.nonnegative_clamp_log_path — the config module is the single
    # source of truth (no env-var layer, per its own contract); runtime
    # control is config.patch(nonnegative_clamp_mode=...).
    # The clamp is a defensive wrap (95c9094da7) for masked lanes whose index
    # can go negative; those lanes are never dereferenced, and the non-affine
    # maximum() demotes the whole load to scalar accesses on triton-ascend
    # (~88x on slice-remap loads, e.g. yolov3 fused_sigmoid). "provable" skips
    # the clamp only when the effective mask provably implies index >= 0 on
    # kept lanes (mathematical no-op); "off" skips always.
    mode = ncfg.nonnegative_clamp_mode
    if mode == "off":
        return False
    if not isinstance(indexing, IndexingOptions) or not indexing.has_tmpmask():
        return False
    try:
        expr = sympy.expand(indexing.index)
        origin = sympy.expand(
            expr.subs({symbol: sympy.Integer(0) for symbol in expr.free_symbols})
        )
        needs = bool(origin.is_number and origin.is_negative)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False
    if not needs:
        return False
    if mode == "provable" and kernel is not None:
        try:
            defs = _npu_body_defs(kernel)
            # Structural axis identity: the kernel's range tree is the
            # SINGLE symbol->axis source. Missing/empty -> empty set -> no
            # symbol is an axis -> the proof declines (clamp kept). No naming
            # convention fallback exists on purpose.
            rt_nodes = getattr(kernel, "range_tree_nodes", None) or ()
            axis_names = {str(sym) for sym in rt_nodes}
            tmp_masks = [
                str(m) for m in indexing.mask_vars
                if str(m).startswith("tmp")
            ]
            lb = {}
            for name in tmp_masks:
                for var, k in _npu_mask_lower_atoms(name, defs, axis_names).items():
                    lb[var] = max(lb.get(var, k), k)
            if lb and _npu_index_provably_nonnegative(expr, lb, axis_names):
                counters["npu_nonnegative_clamp"]["provable_skip"] += 1
                return False
            counters["npu_nonnegative_clamp"]["provable_unproven"] += 1
        except Exception:  # prover must never break codegen
            counters["npu_nonnegative_clamp"]["prover_error"] += 1
    if mode == "log":
        counters["npu_nonnegative_clamp"][
            f"masks={sorted(str(m) for m in indexing.mask_vars)} index={expr}"
        ] += 1
        log_path = ncfg.nonnegative_clamp_log_path
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"{expr}\t{sorted(str(m) for m in indexing.mask_vars)}\n")
    return True
