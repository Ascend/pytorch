# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Clone-fold contract judgment core -- the single place the rescue physics
lives.  Both defense lines import only this module, so the feed points
cannot drift:

  Line 1 (lowering)  fold_verdict -- tile-INDEPENDENT fold/materialize
      decision from the clone source layout: trailing dense run (Family T)
      or rescue bet (Family R).
  Line 2 (codegen)   judge_load + tile_chain -- TILE-DEPENDENT per-load
      verdict under the real or simulated greedy chain, to enforce (axis
      promotion) and verify Line 1's fold bets.

SUBSTRATE: every comparison in the tree is a substrate predicate -- ONE code
path, no static/symbolic fork (the same shape as upstream
ir.is_contiguous_strides_for_shape).  IntSubstrate (default) is identity int
semantics, so on concrete ints the tree is byte-identical to the validated
static mechanism (proven vs the frozen research twin over 35773 layouts).
SizevarsSubstrate answers the same five queries (known_eq / dead1 / gt /
threshold_ge / min_lb) with the guard-free upstream family --
statically_known_* / bound_sympy / optimization_hint; int()/guard_int (what
historically broke mark_dynamic) is never used, and no bound arithmetic is
re-derived by hand (an earlier hand-rolled layer was falsifiable at range
boundaries).  gt is TRI-STATE: None = order undecidable -> the caller
conservatively materializes (same for any unprovable gate: an unrescued
fold costs 30-130x, over-materializing 1.5-3x).  threshold_ge may take an
example-shape hint ballot under config.clone_dynamic_hint_tier --
bucket-dependent by design, both outcomes numerically correct.

AXIS-ORDER CONTRACT (load-bearing): ordered feeds to judge_permute /
judge_load are TILE-DIM ORDER, OUTERMOST FIRST (PtrAnalysis merges axes by
dimIndex).  The codegen feed must sort free_tile_nodes by divisor
DESCENDING -- a reversed order flips the identity test and misjudges both
ways.

Provenance: triton-ascend lib/TritonToStructured/PtrAnalysis.cpp:1374-1533
(analyzePermute / countContiguousAxes); docs/project/bisheng permute
recover/research/step1+step2 (~100 adversarial configs, predictor 102/102);
design doc/clone_fold_contract_e2e.md.
"""

import functools

# Per-ROW 32B padding of UB staging (measured: [256,12] lands as [256,16] in UB)
CEIL_ALIGN_B = 32
UB_BUDGET_FALLBACK_B = 196608  # PlanMemory.cpp:51-52 (192KB per core)


def ub_budget_b():
    """PlanMemory UB budget of the current device (192KB 910B-class, 256KB
    Ascend950); lazy via device_props, hardcoded 192KB fallback."""
    try:
        from .device_props import get_npu_ub_size_bytes
        return get_npu_ub_size_bytes()
    except Exception:
        return UB_BUDGET_FALLBACK_B


def ceil_align(nbytes, align=CEIL_ALIGN_B):
    return (nbytes + align - 1) // align * align


# Substrate protocol (see module docstring).

class UndecidableOrder(Exception):
    """Two strides cannot be provably ordered."""


class IntSubstrate:
    """Identity int semantics; the default substrate.  Equivalence tests
    prove default == explicit instance == frozen research twin."""

    def known_eq(self, a, b):
        return a == b

    def dead1(self, a):
        return a <= 1

    def gt(self, a, b):
        return a > b

    def threshold_ge(self, x, t):
        return x >= t

    def min_lb(self, a):
        return a if isinstance(a, int) else None


INT_SUBSTRATE = IntSubstrate()   # stateless: shared default instance


class SizevarsSubstrate:
    """Guard-free provable substrate over V.graph.sizevars: upstream
    statically_known_* / optimization_hint verbatim (no hand-rolled bound
    arithmetic -- it was falsifiable at range boundaries).  gt stays
    tri-state: None -> UndecidableOrder -> conservative materialize."""

    def __init__(self, sizevars):
        self.sv = sizevars
        self.se = getattr(sizevars, "shape_env", None)

    def known_eq(self, a, b):
        try:
            return bool(self.sv.statically_known_equals(a, b))
        except Exception:                                            # noqa: BLE001
            return False

    def dead1(self, a):
        if isinstance(a, int):
            return a <= 1
        try:
            return bool(self.sv.statically_known_leq(a, 1))
        except Exception:                                            # noqa: BLE001
            return False

    def gt(self, a, b):
        """Provable a > b -> True; provable a <= b -> False; else None."""
        try:
            if self.sv.statically_known_gt(a, b):
                return True
            if self.sv.statically_known_leq(a, b):
                return False
        except Exception:                                            # noqa: BLE001
            pass
        return None

    def min_lb(self, a):
        """Provable lower bound int, reason strings only."""
        from torch._inductor.utils import convert_symint_to_expr
        e = convert_symint_to_expr(a)
        if isinstance(e, int):
            return e
        if self.se is None:
            return None
        try:
            lo = self.se.bound_sympy(e).lower
            return int(lo)
        except Exception:                                            # noqa: BLE001
            return None

    def threshold_ge(self, x, t):
        """Provable x >= t, else the hint ballot (clone_dynamic_hint_tier):
        optimization_hint is upstream-sanctioned for non-guarding
        optimization decisions, and fold-vs-materialize is one -- verdicts
        may track the shape bucket on this tier only."""
        try:
            if self.sv.statically_known_geq(x, t):
                return True
        except Exception:                                            # noqa: BLE001
            pass
        try:
            from . import config as ncfg
            if ncfg.clone_dynamic_hint_tier:
                return int(self.sv.optimization_hint(x)) >= t
        except Exception:                                            # noqa: BLE001
            pass
        return False


def _sort_stable(items, key, substrate, *, descending):
    """Stable sort under the substrate's provable order; raises
    UndecidableOrder on incomparable pairs; == sorted(..., reverse=True)
    on total orders (ties keep slot order)."""
    def cmp(x, y):
        kx, ky = key(x), key(y)
        r = substrate.gt(kx, ky) if descending else substrate.gt(ky, kx)
        if r is None:
            raise UndecidableOrder(f"{kx} vs {ky}")
        if r:
            return -1
        r2 = substrate.gt(ky, kx) if descending else substrate.gt(kx, ky)
        if r2 is None:
            raise UndecidableOrder(f"{kx} vs {ky}")
        return 1 if r2 else 0
    return sorted(items, key=functools.cmp_to_key(cmp))


def _live(length, stride, substrate):
    """Keep rule ``L >= 2 and c != 0``; the substrate keeps an axis unless
    provably dead."""
    return not substrate.dead1(length) and not substrate.known_eq(stride, 0)


def bet_key(name, pairs, substrate=None, resolve=None):
    """Cross-line fold-bet join key: (buffer identity, LIVE (length,
    stride) multiset).  The ONE construction point shared by lowering's
    _record_fold_bet and the codegen cross-check, so the liveness rule
    cannot drift between the two ends of the join (a drifted rule
    silently un-matches the WARN).  ``pairs`` are RAW (length, stride)
    tuples; optional ``resolve`` maps each live pair to the join
    representation -- lowering passes the optimization_hint ballot so
    symbolic and hinted-int layouts land on one key (resolution failure
    keeps the raw pairs).  The multiset is permutation-invariant; the
    name stops equal layouts of distinct buffers from cross-matching."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    live = [(length, stride) for length, stride in pairs
            if _live(length, stride, substrate)]
    if resolve is not None:
        try:
            live = [resolve(length, stride) for length, stride in live]
        except Exception:                                          # noqa: BLE001
            pass
    return (name, frozenset(live))


def count_contiguous_axes(strides, lengths, inner_first, substrate=None):
    """PtrAnalysis::countContiguousAxes mirror (:1512-1533): innermost-first
    walk, expected doubles per dense axis, stop at the first mismatch;
    returns (count, run_elements)."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    expected, cnt = 1, 0
    for k in inner_first:
        if not substrate.known_eq(strides[k], expected):
            break
        cnt += 1
        expected *= lengths[k]
    return cnt, expected


def judge_permute(strides, lengths, substrate=None):
    """Step-B core (:1374-1465): will ImplicitPermute fire?  Feed is
    TILE-DIM ORDER (module contract).  Returns dict(verdict, inner_first,
    new_cnt, old_cnt, run): verdict in {rescue, contig, no-rescue:tail |
    contig | identity-tail | undecidable, degenerate} -- "undecidable"
    (Sizevars substrate only) means the caller conservatively rejects."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    live = [k for k in strides
            if _live(lengths.get(k, 1), strides[k], substrate)]
    if len(live) < 2:
        return dict(verdict="degenerate", inner_first=[], new_cnt=0,
                    old_cnt=0, run=1)

    # Stable descending sort == PtrAnalysis.cpp:1421-1431: equal strides
    # keep slot order (NO name tiebreak -- it would flip two-stride-1
    # expand layouts out of ①).
    try:
        desc = _sort_stable(live, lambda k: strides[k], substrate,
                            descending=True)
    except UndecidableOrder as why:
        return dict(verdict="no-rescue:undecidable", inner_first=[],
                    new_cnt=0, old_cnt=0, run=None, why=str(why))
    inner_first = desc[::-1]
    if inner_first == live[::-1]:
        # :1429 identity: the slot order already equals the storage order.
        if substrate.known_eq(strides[live[-1]], 1):
            return dict(verdict="contig", inner_first=inner_first,
                        new_cnt=len(live), old_cnt=len(live),
                        run=None)
        return dict(verdict="no-rescue:identity-tail", inner_first=inner_first,
                    new_cnt=0, old_cnt=0, run=None)
    if not substrate.known_eq(strides[inner_first[0]], 1):
        # :1435-1440 tail rule: the smallest-stride axis must be static 1.
        return dict(verdict="no-rescue:tail", inner_first=inner_first,
                    new_cnt=0, old_cnt=0, run=None)
    new_cnt, run = count_contiguous_axes(strides, lengths, inner_first,
                                         substrate)
    old_cnt, _ = count_contiguous_axes(strides, lengths, live[::-1], substrate)
    if new_cnt <= old_cnt:
        # :1448-1461 acceptance: provable contiguity must strictly increase.
        return dict(verdict="no-rescue:contig", inner_first=inner_first,
                    new_cnt=new_cnt, old_cnt=old_cnt, run=run)
    return dict(verdict="rescue", inner_first=inner_first,
                new_cnt=new_cnt, old_cnt=old_cnt, run=run)


def judge_load(axes, esz=4):
    """Final per-load verdict under concrete tiles (Line 2 feed): axes
    [(name, coeff, tile)] in tile-dim order.  Returns dict(verdict, health,
    mouth_B, ub_B, detail); health = landed ①/②."""
    live = [(nm, c, L) for nm, c, L in axes if _live(L, c, INT_SUBSTRATE)]
    if not live:
        return dict(verdict="scalar", health=False, mouth_B=esz, ub_B=None,
                    detail="no live axis in tile (all odometer/broadcast)")
    if len(live) == 1:
        nm, c, L = live[0]
        if c == 1:
            return dict(verdict="contig", health=True, mouth_B=L * esz,
                        ub_B=L * esz, detail=f"single dense axis {nm}")
        return dict(verdict="no-rescue", health=False, mouth_B=esz, ub_B=None,
                    detail=f"single strided axis {nm} coeff {c}")
    tile_of = {nm: L for nm, _, L in live}
    v = judge_permute({nm: c for nm, c, _ in live}, tile_of)
    if v["verdict"] == "contig":
        C = live[-1][2]
        D = 1
        for _, _, L in live[:-1]:
            D *= L
        return dict(verdict="contig", health=True, mouth_B=C * esz,
                    ub_B=D * C * esz,
                    detail=f"identity, inner {live[-1][0]} dense")
    if v["verdict"] == "rescue":
        tail = v["inner_first"][0]
        C = tile_of[tail]
        D = 1
        for nm, _, L in live:
            if nm != tail:
                D *= L
        row_b = ceil_align(C * esz)
        return dict(verdict="rescue", health=True, mouth_B=C * esz,
                    ub_B=2 * D * row_b,
                    detail=f"copy+transpose, mouth {C}x{esz}B, "
                           f"ub=2x{D}xceil32({C}x{esz})={2 * D * row_b}B")
    return dict(verdict="no-rescue", health=False, mouth_B=esz, ub_B=None,
                detail=v["verdict"])


def tile_chain(order, lengths, xb):
    """Greedy prefix-fill over ``order`` at budget ``xb``, mirroring the
    emitter line for line: remainder floors at 1, no early break, unknown
    length eats the whole remainder."""
    tiles, rem = {}, xb
    for nm in order:
        L = lengths.get(nm)
        t = max(1, rem) if L is None else min(L, max(1, rem))
        tiles[nm] = t
        rem = max(1, rem // t)
    return tiles


def trailing_run(sizes, strides, substrate=None):
    """Elements of the source's dense TRAILING segment (Family T):
    reverse-accumulate while stride == running product; size-1 skipped,
    stride-0/mismatch stops.  run == numel means fully row-major."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    run = 1
    for k in range(len(sizes) - 1, -1, -1):
        if substrate.known_eq(sizes[k], 1):
            continue
        if substrate.known_eq(strides[k], run):
            run *= sizes[k]
        else:
            break
    return run


def mouth_pad_ratio(mouth_B):
    """ceil32(mouth)/mouth padding waste of a rescue row; the bet's 8B
    floor == pad ratio <= 4x, the measured win/loss boundary."""
    return ceil_align(mouth_B) / mouth_B


def _rescue_feasible(sizes, strides, itemsize, *, max_swaps, min_axis,
                     min_mouth_b, substrate=None):
    """(True, why) iff the fold bet is feasible -- four layers, each
    mirroring a verified downstream behavior; any doubt -> (False,
    <layer reason>) -> materialize."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    n = len(sizes)
    dims = [k for k in range(n) if _live(sizes[k], strides[k], substrate)]

    # L1 -- will ImplicitPermute fire?  Whole-tensor extents: a necessary
    # condition (tile truncation can change the live set; the final verdict
    # belongs to the codegen feed).
    v = judge_permute(
        {k: strides[k] for k in dims}, {k: sizes[k] for k in dims}, substrate)
    if v["verdict"] != "rescue":
        return False, f"L1:{v['verdict']}"
    inner_first = v["inner_first"]
    new_cnt = v["new_cnt"]

    # L3 -- read-side mouth: prices the per-row 32B padding waste; 8B is
    # the measured boundary, not the conservative 32B (a clone pays a full
    # extra HBM pass, which usually costs more than the padding).
    mouth = sizes[inner_first[0]] * itemsize
    if not substrate.threshold_ge(mouth, min_mouth_b):
        if isinstance(mouth, int):
            return False, (f"L3:mouth {mouth}B < {min_mouth_b}B "
                           f"(pad x{mouth_pad_ratio(mouth):.0f})")
        return False, (f"L3:mouth {mouth}B "
                       f"(lb={substrate.min_lb(mouth)}) < {min_mouth_b}B")

    # L2 -- value-side transpose trips = misplaced - #nontrivial cycles
    # (pure permutation walk; only the storage order needs the substrate).
    try:
        storage = _sort_stable(dims, lambda k: strides[k], substrate,
                               descending=False)               # inner first
    except UndecidableOrder as why:
        return False, f"L2:undecidable-order {why}"
    walk = dims[::-1]                                        # index-reversed
    pi = dict(zip(walk, storage))
    misplaced = [w for w, s in zip(walk, storage) if w != s]
    seen, cycles = set(), 0
    for k in walk:
        j, chain_len = k, 0
        while j not in seen:
            seen.add(j)
            j = pi[j]
            chain_len += 1
        if chain_len > 1:
            cycles += 1
    n_swap = len(misplaced) - cycles
    if n_swap > max_swaps:
        return False, f"L2:n_swap={n_swap}>{max_swaps}"

    # L4 -- EVERY misplaced axis must be long; no stride-1 exemption (the
    # probe data suggested one, mobilevit_s disproved it end-to-end:
    # 0.28x through chained clone->conv consumers).  The check rides the
    # threshold channel (provable then hint ballot); min_lb stays
    # display-only, so the reason reports the provable lower bound.
    lbs = {k: substrate.min_lb(sizes[k]) for k in misplaced}
    bad = [k for k in misplaced
           if not substrate.threshold_ge(sizes[k], min_axis)]
    if bad:
        # min over failing axes == global min on the int substrate.
        shown = [lbs[k] for k in bad if lbs[k] is not None]
        return False, f"L4:misplaced-min {min(shown) if shown else '?'}"

    if isinstance(mouth, int):
        pad = f"pad=x{mouth_pad_ratio(mouth):.0f}"
    else:
        pad = f"lb={substrate.min_lb(mouth)}"
    if misplaced:
        shown = [v for v in lbs.values() if v is not None]
        return True, (f"new={new_cnt} n_swap={n_swap} mouth={mouth}B "
                      f"{pad} "
                      f"misplaced_min={min(shown) if shown else '?'}")
    return True, (f"new={new_cnt} n_swap={n_swap} mouth={mouth}B "
                  f"{pad} misplaced_min=-")


def fold_verdict(sizes, strides, itemsize, *, max_swaps=2, min_axis=8,
                 min_mouth_b=8, min_run_b=32, substrate=None):
    """Line-1 tile-independent decision.  ("fold", trailing_run...) Family T
    (the consumer's lane rides a >= min_run_b dense tail); ("fold",
    rescue_bet ...) Family R (valid under the Line-2 survival contract);
    ("materialize", <layer>) everything else.  Thresholds are
    END-TO-END-MEASURED constants -- do not tighten analytically.
    ``substrate``: the predicate substrate (default plain int)."""
    substrate = INT_SUBSTRATE if substrate is None else substrate
    run = trailing_run(sizes, strides, substrate)
    if substrate.threshold_ge(run * itemsize, min_run_b):
        return "fold", f"trailing_run run={run} ({run * itemsize}B)"
    ok, why = _rescue_feasible(
        sizes, strides, itemsize, max_swaps=max_swaps, min_axis=min_axis,
        min_mouth_b=min_mouth_b, substrate=substrate)
    if ok:
        return "fold", f"rescue_bet {why}"
    if "undecidable" in why:
        # Order unprovable (Sizevars substrate only).  Under the hint tier
        # (same knob as the threshold ballot) decide the WHOLE layout at
        # the example shape via the original int algorithm -- bucket-
        # dependent by design, never taken on the provable tier.
        try:
            from . import config as ncfg
            if ncfg.clone_dynamic_hint_tier:
                hint = substrate.sv.optimization_hint
                c_sizes = [int(hint(v)) for v in sizes]
                c_strides = [int(hint(v)) for v in strides]
                v2, why2 = fold_verdict(
                    c_sizes, c_strides, itemsize, max_swaps=max_swaps,
                    min_axis=min_axis, min_mouth_b=min_mouth_b,
                    min_run_b=min_run_b)
                return v2, f"{why2} (order undecidable; decided at example shape)"
        except Exception:                                            # noqa: BLE001
            pass
    return "materialize", why
