# Owner(s): ["module: tests"]
# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Consolidated autotune tests for the formula-driven candidate generators
# ``_pw1d_formula_configs`` / ``_red_formula_configs`` (1D-pointwise + reduction
# default autotune paths; ``autotune_fallback`` switch).
#
# Two layers in ONE file:
#   * TestAutotuneFormula / TestFormulaMinimal / TestAutotuneDispatch (L0,
#     device-free) -- direct generator + dispatch calls; no torch.compile, no NPU
#     launch. Hardware probes (core count, UB size) fall back device-free.
#   * TestAutotuneE2ECapture (L1, needs NPU) -- real ``torch.compile`` under
#     ``npu_backend=triton_experimental``; the autotuner ``run`` is no-op'd so NO
#     real triton compile occurs. Each model runs in its OWN subprocess
#     (``--capture``) to sidestep inductor's in-process codecache skip-on-3rd.
#
# Run (PYTHONPATH needed for the ``testutils`` import):
#   source env.sh
#   # L0 (device-free):  ... -k "not E2ECapture" -v
#   # L1 (needs NPU):    ... ::TestAutotuneE2ECapture -v
import json
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401  (activates triton_experimental backend)

from torch.testing._internal.common_utils import (
    TestCase, run_tests, parametrize, instantiate_parametrized_tests)
from testutils import TestUtils

import torch_npu._inductor.triton_experimental.npu_triton_heuristics as _heur
from torch_npu._inductor.triton_experimental.npu_triton_heuristics import (
    _pw1d_formula_configs, _red_formula_configs,
    npu_triton_config,
    _npu_config_key, _npu_unique_configs, _npu_hash_configs,
    _NPU_PTR_ELEM_BYTES, _NPU_UB_CAPACITY_BYTES, _NPU_UB_OVERHEAD_FACTOR)
from torch_npu._inductor.triton_experimental import config as ncfg

UB = _NPU_UB_CAPACITY_BYTES  # device-free fallback (~196608 on Ascend910B4)
HOLE_X = (1, 2, 4, 8)
HOLE_R = (512, 1024, 2048, 4096)


# L0 helpers: call the pure generators directly with synthesized dicts.
def _norm(c):
    """Config object OR serialized {kwargs, num_warps, num_stages} dict -> tuple.

    Mirrors the backend identity key (_npu_config_key): kwargs + num_warps +
    num_stages + the backend extra_options slot, so auto_blockify variants of
    one tile shape count as distinct configs."""
    is_obj = hasattr(c, "kwargs")
    kw = c.kwargs if is_obj else c["kwargs"]
    nw = c.num_warps if hasattr(c, "num_warps") else c["num_warps"]
    ns = c.num_stages if hasattr(c, "num_stages") else c["num_stages"]
    extra = (getattr(c, "extra_options", None) if is_obj else c.get("extra_options")) or {}
    return (
        tuple(sorted(kw.items()))
        + (("num_warps", nw), ("num_stages", ns))
        + tuple(sorted(extra.items()))
    )


def _norm_list(cfgs):
    return [_norm(c) for c in cfgs]


def _is_pow2(v):
    return v >= 1 and (v & (v - 1)) == 0


def pw1d(numel, dtype, num_load, signature=None):
    """1D-pointwise configs; ``signature`` overrides the default {in0: *dtype}."""
    sig = {"in0": f"*{dtype}"} if signature is None else signature
    return _pw1d_formula_configs({"x": numel}, {"signature": sig}, {"num_load": num_load}, 0)


def pw1d_xblocks(numel, dtype, num_load, signature=None):
    return [c.kwargs["XBLOCK"] for c in pw1d(numel, dtype, num_load, signature)]


def red(xnumel, rnumel, dtype, num_load, num_reduction, signature=None):
    sig = {"in0": f"*{dtype}"} if signature is None else signature
    return _red_formula_configs({"x": xnumel, "r0_": rnumel}, {"signature": sig},
                                {"num_load": num_load, "num_reduction": num_reduction})


def red_pairs(xnumel, rnumel, dtype, num_load, num_reduction, signature=None):
    return [(c.kwargs["XBLOCK"], c.kwargs["R0_BLOCK"])
            for c in red(xnumel, rnumel, dtype, num_load, num_reduction, signature)]


def _dt(dtype):
    return _NPU_PTR_ELEM_BYTES.get(dtype, 4)


def _pw1d_hi(numel, dtype, num_load):
    return min(UB // (_dt(dtype) * (num_load + 1) * 2), int(numel))


def _pw1d_align(dtype):
    return max(1, 32 // _dt(dtype))


def _ptr_widths(signature):
    """Element-byte widths of all '*'-pointer args in a signature dict."""
    return [_NPU_PTR_ELEM_BYTES.get(v[1:], 4) for v in (signature or {}).values()
            if isinstance(v, str) and v.startswith("*")]


def _pw1d_dt(signature):
    """ceil(mean) pointer element width -- the dt ``_pw1d_formula_configs`` picks."""
    ws = _ptr_widths(signature)
    return (sum(ws) + len(ws) - 1) // len(ws) if ws else 4


# L0: device-free functional unit tests for the formula generators.
class TestAutotuneFormula(TestCase):
    # Canonical 1D-pointwise golden lists (numel=1e6, fp32, num_load=2); UB-cap
    # binds (pred=0.74*ub_cap). UB is patched per case -> chip-agnostic (passes
    # device-free / Ascend910B / Ascend950).
    _EXACT_REPORT_CASES = [
        (196608, [3032, 6064, 8192]),    # Ascend910B4, UB = 192 KiB
        (262144, [4048, 8088, 10920]),   # Ascend950,   UB = 256 KiB
    ]

    @parametrize("case", _EXACT_REPORT_CASES)
    def test_exact_report_example(self, case):
        ub_bytes, expected = case
        saved = _heur._NPU_UB_CAPACITY_BYTES
        _heur._NPU_UB_CAPACITY_BYTES = ub_bytes
        try:
            self.assertEqual(pw1d_xblocks(1_000_000, "fp32", 2), expected)
        finally:
            _heur._NPU_UB_CAPACITY_BYTES = saved

    def test_dtype_drives_alignment(self):
        # fp32/bf16/i1: candidates 32B-aligned and within [1, hi].
        for numel, dtype, nl in [(1_000_000, "fp32", 2), (1_000_000, "bf16", 2),
                                 (1_000_000, "i1", 2), (500_000, "fp16", 3)]:
            with self.subTest(numel=numel, dtype=dtype, nl=nl):
                cands = pw1d_xblocks(numel, dtype, nl)
                align, hi = _pw1d_align(dtype), _pw1d_hi(numel, dtype, nl)
                self.assertTrue(all(v % align == 0 for v in cands))
                self.assertTrue(all(1 <= v <= hi for v in cands))

    def test_ub_cap_bound(self):
        # large num_load -> small hi; every candidate <= hi, cap present.
        hi = _pw1d_hi(1_000_000, "fp32", 26)
        cands = pw1d_xblocks(1_000_000, "fp32", 26)
        self.assertTrue(all(v <= hi for v in cands))
        self.assertEqual(cands[-1], (hi // _pw1d_align("fp32")) * _pw1d_align("fp32"))

    def test_num_warps_and_stages(self):
        for c in pw1d(1_000_000, "fp32", 2):
            self.assertEqual(c.num_warps, 8)
            self.assertEqual(c.num_stages, 1)

    def test_huge_kernel_auto_blockify_on(self):
        # all_blocks_parallel=True (default): huge numel appends auto_blockify_size
        # {2,4,8} on the cap XBLOCK. The value rides the backend Config.extra_options
        # slot (NOT cfg.kwargs, whose keys must all be kernel-signature constexprs).
        self.assertTrue(ncfg.all_blocks_parallel)
        cfgs = pw1d(2_000_000_000, "fp32", 1)
        blockify = sorted(c.extra_options["auto_blockify_size"]
                          for c in cfgs
                          if getattr(c, "extra_options", None))
        self.assertEqual(blockify, [2, 4, 8])
        cap_xb = max(c.kwargs["XBLOCK"] for c in cfgs)
        self.assertTrue(all(c.kwargs["XBLOCK"] == cap_xb
                            for c in cfgs if getattr(c, "extra_options", None)))
        # The slot separation is the contract: no backend option may leak into
        # kwargs (constants channel).
        self.assertFalse(any("auto_blockify_size" in c.kwargs for c in cfgs))

    def test_numel_one(self):
        # numel < align: bracket floors at 1, cap skipped -> single [1].
        self.assertEqual(pw1d_xblocks(1, "fp32", 1), [1])

    def test_hole_grid(self):
        # rnumel>=50000 -> {1,2,4,8} x {512,1024,2048,4096} (16 pairs).
        self.assertEqual(set(red_pairs(4096, 50176, "fp32", 6, 2)),
                         {(X, R) for X in HOLE_X for R in HOLE_R})

    def test_small_rset(self):
        # rnumel<=64 -> R0 in {rnumel, next_pow2(rnumel)}; X ladder.
        pairs = red_pairs(8192, 49, "fp32", 2, 1)
        self.assertEqual({r for _, r in pairs}, {49, 64})
        self.assertEqual(sorted({x for x, _ in pairs}), [1, 16, 32, 64, 128, 256])

    def test_mid_anchor(self):
        # hand-verified 27-pair set for (2048, 768, fp32, 6, 2).
        self.assertEqual(red_pairs(2048, 768, "fp32", 6, 2), [
            (1, 64), (1, 128), (1, 256), (1, 768), (2, 768), (4, 768), (8, 768),
            (16, 64), (16, 128), (16, 256), (16, 768), (32, 16), (32, 32), (32, 64),
            (32, 128), (32, 256), (32, 768), (64, 64), (64, 128), (64, 256), (64, 768),
            (128, 768), (2048, 2), (2048, 8), (2048, 64), (2048, 128), (2048, 256),
        ])

    def test_boundary_mid_to_hole(self):
        # the 50000 boundary flips MID output into the HOLE grid.
        hole_set = {(X, R) for X in HOLE_X for R in HOLE_R}
        self.assertTrue(set(red_pairs(128, 50000, "fp32", 2, 1)) <= hole_set)
        self.assertFalse(set(red_pairs(128, 49999, "fp32", 2, 1)) <= hole_set)

    def test_xnumel_capped_at_4096(self):
        # x_cap = p2floor(min(xnumel,4096)) -> no XBLOCK > 4096 even for huge xnumel.
        pairs = red_pairs(65536, 768, "fp32", 6, 2)
        self.assertTrue(all(X <= 4096 for X, _ in pairs))
        self.assertIn((4096, 2), pairs)  # hugex anchor at the capped x_cap

    def test_min_ptr_elem_bytes_overrides_dtype(self):
        # mixed-pointer signature forces min d=2, ignoring the nominal dtype arg:
        # fp32-sig-with-bf16-operand == pure bf16.
        self.assertEqual(
            set(red_pairs(2048, 768, "fp32", 6, 2,
                          signature={"in0": "*fp32", "in1": "*bf16"})),
            set(red_pairs(2048, 768, "bf16", 6, 2)))

    # mixed-dtype reduction signatures (ceil-mean ub_cap). num_load+num_reduction+1
    # == #pointers (real reduction: input ptrs + reduction ptrs + 1 output).
    RED_MIXED_SIG_CASES = [
        ({"in0": "*fp16", "in1": "*fp16", "in2": "*fp16", "out_ptr0": "*fp32"}, 3, 0),
        ({"in0": "*bf16", "in1": "*bf16", "out_ptr0": "*fp32"}, 2, 0),
    ]

    @parametrize("case", RED_MIXED_SIG_CASES)
    def test_red_mixed_dtype_respects_true_ub(self, case):
        # MID regime (64<rnumel<50000): every R0 <= TRUE physical UB cap
        # UB//(Σwidths*overhead). ceil(mean)*buffers >= Σwidths -> the formula's
        # ub_cap stays conservative; old min-based d under-counted -> overshoot.
        sig, nl, nr = case
        true_ub_cap = _NPU_UB_CAPACITY_BYTES / (sum(_ptr_widths(sig)) * _NPU_UB_OVERHEAD_FACTOR)
        pairs = red_pairs(2048, 768, "fp32", nl, nr, signature=sig)
        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(all(R <= true_ub_cap for _, R in pairs),
                        f"R overshoot true_ub_cap={true_ub_cap}: {pairs}")

    def test_red_mixed_dtype_old_min_overshoots(self):
        # Sanity: OLD min-based ub_cap WOULD over-state capacity on fp16*3+fp32.
        sig = {"in0": "*fp16", "in1": "*fp16", "in2": "*fp16", "out_ptr0": "*fp32"}
        buffers = 3 + 0 + 1
        old_ub = _NPU_UB_CAPACITY_BYTES / (min(_ptr_widths(sig)) * buffers * _NPU_UB_OVERHEAD_FACTOR)
        true_ub = _NPU_UB_CAPACITY_BYTES / (sum(_ptr_widths(sig)) * _NPU_UB_OVERHEAD_FACTOR)
        self.assertGreater(old_ub, true_ub)

    def test_red_homogeneous_signature_no_regression(self):
        # Homogeneous: ceil(mean)==min==4 -> unchanged vs single-pointer default.
        homo = {"in0": "*fp32", "in1": "*fp32", "out_ptr0": "*fp32"}
        self.assertEqual(set(red_pairs(2048, 768, "fp32", 2, 1, signature=homo)),
                         set(red_pairs(2048, 768, "fp32", 2, 1)))

    # invariants over a scenario grid (union of the deep-regime + broad-contract
    # pointwise grids; the broad-contract cases all satisfy the stricter asserts).
    PW1D_CASES = [
        (1, "fp32", 1), (7, "fp32", 1), (8, "bf16", 1), (64, "bf16", 2),
        (4096, "fp16", 3), (1_000_000, "fp32", 2), (1_000_000, "bf16", 6),
        (1_000_000, "fp8", 6), (1_000_000_000, "fp16", 3),
        (2_000_000_000, "fp32", 1), (2_000_000_000, "fp16", 2),
    ]

    @parametrize("case", PW1D_CASES)
    def test_pw1d_invariants(self, case):
        numel, dtype, nl = case
        cfgs = pw1d(numel, dtype, nl)
        align, hi = _pw1d_align(dtype), _pw1d_hi(numel, dtype, nl)
        # dedup at config level: auto_blockify variants share the cap XBLOCK but
        # differ in auto_blockify_size.
        norms = _norm_list(cfgs)
        self.assertEqual(len(norms), len(set(norms)), f"dup configs: {norms}")
        for c in cfgs:
            self.assertEqual(c.num_warps, 8)
            self.assertEqual(c.num_stages, 1)
        cands = [c.kwargs["XBLOCK"] for c in cfgs]
        self.assertGreaterEqual(len(cands), 1)
        self.assertLessEqual(len(cands), 8)  # <= 4 bracket + cap + 3 auto_blockify
        # 32B-align + [1,hi] bounds + cap-present hold when hi >= align; tiny
        # kernels (numel < align) use the max(1,..) floor instead.
        if hi >= align:
            self.assertTrue(all(v % align == 0 for v in cands), f"not {align}-aligned: {cands}")
            self.assertTrue(all(1 <= v <= hi for v in cands), f"out of [1,{hi}]: {cands}")
            cap = (hi // align) * align
            self.assertTrue(any(c.kwargs["XBLOCK"] == cap for c in cfgs), f"cap {cap} missing: {cands}")

    # mixed-dtype pointer signatures (ceil-mean dt); num_load = #pointers-1.
    MIXED_SIG_CASES = [
        ({"in0": "*fp16", "in1": "*fp16", "in2": "*fp16", "out_ptr0": "*fp32"}, 3),
        ({"in0": "*bf16", "in1": "*bf16", "out_ptr0": "*fp32"}, 2),
        ({"in0": "*fp16", "in1": "*bf16", "out_ptr0": "*fp32"}, 2),
    ]

    @parametrize("case", MIXED_SIG_CASES)
    def test_pw1d_mixed_dtype_respects_true_ub(self, case):
        # Every XBLOCK <= TRUE physical UB cap UB//(Σwidths*2). The old first-pointer
        # rule could overshoot (fp16 in0 -> dt=2 -> oversized candidates that all
        # fail in _precompile_config); dt=ceil(mean) keeps the generator's hi <= cap.
        sig, nl = case
        true_cap = UB // (sum(_ptr_widths(sig)) * 2)
        self.assertLessEqual(min(UB // (_pw1d_dt(sig) * (nl + 1) * 2), 1_000_000), true_cap)
        cands = pw1d_xblocks(1_000_000, "fp32", nl, signature=sig)
        self.assertGreaterEqual(len(cands), 1)
        self.assertTrue(all(1 <= v <= true_cap for v in cands), f"overshoot true_cap={true_cap}: {cands}")

    def test_pw1d_mixed_dtype_old_first_pointer_overshoots(self):
        # Sanity: OLD first-pointer rule WOULD overshoot on fp16*3+fp32.
        sig = {"in0": "*fp16", "in1": "*fp16", "in2": "*fp16", "out_ptr0": "*fp32"}
        old_hi = UB // (_NPU_PTR_ELEM_BYTES["fp16"] * (3 + 1) * 2)
        self.assertGreater(old_hi, UB // (sum(_ptr_widths(sig)) * 2))

    def test_pw1d_homogeneous_signature_no_regression(self):
        # Homogeneous: ceil(mean) == old first-pointer dt -> byte-identical to default.
        homo = {"in0": "*fp32", "in1": "*fp32", "out_ptr0": "*fp32"}
        self.assertEqual(pw1d_xblocks(1_000_000, "fp32", 2, signature=homo),
                         pw1d_xblocks(1_000_000, "fp32", 2))

    RED_CASES = [
        (4096, 50176), (4096, 64), (256, 1), (2048, 768), (4096, 49999), (65536, 768),
    ]

    @parametrize("case", RED_CASES)
    def test_red_invariants(self, case):
        xnumel, rnumel = case
        pairs = red_pairs(xnumel, rnumel, "fp32", 2, 1)
        self.assertEqual(pairs, sorted(set(pairs)), f"not sorted/dedup: {pairs}")
        self.assertGreaterEqual(len(pairs), 1, f"empty: {pairs}")
        for X, R in pairs:
            self.assertTrue(_is_pow2(X), f"X={X} not pow2: {pairs}")
            self.assertLessEqual(X, min(xnumel, 4096), f"X={X}>x_cap: {pairs}")
            self.assertGreaterEqual(R, 1)
        if rnumel >= 50000:  # HOLE
            self.assertTrue(all(X in HOLE_X and R in HOLE_R for X, R in pairs), f"non-HOLE: {pairs}")
            self.assertLessEqual(len(pairs), len(HOLE_X) * len(HOLE_R))

    @parametrize("name", ["pw1d", "red"])
    def test_generator_invariant_under_fallback(self, name):
        # Both generators are pure: a flip of autotune_fallback must leave output
        # identical (the switch selects formula-vs-legacy only at the dispatch entry
        # points, covered by TestAutotuneE2ECapture below).
        cases = {
            "pw1d": (({"x": 1_000_000}, {"signature": {"in0": "*fp32"}}, {"num_load": 2}, 0),
                     _pw1d_formula_configs),
            "red": (({"x": 2048, "r0_": 768}, {"signature": {"in0": "*fp32"}},
                     {"num_load": 6, "num_reduction": 2}), _red_formula_configs),
        }
        args, fn = cases[name]
        saved = ncfg.autotune_fallback
        try:
            ncfg.autotune_fallback = False
            off = _norm_list(fn(*args))
            ncfg.autotune_fallback = True
            on = _norm_list(fn(*args))
        finally:
            ncfg.autotune_fallback = saved
        self.assertEqual(off, on)


instantiate_parametrized_tests(TestAutotuneFormula)


# L0: auto_blockify slot + gate contract. auto_blockify_size is a triton-ascend
# compile option (NPUOptions field), NOT a kernel constexpr: it must ride the
# upstream third-party backend slot Config.extra_options (AutotuneCache.save
# serializes it at autotune_cache.py:325, read_best restores it at :674/:694/:701)
# and reach triton.compile via the options dict. If it ever lands in
# Config.kwargs, the constants merge in _precompile_config feeds it to
# ast_to_ttir and every compile of the config dies with
# "ValueError: 'auto_blockify_size' is not in list". The gate semantics are the
# pre-slot-era ones, unchanged: all_blocks_parallel=True AND
# ceildiv(numel, cap_XBLOCK) > 65535 -> append {2,4,8} on the cap tile.
class TestAutoBlockify(TestCase):

    HUGE = 2_000_000_000  # fp32/num_load=1: cap 12288, grid 162760 >> 65535

    def _ab_cfgs(self, numel, dtype="fp32", num_load=1):
        cfgs = pw1d(numel, dtype, num_load)
        with_ab = [c for c in cfgs if getattr(c, "extra_options", None)]
        without_ab = [c for c in cfgs if getattr(c, "extra_options", None) is None]
        return cfgs, with_ab, without_ab

    def test_slot_placement_when_on(self):
        # ON case: {2,4,8} ride extra_options, exactly one key per config,
        # kwargs stay pure constexprs, and only the cap tile carries the slot.
        cfgs, with_ab, without_ab = self._ab_cfgs(self.HUGE)
        self.assertEqual(
            sorted(c.extra_options["auto_blockify_size"] for c in with_ab), [2, 4, 8])
        for c in with_ab:
            self.assertEqual(set(c.extra_options), {"auto_blockify_size"})
        self.assertFalse(any("auto_blockify_size" in c.kwargs for c in cfgs))
        self.assertEqual(len(with_ab) + len(without_ab), len(cfgs))
        cap = max(c.kwargs["XBLOCK"] for c in cfgs)
        self.assertTrue(all(c.kwargs["XBLOCK"] == cap for c in with_ab))

    def test_gate_off_below_grid_threshold(self):
        # ceildiv(numel, cap) == 65535 exactly: OFF (the gate is strictly >).
        align = _pw1d_align("fp32")
        hi = _pw1d_hi(self.HUGE, "fp32", 1)
        cap = (hi // align) * align
        _, with_ab, _ = self._ab_cfgs(65535 * cap)
        self.assertEqual(with_ab, [])

    def test_gate_on_just_above_grid_threshold(self):
        # One block over the 65535 coreDim limit: ON -- the boundary the
        # mobilevit_s bs=128 crash kernel sits exactly on (grid 65536).
        align = _pw1d_align("fp32")
        hi = _pw1d_hi(self.HUGE, "fp32", 1)
        cap = (hi // align) * align
        _, with_ab, _ = self._ab_cfgs(65536 * cap)
        self.assertEqual(
            sorted(c.extra_options["auto_blockify_size"] for c in with_ab), [2, 4, 8])

    def test_gate_small_kernel_off(self):
        # grid far below 65535: never generates backend-option candidates.
        _, with_ab, _ = self._ab_cfgs(1_000_000, "fp32", 2)
        self.assertEqual(with_ab, [])

    def test_gate_all_blocks_parallel_off(self):
        # The feature switch (config all_blocks_parallel, default True) kills
        # candidate generation even for a huge grid.
        saved = ncfg.all_blocks_parallel
        try:
            ncfg.all_blocks_parallel = False
            _, with_ab, _ = self._ab_cfgs(self.HUGE)
            self.assertEqual(with_ab, [])
        finally:
            ncfg.all_blocks_parallel = saved

    def test_identity_distinguishes_backend_options(self):
        # Upstream's dedup/hash key (kwargs+num_warps+num_stages, see
        # runtime_utils.triton_config_to_hashable / triton_heuristics.hash_configs)
        # cannot see extra_options: the {2,4,8} candidates share one tile shape
        # and would collapse into one. _npu_config_key/_npu_unique_configs/
        # _npu_hash_configs (used by cached_autotune) must keep them distinct.
        size_hints = {"x": self.HUGE}
        plain = npu_triton_config(size_hints, 4096)
        ab = [npu_triton_config(size_hints, 4096, auto_blockify_size=v)
              for v in (2, 4, 8)]
        keys = [_npu_config_key(c) for c in [plain] + ab]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(_npu_unique_configs([plain] + ab)), 4)
        self.assertEqual(len(_npu_unique_configs([plain, plain])), 1)
        self.assertNotEqual(_npu_hash_configs([plain]), _npu_hash_configs([ab[0]]))

    def test_norm_roundtrip_carries_extra_options(self):
        # The serialized config dict (the shape AutotuneCache.save writes /
        # read_best restores) and the live Config must normalize to the same
        # identity -- losing extra_options must change the identity.
        size_hints = {"x": self.HUGE}
        cfg = npu_triton_config(size_hints, 4096, auto_blockify_size=4)
        d = {
            "kwargs": dict(cfg.kwargs),
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "extra_options": dict(cfg.extra_options),
        }
        self.assertEqual(_norm(cfg), _norm(d))
        d.pop("extra_options")
        self.assertNotEqual(_norm(cfg), _norm(d))


# L0: broad-grid minimal contract for the REDUCTION generator only (varied dtype /
# num_load / num_reduction that the fixed-param invariant tests above do not cover)
# -- non-empty, well-formed, positive, in-bounds, num_stages==1.
class TestFormulaMinimal(TestCase):
    RED_MIN_CASES = [
        (1, 1, "fp32", 1, 1), (8, 49, "fp32", 2, 1), (4096, 768, "fp32", 6, 2),
        (8192, 65536, "fp32", 2, 1), (65536, 768, "bf16", 6, 2),
        (1 << 20, 64, "fp16", 2, 1), (16384, 4096, "fp32", 6, 2),
    ]

    @parametrize("case", RED_MIN_CASES)
    def test_red_minimal_contract(self, case):
        xnumel, rnumel, dtype, num_load, num_reduction = case
        cfgs = _red_formula_configs(
            {"x": xnumel, "r0_": rnumel}, {"signature": {"in0": f"*{dtype}"}},
            {"num_load": num_load, "num_reduction": num_reduction})
        self.assertIsInstance(cfgs, list)
        self.assertGreaterEqual(len(cfgs), 1, f"empty for {case}")
        for c in cfgs:
            x, r = c.kwargs["XBLOCK"], c.kwargs["R0_BLOCK"]
            self.assertGreaterEqual(x, 1, f"X<1 for {case}")
            self.assertGreaterEqual(r, 1, f"R<1 for {case}")
            self.assertLessEqual(x, xnumel, f"X>{xnumel} for {case}")
            self.assertEqual(c.num_stages, 1, f"num_stages!=1 for {case}")
            self.assertGreaterEqual(c.num_warps, 1)


instantiate_parametrized_tests(TestFormulaMinimal)


# L0: pointwise()/reduction() dispatch entry points (autotune_enhance / max_autotune
# / autotune_fallback control flow). Captures the configs each entry point feeds to
# cached_autotune (autotuner run no-op'd -> no compile/launch).
def _capture_dispatch(fn, *args, **kwargs):
    """Call a dispatch entry point in-process and return the configs it feeds to
    cached_autotune, without compiling. Caller sets the live-read flags first."""
    captured = []

    def _cap(size_hints, configs, triton_meta=None, heuristic_type=None,
             filename=None, inductor_meta=None, custom_kernel=False, **_kw):
        captured.extend(configs)
        return None  # dispatch's `return cached_autotune(...)` value is unused

    saved_ca, saved_run = _heur.cached_autotune, _heur.NPUCachingAutotuner.run
    _heur.cached_autotune = _cap
    _heur.NPUCachingAutotuner.run = lambda self, *a, **k: None
    try:
        fn(*args, **kwargs)
    finally:
        _heur.cached_autotune = saved_ca
        _heur.NPUCachingAutotuner.run = saved_run
    return captured


class TestAutotuneDispatch(TestCase):
    @parametrize("entry", ["reduction", "pointwise"])
    def test_dispatch_max_autotune_widens(self, entry):
        # default (flag off) -> formula configs; flag on -> wider grid. Regression:
        # the formula early-return used to shadow max_autotune[_pointwise] under the
        # default config.
        from torch._inductor import config as _tcfg
        if entry == "reduction":
            fn, flag, sh = _heur.reduction, "max_autotune", {"x": 2048, "r0_": 768}
            tm = {"signature": {"in0": "*fp32", "out_ptr0": "*fp32"}}
            im = {"num_load": 1, "num_reduction": 0}
            formula = _norm_list(_red_formula_configs(sh, tm, im))
        else:
            fn, flag, sh = _heur.pointwise, "max_autotune_pointwise", {"x": 1_000_000}
            tm = {"signature": {"in0": "*fp32", "out_ptr0": "*fp32"}, "device": None}
            im = {"num_load": 1}
            formula = _norm_list(_pw1d_formula_configs(sh, tm, im, 0))
        saved = (_tcfg.max_autotune, _tcfg.max_autotune_pointwise)
        _tcfg.max_autotune = _tcfg.max_autotune_pointwise = False
        try:
            off = _norm_list(_capture_dispatch(fn, sh, triton_meta=tm, inductor_meta=im))
            setattr(_tcfg, flag, True)
            on = _norm_list(_capture_dispatch(fn, sh, triton_meta=tm, inductor_meta=im))
        finally:
            _tcfg.max_autotune, _tcfg.max_autotune_pointwise = saved
        self.assertEqual(off, formula)
        self.assertGreater(len(on), len(off))
        self.assertNotEqual(on, formula)


instantiate_parametrized_tests(TestAutotuneDispatch)


# L1: end-to-end config capture (needs NPU; no real triton compile). This file
# doubles as the capture subprocess: invoked as ``python <this_file> --capture
# '<json>'`` it runs ONE model, hooks cached_autotune to capture the candidate
# list, no-ops the autotuner run (no compile/launch), and prints one CAPTURE_JSON
# line. A fresh process sidesteps inductor's in-process codecache skip-on-3rd.
_RUNNER = os.path.abspath(__file__)
_DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
# name -> (callable, [shapes], [dtypes]); one fp32 + one bf16 pointwise, one model
# per reduction regime (HOLE / MID / SMALL).
E2E_MODELS = {
    "pw_relu": (lambda x: torch.relu(x), [(1 << 20,)], ["float32"]),
    "pw_add": (lambda a, b: a + b, [(1 << 16,), (1 << 16,)], ["float32", "float32"]),
    "pw_mul_bf16": (lambda a, b: a * b, [(1 << 24,), (1 << 24,)], ["bfloat16", "bfloat16"]),
    "red_sum_hole": (lambda x: x.sum(-1), [(8192, 65536)], ["float32"]),
    "red_sum_mid": (lambda x: x.sum(-1), [(4096, 1024)], ["float32"]),
    "red_sum_small": (lambda x: x.sum(-1), [(1 << 20, 64)], ["float32"]),
}


def _gen_pw(size_hints, signature, num_load):
    return sorted(_norm(c) for c in
                  _pw1d_formula_configs(size_hints, {"signature": signature}, {"num_load": num_load}, 0))


def _gen_red(size_hints, signature, num_load, num_reduction):
    return sorted(_norm(c) for c in _red_formula_configs(
        size_hints, {"signature": signature}, {"num_load": num_load, "num_reduction": num_reduction}))


def _gen(c):
    """Expected formula configs for a captured entry (dispatches on heuristic type)."""
    if c["type"] == "POINTWISE":
        return _gen_pw(c["size_hints"], c["signature"], c["num_load"])
    return _gen_red(c["size_hints"], c["signature"], c["num_load"], c["num_reduction"])


def _pick(caps, htype, hint_len):
    return [c for c in caps if c["type"] == htype and len(c["size_hints"]) == hint_len]


def _run_capture(spec):
    """Run the capture subprocess for one model; return the captured entries."""
    proc = subprocess.run(
        [sys.executable, _RUNNER, "--capture", json.dumps(spec)],
        capture_output=True, text=True, env=dict(os.environ), timeout=420)
    for line in proc.stdout.splitlines():
        if line.startswith("CAPTURE_JSON "):
            return json.loads(line[len("CAPTURE_JSON "):])["captures"]
    raise AssertionError(
        f"no CAPTURE_JSON from subprocess (model={spec}).\n"
        f"returncode={proc.returncode}\nstderr tail:\n{proc.stderr[-2000:]}")


class TestAutotuneE2ECapture(TestUtils):
    # captured configs (flags off) must equal the formula generator output.
    _MATCH_CASES = [
        ("pw_relu", "POINTWISE", 1), ("pw_add", "POINTWISE", 1), ("pw_mul_bf16", "POINTWISE", 1),
        ("red_sum_hole", "REDUCTION", 2), ("red_sum_mid", "REDUCTION", 2), ("red_sum_small", "REDUCTION", 2),
    ]

    @parametrize("model,htype,hint_len", _MATCH_CASES)
    def test_matches_generator(self, model, htype, hint_len):
        caps = _run_capture({"model": model, "flags": "off"})
        got = _pick(caps, htype, hint_len)
        self.assertTrue(
            got, f"no {htype}(len=={hint_len}) kernel captured; got {[(c['type'], c['size_hints']) for c in caps]}")
        for c in got:
            self.assertEqual(sorted(_norm(x) for x in c["configs"]), _gen(c), f"mismatch {c['size_hints']}")

    # off (default) == formula generator; turning any of these flags on widens the
    # search beyond the formula set. (autotune_fallback = legacy exhaustive sweep;
    # max_autotune[_pointwise] = exhaustive grid.) Regression: the formula early-
    # return used to shadow max_autotune[_pointwise] under the default config.
    _WIDEN_CASES = [
        # (model, flag, heuristic_type, size_hints_len)
        ("pw_add", "autotune_fallback", "POINTWISE", 1),
        ("pw_add", "max_autotune_pointwise", "POINTWISE", 1),
        ("red_sum_mid", "max_autotune", "REDUCTION", 2),
    ]

    @parametrize("model,flag,htype,hint_len", _WIDEN_CASES)
    def test_flag_widens_beyond_formula(self, model, flag, htype, hint_len):
        off = _run_capture({"model": model, "flags": "off"})
        on = _run_capture({"model": model, "flags": flag})
        o = _pick(off, htype, hint_len)[0]
        n = _pick(on, htype, hint_len)[0]
        self.assertEqual(sorted(_norm(x) for x in o["configs"]), _gen(o))
        self.assertGreater(len(n["configs"]), len(o["configs"]))
        self.assertNotEqual(sorted(_norm(x) for x in n["configs"]), _gen(n))


instantiate_parametrized_tests(TestAutotuneE2ECapture)


# subprocess runner role (--capture <json>). Imports kept local so the parent test
# process (which never captures) does not pay for / depend on them.
def _capture_main(spec_json):
    import torch_npu._inductor.triton_experimental.config as tcfg
    import torch_npu._inductor.triton_experimental.npu_triton_heuristics as H
    from torch import _inductor as ti

    cap = []

    def _cap(size_hints, configs, triton_meta, heuristic_type, filename=None,
             inductor_meta=None, custom_kernel=False):
        im = dict(inductor_meta or {})
        cap.append({
            "type": heuristic_type.name,
            "size_hints": dict(size_hints),
            "signature": dict((triton_meta or {}).get("signature", {})),
            "num_load": im.get("num_load", 1),
            "num_reduction": im.get("num_reduction", 0),
            "configs": [{"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages}
                        for c in configs],
        })
        # delegate so the generated wrapper gets a real autotuner object; its run
        # is no-op'd below so no compile/launch happens.
        return _real_cached_autotune(
            size_hints, configs, triton_meta, heuristic_type,
            filename=filename, inductor_meta=inductor_meta, custom_kernel=custom_kernel)

    _real_cached_autotune = H.cached_autotune
    H.cached_autotune = _cap
    H.NPUCachingAutotuner.run = lambda self, *a, **k: None  # short-circuit: no compile/launch

    spec = json.loads(spec_json)
    fn, shapes, dtypes = E2E_MODELS[spec["model"]]
    ti.config.max_autotune = spec["flags"] == "max_autotune"
    ti.config.max_autotune_pointwise = spec["flags"] == "max_autotune_pointwise"
    tcfg.autotune_fallback = spec["flags"] == "autotune_fallback"
    args = [torch.randn(tuple(s), dtype=_DT[d], device="npu") for s, d in zip(shapes, dtypes)]
    try:
        torch.compile(fn, options={"npu_backend": "triton_experimental"})(*args)
    except Exception:
        pass  # capture happens at codegen time; we only need `cap`.
    print("CAPTURE_JSON " + json.dumps({"captures": cap}))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--capture":
        _capture_main(sys.argv[2])
    else:
        run_tests()
