import importlib.util
import os
import sys
import types

import numpy as np
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests,
)
from testutils import TestUtils

# dynamic_filter_algo imports `from .dynamic_filter_config import fasta_dynamic_filter as df_cfg`
# which may not be available when the native torch_npu extension isn't loaded.
# Stub the config module so we can import dynamic_filter_algo by path without
# that dependency.

# Create a stub for fasta_dynamic_filter with the same attributes as the dataclass
class _FastaDynamicFilterStub:
    r1_pct = 0.3
    base_budget = 0.35
    high_budget = 0.4
    low_budget = 0.25
    max_rounds = 2

_cfg_stub = types.ModuleType("torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config")
import logging
_cfg_stub.log = logging.getLogger("dynamic_filter_test")
_cfg_stub.fasta_dynamic_filter = _FastaDynamicFilterStub()
sys.modules["torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config"] = _cfg_stub

cfg_mod = types.ModuleType("torch_npu._inductor.config")
cfg_mod.log = logging.getLogger("dynamic_filter_test")
sys.modules["torch_npu._inductor.config"] = cfg_mod

# Provide the parent package so relative imports resolve.
for name in [
    "torch_npu._inductor",
    "torch_npu._inductor.experimental",
    "torch_npu._inductor.experimental.dynamic_filter",
]:
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg

# Load dynamic_filter_algo by path using importlib.util
_DFA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "torch_npu", "_inductor", "experimental",
    "dynamic_filter", "dynamic_filter_algo.py",
)
_spec = importlib.util.spec_from_file_location(
    "torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_algo",
    os.path.abspath(_DFA_PATH)
)

algo = importlib.util.module_from_spec(_spec)
sys.modules["torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_algo"] = algo
_spec.loader.exec_module(algo)

# Load dynamic_filter_if by path using importlib.util
_DFI_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "torch_npu", "_inductor", "experimental",
    "dynamic_filter", "dynamic_filter_if.py",
)
_spec = importlib.util.spec_from_file_location(
    "torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_if",
    os.path.abspath(_DFI_PATH)
)

_dfi = importlib.util.module_from_spec(_spec)
sys.modules["torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_if"] = _dfi
_spec.loader.exec_module(_dfi)

DynamicFilter = _dfi.DynamicFilter
FastaCheckError = _dfi.FastaCheckError


class _FakeConfig:
    def __init__(self, kwargs):
        self.kwargs = dict(kwargs)


def _unique_grid(n, rng, d):
    side = int(np.ceil(n ** (1.0 / d))) + 2
    seen = {}
    while len(seen) < n:
        seen[tuple(int(rng.randint(1, side + 1)) for _ in range(d))] = True
    return list(seen.keys())[:n], side


def _make_pool(n, rng, d=2, difficulty="easy"):
    grid, side = _unique_grid(n, rng, d)
    center = np.array([(side + 1) / 2.0] * d)

    if difficulty == "easy":
        amp, noise_sd = 0.20, 0.01
    elif difficulty == "medium":
        amp, noise_sd = 0.08, 0.05
    elif difficulty == "worst":
        amp, noise_sd = 0.015, 0.12
    else:
        raise ValueError(f"unknown difficulty {difficulty!r}")

    needle = grid[rng.randint(0, len(grid))]  # only used in worst
    norm = d * (side ** 2) / 4.0  # keep bowl term O(amp) regardless of grid size
    configs, durs = [], []
    for gp in grid:
        arr = np.array(gp, dtype=float)
        if difficulty == "worst":
            base = 1.0 + amp * float(rng.rand())  # flat plateau with small jitter
            if gp == needle:
                base = 0.90                        # the single real winner (~10% better)
        else:
            base = 1.0 + amp * float(np.sum((arr - center) ** 2)) / norm
        dur = base * (1.0 + noise_sd * float(rng.randn()))
        kw = {f"BLOCK_{i}": gp[i] for i in range(d)}
        kw["X1BLOCK_SUB"] = 512            # constant -> must be dropped as feature
        kw["multibuffer"] = bool(rng.randint(0, 2))  # skip-listed -> dropped
        configs.append(_FakeConfig(kw))
        durs.append(max(dur, 1e-3))
    return configs, np.array(durs)


def _drive(flt, durs_by_idx):
    id_to_idx = {id(c): i for i, c in enumerate(flt._all_items)}
    measured_order = []

    def bench(batch):
        out = []
        for c in batch:
            i = id_to_idx[id(c)]
            measured_order.append(i)
            out.append(float(durs_by_idx[i]))
        return out

    batch = flt.r1_configs
    durations = bench(batch)
    batch = flt.refine(durations)
    while batch:
        durations = bench(batch)
        batch = flt.refine(durations)
    return measured_order


def _regret_pct(chosen, true_best):
    return 100.0 * (chosen - true_best) / chosen


def _percentile_in_pool(value, durs):
    vals = np.sort(durs)
    return 100.0 * np.searchsorted(vals, value, side="left") / len(vals)


class TestDynamicFilter(TestUtils):
    def setUp(self):
        self.original_fastautotune = os.environ.get("FASTAUTOTUNE")
        os.environ["FASTAUTOTUNE"] = "1"

    def tearDown(self):
        if self.original_fastautotune is not None:
            os.environ["FASTAUTOTUNE"] = self.original_fastautotune
        else:
            os.environ.pop("FASTAUTOTUNE", None)

    # ---- algo: feature extraction (D=2 and D=3) ---------------------
    @parametrize("d", [2, 3])
    def test_extract_features_drops_constant_and_skip_keys(self, d):
        rng = np.random.RandomState(0)
        configs, _ = _make_pool(60, rng, d=d)
        X, names = algo.extract_features(configs)
        self.assertEqual(X.shape[0], len(configs))
        self.assertEqual(sorted(names), [f"BLOCK_{i}" for i in range(d)])
        self.assertEqual(X.shape[1], d)
        self.assertNotIn("X1BLOCK_SUB", names)
        self.assertNotIn("multibuffer", names)

    def test_extract_features_no_numeric_keys_gives_zero_D(self):
        configs = [_FakeConfig({"multibuffer": True, "compile_mode": "x"})
                   for _ in range(8)]
        X, names = algo.extract_features(configs)
        self.assertEqual(names, [])
        self.assertEqual(X.shape[1], 0)

    # ---- algo: viability --------------------------------------------
    @parametrize("D,n,expect_path", [
        (0, 50, "coverage"),
        (2, 4, "coverage"),
        (2, 60, "model"),
        (3, 60, "model"),
    ])
    def test_assess_viability(self, D, n, expect_path):
        path, mode = algo.assess_viability(n, D)
        self.assertEqual(path, expect_path)
        self.assertIn(mode, ("coverage", "linear", "quad", "full"))

    # ---- algo: surrogate predicts on a learnable pool (D=2 and D=3) -
    @parametrize("d", [2, 3])
    def test_select_mode_and_predict_fits_quadratic_bowl(self, d):
        rng = np.random.RandomState(1)
        configs, durs = _make_pool(90, rng, d=d)
        X, _ = algo.extract_features(configs)
        D = X.shape[1]
        self.assertEqual(D, d)
        xmin = X.min(axis=0); xr = X.max(axis=0) - xmin; xr[xr == 0] = 1.0
        Xn = (X - xmin) / xr
        m = sorted(rng.choice(len(configs), size=45, replace=False).tolist())
        u = [i for i in range(len(configs)) if i not in m]
        d_hat, r_sq, mode = algo.select_mode_and_predict(Xn[m], durs[m], Xn[u], D)
        self.assertEqual(d_hat.shape, (len(u),))
        self.assertFalse(np.any(np.isnan(d_hat)))
        self.assertGreaterEqual(r_sq, 0.0)
        self.assertLessEqual(r_sq, 1.0)
        rho = algo.numpy_spearman(durs[u], d_hat)
        self.assertGreater(rho, 0.5)

    # ---- if: budget routing by N-class ------------------------------
    @parametrize("n,d,expect_class", [
        (60, 2, "high"),
        (250, 2, "med"),
        (500, 2, "med"),
        (200, 4, "low"),
    ])
    def test_budget_routing_by_pool_size(self, n, d, expect_class):
        rng = np.random.RandomState(2)
        configs, _ = _make_pool(n, rng, d=d)
        flt = DynamicFilter(configs, kernel_name=f"k_{n}")
        self.assertEqual(flt._budget_class, expect_class)
        self.assertGreaterEqual(flt._total_budget, flt._r1_size)
        self.assertLessEqual(flt._total_budget, flt._N)

    # ---- if: full loop, invariants + budget cap (regimes x D) -------
    @parametrize("d", [2, 3])
    @parametrize("difficulty", ["easy", "medium", "worst"])
    @parametrize("n", [60, 130, 300])
    def test_full_loop_invariants_and_budget(self, n, difficulty, d):
        rng = np.random.RandomState(3)
        configs, durs = _make_pool(n, rng, d=d, difficulty=difficulty)
        durs_by_idx = {i: durs[i] for i in range(len(configs))}
        flt = DynamicFilter(configs, kernel_name=f"loop_{difficulty}_{n}_d{d}")
        planned = flt._total_budget
        measured = _drive(flt, durs_by_idx)

        self.assertEqual(len(measured), len(set(measured)),
                         "a config was benchmarked more than once")
        self.assertLessEqual(flt._used, planned + 1)
        self.assertEqual(len(flt._r1_indices), len(set(flt._r1_indices)))
        st = flt.stats
        self.assertEqual(st["N"], n)
        self.assertEqual(st["evals"], flt._used)
        self.assertGreater(st["savings_pct"], 0.0)
        self.assertTrue(np.isfinite(st["best_dur"]))
        self.assertIn(st["confidence"], ("low", "med", "high"))

    # ---- if: selection quality by regime (D=2 and D=3) --------------
    # thresholds grounded in a 20-seed sweep of the real DynamicFilter:
    #   easy   regret max ~1.9%   medium median ~0.15% (p90 ~2.7%)
    #   worst  regret noisy (mean ~10%) but placement always top ~3 percentile
    @parametrize("d", [2, 3])
    @parametrize("difficulty,regret_bound", [
        ("easy", 3.5),     # smooth bowl: tight bound on a single seed
        ("medium", 6.0),   # moderate noise: still within a few percent
    ])
    def test_selection_quality_easy_medium(self, difficulty, regret_bound, d):
        rng = np.random.RandomState(4)
        n = 150
        configs, durs = _make_pool(n, rng, d=d, difficulty=difficulty)
        durs_by_idx = {i: durs[i] for i in range(len(configs))}
        flt = DynamicFilter(configs, kernel_name=f"q_{difficulty}_d{d}")
        _drive(flt, durs_by_idx)
        regret = _regret_pct(flt._best_dur, float(np.min(durs)))
        self.assertLess(
            regret, regret_bound,
            f"{difficulty} d={d}: regret {regret:.1f}% exceeded {regret_bound}% "
            f"(chose {flt._best_dur:.3f} vs best {float(np.min(durs)):.3f})")

    @parametrize("d", [2, 3])
    def test_selection_quality_worst_case_placement(self, d):
        n = 150
        pcts = []
        for seed in range(5):
            rng = np.random.RandomState(40 + seed)
            configs, durs = _make_pool(n, rng, d=d, difficulty="worst")
            durs_by_idx = {i: durs[i] for i in range(len(configs))}
            flt = DynamicFilter(configs, kernel_name=f"worst_{seed}_d{d}")
            _drive(flt, durs_by_idx)
            self.assertTrue(np.isfinite(flt._best_dur))
            pcts.append(_percentile_in_pool(flt._best_dur, durs))
        self.assertLessEqual(np.median(pcts), 25.0,
                             f"worst-case d={d} median placement "
                             f"{np.median(pcts):.1f} pctile worse than top-25%")
        self.assertLessEqual(max(pcts), 50.0,
                             f"worst-case d={d} picked below pool median "
                             f"(pctiles={pcts})")

    # ---- if: input-contract guards ----------------------------------
    def test_empty_configs_raises(self):
        with self.assertRaises(FastaCheckError):
            DynamicFilter([], kernel_name="empty")

    def test_refine_wrong_length_raises(self):
        rng = np.random.RandomState(5)
        configs, _ = _make_pool(60, rng, d=2)
        flt = DynamicFilter(configs, kernel_name="badlen")
        _ = flt.r1_configs
        with self.assertRaises(FastaCheckError):
            flt.refine([1.0, 2.0])

    def test_refine_rejects_negative_and_nan(self):
        rng = np.random.RandomState(6)
        configs, _ = _make_pool(60, rng, d=2)
        flt = DynamicFilter(configs, kernel_name="badval")
        batch = flt.r1_configs
        bad = [1.0] * len(batch)
        bad[0] = -1.0
        with self.assertRaises(FastaCheckError):
            flt.refine(bad)

    # ---- if: D==0 coverage path -------------------------------------
    def test_zero_feature_pool_uses_coverage(self):
        configs = [_FakeConfig({"BLOCK_0": 8, "multibuffer": (i % 2 == 0)})
                   for i in range(40)]
        durs_by_idx = {i: 1.0 + 0.01 * i for i in range(len(configs))}
        flt = DynamicFilter(configs, kernel_name="cover")
        self.assertEqual(flt._D, 0)
        self.assertEqual(flt._path, "coverage")
        measured = _drive(flt, durs_by_idx)
        self.assertEqual(len(measured), len(set(measured)))
        self.assertTrue(np.isfinite(flt._best_dur))


instantiate_parametrized_tests(TestDynamicFilter)

if __name__ == "__main__":
    run_tests()
