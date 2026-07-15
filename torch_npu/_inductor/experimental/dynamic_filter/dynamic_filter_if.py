"""
dynamic filter — adaptive tiling config selection.

thin wrapper over dynamic_filter_algo. the math lives in the algo module.

the production caller (dynamic_filter_scheduler.py) drives the batch api
directly:

  init phase (during model compilation):
      flt = DynamicFilter(all_configs, kernel_name)
      # caller compiles ALL configs via precompile_parallel()
      # flt is stored on the autotuner instance

  bench phase (first kernel.run()):
      batch = flt.r1_configs
      while batch:
          durations = bench(batch)
          batch = flt.refine(durations)
      # then read flt.stats / flt._best_idx

the self-driving loop (run(bench_fn)) lives in the offline harness
(offline_exp/run_filter.py), not here — production never calls it.

env:
  ALGO_IF_LOG=1 — verbose debug logging (default 0)
"""
import os
import time
import numpy as np
import torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_algo as algo
from torch_npu._inductor.config import log as _prod_log

_TAG = "[FASTA_DYN_FILTER_IF]"

ALGO_IF_LOG = int(os.environ.get('ALGO_IF_LOG', '0'))


def _selector_log(msg, level='debug'):
    if not ALGO_IF_LOG:
        return
    from ...config import log
    full = f'[dyn filter] {msg}'
    if level == 'info':
        log.info(full)
    elif level == 'warning':
        log.warning(full)
    else:
        log.debug(full)


# =====================================================================
# walk-through (read before changing the algorithm)
# =====================================================================
#
# PHASE I -- INIT (during model compilation, in the subprocess pool):
#   I.1  store items/kernel_name/rng/counters; assert the input contract
#   I.2  algo.extract_features(items) -> (X, feat_names); D = X.shape[1]
#   I.3  pick r1_size from R1_PCT and R1_FLOOR_PCT bounds
#   I.4  pick total_budget by routing: high (small low-D), low (large high-D),
#        high_legacy (high-D underflow), med (rest)
#   I.5  algo.stratified_sample(X, r1_size, rng) -> r1_indices  (D>0)
#        else random.choice(N, r1_size)                          (D==0)
#   I.6  algo.assess_viability(r1_size, D) -> (path, poly_mode)
#   I.7  normalize X to [0,1] per column -> Xn
#   I.8  unmeasured = set(range(N)) - set(r1_indices)
#
# PHASE II -- R1 (once, just after init):
#   II.1 r1_configs (state-advancing, idempotent) returns items[r1_indices]
#        and sets _last_batch_indices so the next refine() can record
#   II.2 caller benchmarks them, returns durations in the same order
#
# PHASE III -- REFINE LOOP (repeat until an empty list comes back):
#   III.1 record durations; update _best_dur/_best_idx; check invariants
#   III.2 if all measured or _done -> return [] (terminal)
#   III.3 dispatch:
#         path == 'coverage': _coverage_step (one shot, then done)
#         else              : _model_step  (predict + softmax pick)
#   III.4 advance _round, store _last_batch_indices, return the next batch
#
# PHASE IV -- MODEL STEP:
#   IV.1 Xm = Xn[measured], dm = durations[measured]; Xu = Xn[unmeasured]
#   IV.2 algo.select_mode_and_predict(Xm, dm, Xu, D) -> (d_hat, R^2, mode)
#   IV.3 algo.should_stop(dm, best, d_hat, prev_hist) -> bool
#   IV.4 batch_size = total_budget // MAX_ROUNDS, capped by remaining
#   IV.5 D>=MARGINAL_D: half via softmax(d_hat), half via marginal score
#        else          : all via softmax(d_hat)
#   IV.6 return uidx[merged]   (indices into _all_items)
#
# every phase boundary is guarded by _check. failures raise FastaCheckError
# carrying the kernel name and the condition. always on by design: catch the
# bug at the first crime, not the second.
# =====================================================================


class FastaCheckError(AssertionError):
    """raised when a filter input/invariant is violated."""


def _check(cond, kernel_name, fmt, *args):
    """always-on assertion. fmt uses %-format with optional args."""
    if not cond:
        msg = fmt % args if args else fmt
        raise FastaCheckError(f"[FASTA-CHECK] {kernel_name}: {msg}")


def _confidence_label(r_sq, margin, k_est):
    """confidence in the selected best, from the last round's kpis.

    the dangerous case is a sharp single winner (k_est==1) the surrogate could
    not model (low r_sq) and did not separate from the field (margin<1). the
    safe case is either many near-ties (any pick is fine) or a well-fit
    surrogate confident the winner is already in hand (margin>=1).
    """
    # many near-optimal configs -> the pick barely matters
    if k_est >= 3:
        return 'high'
    # surrogate is confident the winner is already measured
    if margin >= 1.0 and r_sq >= 0.5:
        return 'high'
    if margin >= 1.0 or r_sq >= 0.6:
        return 'med'
    # sharp single winner the model could not explain -> least trustworthy
    if k_est <= 1 and r_sq < 0.3:
        return 'low'
    return 'med'


class DynamicFilter:
    """adaptive config filter for autotuning.

    init entry (during model compilation):
        flt = DynamicFilter(configs, kernel_name)
        calls: algo.extract_features, algo.assess_viability,
               algo.stratified_sample

    batch api (driven by the production caller):
        batch = flt.r1_configs       # state-advancing single-shot
        durations = my_benchmark(batch)
        batch = flt.refine(durations) # calls algo.select_mode_and_predict/should_stop/
        ...until batch is empty...    #   compute_tau/softmax_select/marginal_scores
        # results via flt.stats
    """

    def __init__(self, items, kernel_name='unknown'):
        """init entry — feature extraction, R1 planning, budget sizing.

        runs during model compilation. after this returns, the caller compiles
        R1 via precompile_parallel.

        args:
            items: list of config objects with a .kwargs dict.
            kernel_name: str for logging.
        """
        self._t_init = time.perf_counter_ns()

        # PHASE I.1: input contract
        _check(items is not None, kernel_name, "items is None")
        _check(isinstance(kernel_name, str) and kernel_name,
               kernel_name or "<empty>",
               "kernel_name must be non-empty str, got %r", kernel_name)

        self._kernel_name = kernel_name
        self._all_items = list(items)
        self._N = len(self._all_items)

        _check(self._N > 0, kernel_name, "empty configs list")
        _bad = [i for i, c in enumerate(self._all_items) if not hasattr(c, 'kwargs')]
        _check(not _bad, kernel_name,
               "%d/%d configs lack .kwargs (first idx=%d)",
               len(_bad), self._N, _bad[0] if _bad else -1)
        # id() collisions break the strategy-side _launcher_map (keyed by id).
        _ids = [id(c) for c in self._all_items]
        _check(len(_ids) == len(set(_ids)), kernel_name,
               "duplicate config object ids in items (N=%d unique=%d)",
               self._N, len(set(_ids)))

        self._rng = np.random.RandomState()
        self._round = 0
        self._measured = {}
        self._best_idx = None
        self._best_dur = float('inf')
        self._last_batch_indices = []
        self._overhead_ns = 0
        self._prev_hist = None
        self._round_estimators = []   # estimator (basis mode) picked per refine round
        self._round_records = []      # per-round kpi dicts: round/est/r_sq/margin/k_est/batch/best
        self._last_r_sq = 0.0         # last surrogate fit quality
        self._last_margin = float('inf')  # last best_pred/best_measured ratio
        self._last_k_est = 0          # last count of configs within K_TOLERANCE of best
        self._t_algo_start = time.perf_counter_ns()  # wall clock for the whole selection
        self._end_logged = False
        # configs that failed to compile during R1/R2. dropped from the pool:
        # not measured, not re-proposed.
        self.not_profiled_indices = []

        # PHASE I.2: feature extraction
        self._X, self._feat_names = algo.extract_features(self._all_items)
        _check(self._X.ndim == 2, kernel_name,
               "extract_features X.ndim=%d expected 2", self._X.ndim)
        _check(self._X.shape[0] == self._N, kernel_name,
               "extract_features X.shape[0]=%d expected N=%d",
               self._X.shape[0], self._N)
        _check(len(self._feat_names) == self._X.shape[1], kernel_name,
               "feat_names len=%d != X.shape[1]=%d",
               len(self._feat_names), self._X.shape[1])
        self._D = self._X.shape[1] if self._X.shape[1] > 0 else 0

        # PHASE I.3 + I.4: budget sizing
        self._r1_size = max(int(algo.df_cfg.r1_pct * self._N), int(algo.R1_FLOOR_PCT * self._N))
        self._r1_size = min(self._r1_size, self._N)
        _check(0 < self._r1_size <= self._N, kernel_name,
               "_r1_size=%d not in (0, %d]", self._r1_size, self._N)

        p_lin = 1 + self._D if self._D > 0 else 1
        np_ratio = self._r1_size / p_lin
        # budget routing (mirrors algo.run):
        #   high:        small low-D kernels -> HIGH_BUDGET
        #   low:         large high-D kernels -> LOW_BUDGET, smaller R1
        #   high_legacy: high-D underflow fallback -> HIGH_BUDGET
        #   med:         rest -> BASE_BUDGET
        if self._D <= algo.HARD_D_THRESH and self._N <= algo.HARD_N_THRESH:
            self._total_budget = max(int(algo.df_cfg.high_budget * self._N), self._r1_size)
            self._budget_class = 'high'
            self._budget_frac = algo.df_cfg.high_budget
        elif self._D >= algo.EASY_D_THRESH and self._N >= algo.EASY_N_THRESH:
            self._total_budget = max(int(algo.df_cfg.low_budget * self._N), self._r1_size)
            self._r1_size = max(int(algo.R1_PCT_LOW * self._N), int(algo.R1_FLOOR_PCT * self._N))
            self._r1_size = min(self._r1_size, self._N)
            self._budget_class = 'low'
            self._budget_frac = algo.df_cfg.low_budget
        elif self._D >= algo.HIGH_D_THRESH and np_ratio < algo.HIGH_NP_THRESH:
            self._total_budget = max(int(algo.df_cfg.high_budget * self._N), self._r1_size)
            self._budget_class = 'high_legacy'
            self._budget_frac = algo.df_cfg.high_budget
        else:
            self._total_budget = max(int(algo.df_cfg.base_budget * self._N), self._r1_size)
            self._budget_class = 'med'
            self._budget_frac = algo.df_cfg.base_budget

        _check(self._total_budget >= self._r1_size, kernel_name,
               "total_budget=%d < r1_size=%d",
               self._total_budget, self._r1_size)

        # PHASE I.5: stratified or random R1 sample
        if self._D > 0:
            self._r1_indices = algo.stratified_sample(
                self._X, self._r1_size, self._rng)
        else:
            self._r1_indices = sorted(
                self._rng.choice(self._N, size=min(self._r1_size, self._N),
                                 replace=False).tolist())
        _check(len(self._r1_indices) > 0, kernel_name,
               "_r1_indices empty (r1_size=%d N=%d D=%d)",
               self._r1_size, self._N, self._D)
        _check(all(0 <= i < self._N for i in self._r1_indices), kernel_name,
               "_r1_indices out of [0,%d): %r",
               self._N, [i for i in self._r1_indices if not (0 <= i < self._N)][:5])
        _check(len(self._r1_indices) == len(set(self._r1_indices)), kernel_name,
               "_r1_indices has duplicates: %d vs %d unique",
               len(self._r1_indices), len(set(self._r1_indices)))

        # PHASE I.6: viability classification
        self._path, self._poly_mode = algo.assess_viability(
            len(self._r1_indices), self._D)
        _check(self._path in ('coverage', 'model'), kernel_name,
               "assess_viability path=%r not in {coverage, model}", self._path)
        if self._path == 'coverage':
            self._poly_mode = 'coverage'  # match algo.run telemetry

        # PHASE I.7: feature normalization. xmin/xmax/xrange are locals — only
        # used here to build _Xn, never read again.
        if self._D > 0:
            xmin = self._X.min(axis=0)
            xrange = self._X.max(axis=0) - xmin
            xrange[xrange == 0] = 1.0
            self._Xn = (self._X - xmin) / xrange
            _check(self._Xn.shape == self._X.shape, kernel_name,
                   "Xn shape %r != X shape %r", self._Xn.shape, self._X.shape)
        else:
            self._Xn = self._X

        # PHASE I.8: measured/unmeasured invariants
        self._unmeasured = set(range(self._N)) - set(self._r1_indices)
        _check(self._unmeasured.isdisjoint(set(self._r1_indices)), self._kernel_name,
               "_unmeasured overlaps _r1_indices")
        _check(len(self._unmeasured) + len(self._r1_indices) == self._N, self._kernel_name,
               "|unmeasured|+|r1|=%d != N=%d",
               len(self._unmeasured) + len(self._r1_indices), self._N)
        self._used = 0
        self._done = False

        self._overhead_ns += time.perf_counter_ns() - self._t_init

        # config banner (once)
        algo.log_algo_config()
        # per-kernel start line
        _prod_log.info(
            _TAG + " event=start kernel=%s N=%d D=%d class=%s budget_frac=%.2f "
            "r1_size=%d planned_budget=%d path=%s",
            self._kernel_name, self._N, self._D, self._budget_class,
            self._budget_frac, self._r1_size, self._total_budget, self._path)

        _selector_log(
            f'{kernel_name}: N={self._N} D={self._D} '
            f'feats={self._feat_names} r1={len(self._r1_indices)} '
            f'budget={self._total_budget} path={self._path} '
            f'poly={self._poly_mode} '
            f'init_overhead={self._overhead_ns / 1e6:.2f}ms',
            level='info')

    # =================================================================
    # batch api — driven by the production caller
    #
    # run(bench_fn) (the self-driving loop) moved to the offline harness
    # (offline_exp/run_filter.py); production never called it.
    # =================================================================

    @property
    def r1_configs(self):
        """return R1 configs (PHASE II.1). state-advancing but idempotent.

        first read advances state (_last_batch_indices = r1_indices, _round=1)
        so refine() can record the durations against the right items. later
        reads are no-ops since the assigned values match what is already there.

        this single property replaces the old (r1_configs peek / r1_batch
        advance) split, which had a footgun: a caller reading r1_configs would
        never trigger refine()'s recording loop and the R1 timings would
        silently never enter _measured.
        """
        if self._round == 0:
            # first read advances state; later reads converge to the same.
            self._last_batch_indices = list(self._r1_indices)
            self._round = 1
        return [self._all_items[i] for i in self._r1_indices]

    def refine(self, durations):
        """feed durations from the last batch, return the next batch.

        PHASE III. returns [] when done (budget exhausted or converged).

        hard contract on `durations`:
          - sized sequence (list / tuple / ndarray) of numeric values
          - len must equal len(self._last_batch_indices)
          - values non-NaN, non-negative (use +inf for failed configs)
          - none of self._last_batch_indices may already be in self._measured
            (that means the same batch was issued twice — the bug we hunt)
        """
        t0 = time.perf_counter_ns()

        # PHASE III.1.a: input contract
        _check(hasattr(durations, '__len__'), self._kernel_name,
               "refine() durations type=%s not sized", type(durations).__name__)
        _check(len(durations) == len(self._last_batch_indices), self._kernel_name,
               "refine() len(durations)=%d != len(last_batch)=%d",
               len(durations), len(self._last_batch_indices))
        for k, d in enumerate(durations):
            try:
                fd = float(d)
            except (TypeError, ValueError):
                raise FastaCheckError(
                    f"[FASTA-CHECK] {self._kernel_name}: refine() "
                    f"durations[{k}] not numeric: type={type(d).__name__} val={d!r}")
            _check(fd == fd, self._kernel_name,
                   "refine() durations[%d] is NaN", k)
            _check(fd >= 0.0, self._kernel_name,
                   "refine() durations[%d]=%r is negative", k, fd)

        # PHASE III.1.b: re-bench detection
        already = [i for i in self._last_batch_indices if i in self._measured]
        _check(not already, self._kernel_name,
               "refine() got indices already measured: %r (round=%d)",
               already[:5], self._round)

        # PHASE III.1.c: record measurements
        for idx, dur in zip(self._last_batch_indices, durations):
            self._measured[idx] = float(dur)
            self._unmeasured.discard(idx)
            if dur < self._best_dur:
                self._best_dur = float(dur)
                self._best_idx = idx
        self._used = len(self._measured)

        # PHASE III.1.d: invariants after recording
        # disjointness must always hold.
        _check(set(self._measured.keys()).isdisjoint(self._unmeasured),
               self._kernel_name,
               "post-record: measured cap unmeasured non-empty")
        # item accounting. an item lives in exactly ONE of:
        #   (a) measured           -- already benchmarked
        #   (b) unmeasured         -- not yet handed out to the caller
        #   (c) last_batch_pending -- handed out, awaiting the next refine() drain
        # after r1_configs is read (the first thing a caller does), the
        # _r1_indices subset moves from "implicit pending" to last_batch. no
        # leak possible: union of (a,b,c) must be exactly range(N).
        _accounted = (set(self._measured.keys())
                      | self._unmeasured
                      | set(self._last_batch_indices))
        _missing = (set(range(self._N))
                    - set(self.not_profiled_indices)
                    - _accounted)
        _check(not _missing, self._kernel_name,
               "post-record: %d/%d indices unaccounted "
               "(measured=%d unmeasured=%d last_batch=%d). missing[:5]=%r",
               len(_missing), self._N,
               len(self._measured), len(self._unmeasured),
               len(self._last_batch_indices), sorted(_missing)[:5])

        if self._done or not self._unmeasured:
            self._overhead_ns += time.perf_counter_ns() - t0
            self._log_done()
            return []

        if self._path == 'coverage':
            batch_indices = self._coverage_step()
            self._done = True
            self._last_batch_indices = batch_indices
            self._round += 1
            self._overhead_ns += time.perf_counter_ns() - t0
            if not batch_indices:
                self._log_done()
                return []
            return [self._all_items[i] for i in batch_indices]

        batch_indices = self._model_step()
        if not batch_indices:
            self._done = True
            self._overhead_ns += time.perf_counter_ns() - t0
            self._log_done()
            return []

        self._last_batch_indices = batch_indices
        self._round += 1
        self._overhead_ns += time.perf_counter_ns() - t0

        _selector_log(
            f'{self._kernel_name}: round {self._round} '
            f'batch={len(batch_indices)} measured={self._used} '
            f'best={self._best_dur:.3f}')

        return [self._all_items[i] for i in batch_indices]

    # =================================================================
    # internal
    # =================================================================

    def _coverage_step(self):
        cov_budget = max(int(algo.df_cfg.high_budget * self._N), self._r1_size)
        remaining = cov_budget - self._used
        if remaining <= 0 or not self._unmeasured:
            return []
        pool = sorted(self._unmeasured)
        n_pick = min(remaining, len(pool))
        chosen = self._rng.choice(len(pool), size=n_pick, replace=False)
        return [pool[c] for c in chosen]

    def _model_step(self):
        if self._round >= algo.df_cfg.max_rounds:
            return []

        midx = sorted(self._measured.keys())
        uidx = sorted(self._unmeasured)
        if not uidx:
            return []

        Xm = self._Xn[midx]
        dm = np.array([self._measured[i] for i in midx])
        Xu = self._Xn[uidx]
        D = self._D

        d_hat, r_sq, chosen_mode = algo.select_mode_and_predict(Xm, dm, Xu, D)
        # PHASE IV.2 output check
        _check(d_hat.shape == (len(uidx),), self._kernel_name,
               "select_mode_and_predict d_hat.shape=%r expected (%d,)",
               d_hat.shape, len(uidx))
        _check(not np.any(np.isnan(d_hat)), self._kernel_name,
               "select_mode_and_predict returned NaN d_hat (mode=%s r_sq=%r)",
               chosen_mode, r_sq)
        _check(0.0 <= r_sq <= 1.0, self._kernel_name,
               "select_mode_and_predict r_sq=%r outside [0,1]", r_sq)
        self._poly_mode = chosen_mode
        self._round_estimators.append(chosen_mode)

        best_m = self._best_dur
        # per-round kpis (computed regardless of the stop decision):
        #   margin = predicted-best-unmeasured / best-measured. margin>=1 means
        #     the surrogate believes the winner is already in hand.
        #   r_sq   = surrogate fit quality in [0,1]; high = landscape understood.
        #   k_est  = configs within K_TOLERANCE of best; high = many near-ties
        #     (easy kernel, any pick fine); 1 = sharp single winner (high stakes).
        best_pred = float(np.min(d_hat)) if len(d_hat) else float('inf')
        margin = (best_pred / best_m) if best_m > 0 else float('inf')
        k_est = sum(1 for v in self._measured.values()
                    if v <= best_m * algo.K_TOLERANCE)
        self._last_r_sq = float(r_sq)
        self._last_margin = float(margin)
        self._last_k_est = int(k_est)
        self._round_records.append({
            'round': self._round, 'est': chosen_mode, 'r_sq': float(r_sq),
            'margin': float(margin), 'k_est': int(k_est),
            'measured': self._used, 'best': float(best_m),
        })
        _prod_log.info(
            _TAG + " event=round kernel=%s round=%d est=%s r_sq=%.3f "
            "margin=%.3f k_est=%d measured=%d/%d best_dur=%.4f",
            self._kernel_name, self._round, chosen_mode, float(r_sq),
            float(margin), int(k_est), self._used, self._N, best_m)

        stop, self._prev_hist = algo.should_stop(dm, best_m, d_hat, self._prev_hist)
        if stop:
            return []

        batch_size = max(1, self._total_budget // algo.df_cfg.max_rounds)
        remaining_budget = self._total_budget - self._used
        batch_size = min(batch_size, len(uidx), remaining_budget)
        if batch_size <= 0:
            return []

        tau = algo.compute_tau(r_sq, k_est)

        if D >= algo.MARGINAL_D:
            n_poly = max(1, batch_size // 2)
            n_marg = max(1, batch_size - n_poly)
            sel_poly = set(algo.softmax_select(d_hat, tau, n_poly, self._rng).tolist())
            m_scores = algo.marginal_scores(self._X[midx], dm, self._X[uidx], D)
            sel_marg = set(algo.softmax_select(
                m_scores, algo.TAU_MIN, n_marg, self._rng).tolist())
            merged = sorted(sel_poly | sel_marg)[:batch_size]
        else:
            merged = algo.softmax_select(d_hat, tau, batch_size, self._rng).tolist()

        return [uidx[c] for c in merged]

    def _log_done(self):
        # per-kernel end line (always on). regret is NA in production: the
        # filter never measures the unmeasured configs, so it cannot know the
        # true best — regret is offline-only. log what we have: real budget
        # used, rounds, per-round estimators, best_dur, quality, wall time.
        if self._end_logged:
            return
        self._end_logged = True
        _algo_wall_ms = (time.perf_counter_ns() - self._t_algo_start) / 1e6
        _real_pct = (100.0 * self._used / self._N) if self._N else 0.0
        _ests = self._round_estimators if self._round_estimators else [self._poly_mode]
        _conf = _confidence_label(self._last_r_sq, self._last_margin, self._last_k_est)
        # stop reason: budget exhausted vs converged early vs all measured
        _planned = self._total_budget
        _stop = 'budget' if self._used >= _planned else ('all' if not self._unmeasured else 'converged')
        _prod_log.info(
            _TAG + " event=end kernel=%s N=%d D=%d class=%s path=%s "
            "real_budget=%d/%d(%.1f%%) planned=%d rounds=%d estimators=[%s] "
            "best_dur=%.4f stop=%s quality=%s r_sq=%.3f margin=%.3f k_est=%d "
            "regret=NA catch5=NA filter_overhead_ms=%.2f algo_wall_ms=%.2f",
            self._kernel_name, self._N, self._D,
            getattr(self, '_budget_class', '?'), self._path,
            self._used, self._N, _real_pct, _planned, self._round,
            ",".join(_ests), self._best_dur, _stop, _conf,
            self._last_r_sq, self._last_margin, self._last_k_est,
            self._overhead_ns / 1e6, _algo_wall_ms)
        _selector_log(
            f'{self._kernel_name}: done. '
            f'evals={self._used}/{self._N} '
            f'savings={100 * (1 - self._used / self._N):.1f}% '
            f'best_dur={self._best_dur:.3f} '
            f'path={self._path} mode={self._poly_mode} '
            f'overhead={self._overhead_ns / 1e6:.2f}ms',
            level='info')

    # =================================================================
    # results
    # =================================================================

    # best property removed with run() -> offline_exp/run_filter.py. production
    # reads the winner via stats / _best_idx, not via a (config, dur) tuple.

    @property
    def overhead_ms(self):
        return self._overhead_ns / 1e6

    @property
    def stats(self):
        return {
            'kernel': self._kernel_name,
            'N': self._N,
            'D': self._D,
            'features': self._feat_names,
            'evals': self._used,
            'savings_pct': 100 * (1 - self._used / self._N) if self._N > 0 else 0,
            'rounds_used': self._round,
            'path': self._path,
            'mode': self._poly_mode,
            'best_dur': self._best_dur,
            'overhead_ms': self.overhead_ms,
            'r_sq': self._last_r_sq,
            'margin': self._last_margin,
            'k_est': self._last_k_est,
            'confidence': _confidence_label(self._last_r_sq, self._last_margin, self._last_k_est),
            'round_records': list(self._round_records),
        }

    def update_batch_indices(self, pvs):
        """reconcile _last_batch_indices with the actual pvs from the bench
        machinery.

        two jobs:
          1. mark configs in the old _last_batch_indices that produced no
             profile value as compile-failed: move them to
             not_profiled_indices and discard from _unmeasured.
          2. rebuild _last_batch_indices in pvs order so refine()'s
             zip(_last_batch_indices, durations) stays aligned.

        job 2 matters because precompile_parallel uses as_completed(), so
        launcher order != config submission order, and pvs is built in launcher
        order. the old in-place filter only handled job 1.
        """
        # uniqueness asserts on inputs.
        batch_objs = [self._all_items[v] for v in self._last_batch_indices]
        _check(len(set(id(o) for o in batch_objs)) == len(batch_objs),
               self._kernel_name,
               "update_batch_indices: duplicate config object in last_batch")
        pv_configs = [pv.config for pv in pvs]
        _check(len(set(id(c) for c in pv_configs)) == len(pv_configs),
               self._kernel_name,
               "update_batch_indices: duplicate config object across pvs")

        # build id -> global index from the OLD last_batch only
        old_id_to_global = {id(self._all_items[v]): v
                            for v in self._last_batch_indices}
        profiled_ids = {id(pv.config) for pv in pvs}

        # (1) failures: in the old batch but not in pvs
        for v in self._last_batch_indices:
            if id(self._all_items[v]) not in profiled_ids:
                self.not_profiled_indices.append(v)
                self._unmeasured.discard(v)

        # (2) rebuild in pvs order
        new_last_batch = []
        for pv in pvs:
            g = old_id_to_global.get(id(pv.config))
            _check(g is not None, self._kernel_name,
                   "update_batch_indices: pv.config not in last_batch "
                   "(kwargs=%r)", pv.config.kwargs)
            new_last_batch.append(g)
        self._last_batch_indices = new_last_batch

        _check(len(self._last_batch_indices) == len(pvs),
               self._kernel_name,
               "update_batch_indices: post-sync %d != pvs %d",
               len(self._last_batch_indices), len(pvs))
