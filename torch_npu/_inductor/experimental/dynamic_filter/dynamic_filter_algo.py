"""
dynamic filter algorithm: efficient kernel selection via adaptive
surrogate modeling.

pure math, minimal INFO logging, no other I/O, no framework deps beyond
stdlib logging + numpy.

motivation: kernel selection costs real wall-clock time. with N configs and
limited measurement budget (8-16% of N typical), brute-force evaluation is
infeasible. this algorithm trades early modeling (R1 stratified sample) for
fast config proposal (softmax-weighted) to converge in 40-50 evals on hard
kernels (K/N ≈ 1-2%).

in:  feature matrix X (N x D), measurement callback (live timing or offline
     precomputed durations).
out: best config index + convergence stats (catch rate, regret, iterations).

protocol: plugs into dynamic_algo_if for uniform online/offline dispatch—same
code runs live on NPU or replays recorded measurements for validation.

given N compiled tiling configs for a triton kernel, find the fastest while
measuring as few as possible:
  1. measure a stratified R1 sample (driven by identifiability)
  2. compete basis models (linear/quad/full/fourier/cubic) by spearman-ranked
     loocv, keep the best surrogate
  3. softmax-sample the next batch toward predicted-fast configs
  4. repeat until the budget runs out or it converges

current estimators (mean rank, quadratic form) capture typical
harmonic + smooth patterns. future estimators (e.g., ratio estimator for tail
risk, worse-kernel baseline) can be composed via bayesian model averaging or
competing loss functions. contatct asaf.goldberg@huawei.com

logging:
  log_algo_config() dumps the global constants once.
  run(..., kernel_name=...) emits the per-kernel start/end lines under the
  [FASTA_DYN_FILTER_ALGO] tag. regret is only real when ground-truth durations
  are passed (offline sim).

"""

# import time  # unused: only the commented-out run()/_log_end referenced it
import logging
from torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config import fasta_dynamic_filter as df_cfg
import numpy as np

log = logging.getLogger("torch._inductor")

_TAG = "[FASTA_DYN_FILTER_ALGO]"


# constants
R1_FLOOR_PCT: float = 0.10
R1_PCT_LOW: float = 0.12
HIGH_D_THRESH: int = 4
HIGH_NP_THRESH: int = 5
HARD_D_THRESH: int = 3
HARD_N_THRESH: int = 150
EASY_D_THRESH: int = 4
EASY_N_THRESH: int = 150
MIN_VIABLE_NP: int = 2
TAU_MIN: float = 0.3
TAU_RANGE: float = 0.4
MARGINAL_D: int = 3
RIDGE_LAMBDA: float = 0.1
MARGIN_CONV: float = 2.0
L1_CONV: float = 0.01
HIST_BINS: int = 11
K_TOLERANCE: float = 1.05
UNDERFLOW_FLOOR: float = 1e-300

SKIP_KWARGS = frozenset({
    'compile_mode', 'multibuffer', 'split_k',
    'remain_programs', 'using_programs',
})

BASE_MODES = ['linear', 'quad', 'full', 'fourier']

# config banner fires once even if called again
_CONFIG_LOGGED = False

def log_algo_config(force=False):
    """dump the global constants once, at INFO. only emits on the first call
    unless force=True."""
    global _CONFIG_LOGGED
    if _CONFIG_LOGGED and not force:
        return
    _CONFIG_LOGGED = True
    log.info(
        _TAG + " event=config "
        "R1_PCT=%s R1_FLOOR_PCT=%s R1_PCT_LOW=%s "
        "BASE_BUDGET=%s HIGH_BUDGET=%s LOW_BUDGET=%s "
        "HARD_D_THRESH=%s HARD_N_THRESH=%s EASY_D_THRESH=%s EASY_N_THRESH=%s "
        "HIGH_D_THRESH=%s HIGH_NP_THRESH=%s MIN_VIABLE_NP=%s "
        "TAU_MIN=%s TAU_RANGE=%s MARGINAL_D=%s RIDGE_LAMBDA=%s "
        "MAX_ROUNDS=%s MARGIN_CONV=%s L1_CONV=%s HIST_BINS=%s "
        "K_TOLERANCE=%s UNDERFLOW_FLOOR=%s",
        df_cfg.r1_pct, R1_FLOOR_PCT, R1_PCT_LOW,
        df_cfg.base_budget, df_cfg.high_budget, df_cfg.low_budget,
        HARD_D_THRESH, HARD_N_THRESH, EASY_D_THRESH, EASY_N_THRESH,
        HIGH_D_THRESH, HIGH_NP_THRESH, MIN_VIABLE_NP,
        TAU_MIN, TAU_RANGE, MARGINAL_D, RIDGE_LAMBDA,
        df_cfg.max_rounds, MARGIN_CONV, L1_CONV, HIST_BINS,
        K_TOLERANCE, UNDERFLOW_FLOOR,
    )




# =====================================================================
# feature extraction
# =====================================================================

def extract_features(configs):
    """build the feature matrix from config kwargs.

    scan every config's kwargs for numeric keys, keep only the columns that
    vary across configs.

    args:
        configs: list of objects with a .kwargs dict.

    returns:
        X: np.ndarray (N, D) feature matrix.
        names: list[str] of length D, sorted feature names.
    """
    N = len(configs)
    all_keys = set()
    for cfg in configs:
        kw = cfg.kwargs if hasattr(cfg, 'kwargs') else {}
        for k, v in kw.items():
            if k in SKIP_KWARGS:
                continue
            try:
                float(v)
                all_keys.add(k)
            except (TypeError, ValueError):
                pass

    if not all_keys or N == 0:
        return np.zeros((N, 0)), []

    sorted_keys = sorted(all_keys)
    X_full = np.zeros((N, len(sorted_keys)))
    for i, cfg in enumerate(configs):
        kw = cfg.kwargs if hasattr(cfg, 'kwargs') else {}
        for j, k in enumerate(sorted_keys):
            val = kw.get(k, 0)
            try:
                X_full[i, j] = float(val)
            except (TypeError, ValueError):
                X_full[i, j] = 0.0

    varying = [j for j in range(len(sorted_keys))
               if len(np.unique(X_full[:, j])) > 1]
    names = [sorted_keys[j] for j in varying]
    X = X_full[:, varying] if varying else np.zeros((N, 0))
    return X, names


# =====================================================================
# viability
# =====================================================================

def assess_viability(n_samples, D):
    """decide whether a polynomial model is viable, and at what complexity.

    returns:
        path: 'coverage' | 'model'
        poly_mode: 'full' | 'quad' | 'linear'
    """
    if D == 0:
        return 'coverage', 'linear'
    p_lin = 1 + D
    if n_samples / p_lin < MIN_VIABLE_NP:
        return 'coverage', 'linear'
    p_full = 1 + 2 * D + D * (D - 1) // 2
    if n_samples / p_full >= MIN_VIABLE_NP + 1:
        return 'model', 'full'
    p_quad = 1 + 2 * D
    if n_samples / p_quad >= MIN_VIABLE_NP:
        return 'model', 'quad'
    return 'model', 'linear'


# =====================================================================
# feature bases: polynomial, fourier, cubic
# =====================================================================

def build_poly_features(X, D, mode):
    """expand raw features into a polynomial basis (linear/quad/full)."""
    n = len(X)
    feats = [np.ones(n)]
    for i in range(D):
        feats.append(X[:, i])
    if mode in ('quad', 'full'):
        for i in range(D):
            feats.append(X[:, i] ** 2)
    if mode == 'full':
        for i in range(D):
            for j in range(i + 1, D):
                feats.append(X[:, i] * X[:, j])
    return np.column_stack(feats)


def build_fourier_features(X, D):
    """fourier basis: x, sin(pi x), cos(pi x), sin(2 pi x), cos(2 pi x)."""
    n = len(X)
    feats = [np.ones(n)]
    for i in range(D):
        xi = X[:, i]
        feats.append(xi)
        feats.append(np.sin(np.pi * xi))
        feats.append(np.cos(np.pi * xi))
        feats.append(np.sin(2 * np.pi * xi))
        feats.append(np.cos(2 * np.pi * xi))
    return np.column_stack(feats)


def build_cubic_features(X, D):
    """full cubic polynomial basis."""
    n = len(X)
    feats = [np.ones(n)]
    for i in range(D):
        feats.append(X[:, i])
    for i in range(D):
        feats.append(X[:, i] ** 2)
    for i in range(D):
        for j in range(i + 1, D):
            feats.append(X[:, i] * X[:, j])
    for i in range(D):
        feats.append(X[:, i] ** 3)
    for i in range(D):
        for j in range(D):
            if j == i:
                continue
            feats.append(X[:, i] ** 2 * X[:, j])
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                feats.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(feats)


def cubic_param_count(D):
    """param count of the cubic basis for D features."""
    p = 1 + D + D + D * (D - 1) // 2
    p += D + D * (D - 1)
    p += D * (D - 1) * (D - 2) // 6 if D >= 3 else 0
    return p


def build_features(X, D, mode):
    """dispatch to the right basis builder."""
    if mode == 'fourier':
        return build_fourier_features(X, D)
    if mode == 'cubic':
        return build_cubic_features(X, D)
    return build_poly_features(X, D, mode)


# =====================================================================
# spearman rank correlation
# =====================================================================

def numpy_spearman(a, b):
    """spearman rank correlation between two vectors."""
    n = len(a)
    if n < 3:
        return 0.0
    order_a = np.argsort(a)
    ranks_a = np.empty(n, dtype=float)
    ranks_a[order_a] = np.arange(1, n + 1, dtype=float)
    order_b = np.argsort(b)
    ranks_b = np.empty(n, dtype=float)
    ranks_b[order_b] = np.arange(1, n + 1, dtype=float)
    std_a, std_b = np.std(ranks_a), np.std(ranks_b)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    corr = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    return 0.0 if np.isnan(corr) else corr


# =====================================================================
# mode selection: spearman-ranked loocv
# =====================================================================

def select_mode_and_predict(Xm_norm, dm, Xu_norm, D):
    """compete basis modes by spearman-ranked loocv, predict the unmeasured set.

    args:
        Xm_norm: (n, D) normalized features of measured configs
        dm: (n,) measured durations
        Xu_norm: (m, D) normalized features of unmeasured configs
        D: int

    returns:
        d_hat: (m,) predicted durations for the unmeasured set
        best_r_sq: float, R^2 of the winning model
        best_mode: str, name of the winning mode
    """
    n_meas = len(dm)
    p_cub = cubic_param_count(D)

    modes = list(BASE_MODES)
    if p_cub < n_meas / 2.0:
        modes.append('cubic')

    best_mode = 'linear'
    best_rank_corr = -2.0
    best_beta = None
    best_r_sq = 0.0

    for mode in modes:
        Fm = build_features(Xm_norm, D, mode)
        n, p = Fm.shape
        gram = Fm.T @ Fm + RIDGE_LAMBDA * np.eye(p)
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            continue

        beta = gram_inv @ Fm.T @ dm
        residuals = dm - Fm @ beta

        H = Fm @ gram_inv @ Fm.T
        h_diag = np.diag(H)
        denom = np.maximum(1.0 - h_diag, 0.01)
        loocv_pred = dm - residuals / denom

        rank_corr = numpy_spearman(dm, loocv_pred)
        if rank_corr > best_rank_corr:
            best_rank_corr = rank_corr
            best_mode = mode
            best_beta = beta
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((dm - np.mean(dm)) ** 2)
            best_r_sq = max(0.0, min(1.0, 1.0 - ss_res / (ss_tot + 1e-12)))

    if best_beta is None:
        Fm = build_features(Xm_norm, D, 'linear')
        gram = Fm.T @ Fm + RIDGE_LAMBDA * np.eye(Fm.shape[1])
        best_beta = np.linalg.solve(gram, Fm.T @ dm)
        best_mode = 'linear'
        best_r_sq = 0.0

    Fu = build_features(Xu_norm, D, best_mode)
    d_hat = Fu @ best_beta
    return d_hat, best_r_sq, best_mode


# =====================================================================
# softmax batch selection
# =====================================================================

def softmax_select(scores, tau, batch_size, rng):
    """pick the next batch to measure, biased toward low scores."""
    s = -scores / tau
    s -= s.max()
    p = np.exp(s)
    p = np.maximum(p, UNDERFLOW_FLOOR)
    p /= p.sum()
    n = min(batch_size, len(scores))

    n_nonzero = int(np.sum(p > 0))
    if n_nonzero >= n:
        return rng.choice(len(scores), size=n, replace=False, p=p)
    if n_nonzero == 0:
        return rng.choice(len(scores), size=n, replace=False)

    nonzero_mask = p > 0
    nonzero_idx = np.where(nonzero_mask)[0]
    zero_idx = np.where(~nonzero_mask)[0]
    p_nz = p[nonzero_mask]
    p_nz /= p_nz.sum()
    softmax_picks = rng.choice(nonzero_idx, size=n_nonzero, replace=False, p=p_nz)
    n_random = n - n_nonzero
    random_picks = rng.choice(zero_idx, size=min(n_random, len(zero_idx)), replace=False)
    return np.concatenate([softmax_picks, random_picks])


# =====================================================================
# marginal voting
# =====================================================================

def marginal_scores(X_measured, d_measured, X_unmeasured, D):
    """score unmeasured configs by per-dimension marginal averages."""
    m = len(X_unmeasured)
    scores = np.zeros(m)
    for dim in range(D):
        val_to_durs = {}
        for j in range(len(X_measured)):
            v = X_measured[j, dim]
            if v not in val_to_durs:
                val_to_durs[v] = []
            val_to_durs[v].append(d_measured[j])
        val_to_mean = {v: np.mean(ds) for v, ds in val_to_durs.items()}
        global_mean = np.mean(d_measured)
        for j in range(m):
            v = X_unmeasured[j, dim]
            scores[j] += val_to_mean.get(v, global_mean)
    return scores


# =====================================================================
# stratified R1 sampling
# =====================================================================

def stratified_sample(X, n_sample, rng):
    """pick R1 indices: one representative per unique value, most-unique dim first."""
    N, D = X.shape
    selected = set()
    used_dims = set()
    remaining = n_sample

    while remaining > 0 and len(used_dims) < D:
        best_dim, best_count = -1, -1
        for dim in range(D):
            if dim in used_dims:
                continue
            n_unique = len(np.unique(X[:, dim]))
            if n_unique > best_count:
                best_dim, best_count = dim, n_unique
        if best_dim < 0:
            break
        used_dims.add(best_dim)
        for val in np.unique(X[:, best_dim]):
            if remaining <= 0:
                break
            candidates = [i for i in range(N)
                          if X[i, best_dim] == val and i not in selected]
            if candidates:
                selected.add(rng.choice(candidates))
                remaining -= 1

    if remaining > 0:
        pool = [i for i in range(N) if i not in selected]
        if pool:
            n_pick = min(remaining, len(pool))
            picks = rng.choice(pool, size=n_pick, replace=False)
            selected.update(picks.tolist())

    return sorted(selected)


# =====================================================================
# temperature
# =====================================================================

def compute_tau(r_sq, k_est):
    """softmax temperature from model quality and needle rarity."""
    k_factor = min(1.0, k_est / 4.0)
    return TAU_MIN + TAU_RANGE * (1.0 - r_sq) * k_factor


# =====================================================================
# convergence
# =====================================================================

def should_stop(dm_array, best_measured, d_hat_unmeasured, prev_hist):
    """margin and histogram-stability stopping checks."""
    if len(dm_array) == 0:
        return False, prev_hist

    if len(d_hat_unmeasured) > 0:
        best_pred = np.min(d_hat_unmeasured)
        margin = best_pred / best_measured if best_measured > 0 else 0
        if margin > MARGIN_CONV:
            return True, prev_hist

    bins = np.linspace(np.min(dm_array), np.max(dm_array), HIST_BINS)
    hist_now = np.histogram(dm_array, bins=bins, density=True)[0]
    hist_now = hist_now / (hist_now.sum() + 1e-12)
    if prev_hist is not None:
        l1 = np.sum(np.abs(hist_now - prev_hist))
        if l1 < L1_CONV:
            return True, hist_now

    return False, hist_now