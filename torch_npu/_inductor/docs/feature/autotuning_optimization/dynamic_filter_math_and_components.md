# Dynamic Filter — Mathematical & Component Reference

Contains full derivations for all mathematical components of the dynamic filter algorithm.

Structure: each section starts with general textbook theory (what the method is, where it comes from), then shows how we specialized it for our problem. Every matrix operation includes shapes. Every component lists its inputs, outputs, and connections to other components.

## Notation

### Index conventions

Subscripts on x (parameters within a single config):
xᵢ = the i-th tiling parameter of a single config. i ranges from 1 to D. Example: x₁ = Y0BLOCK, x₂ = Y0BLOCK_SUB, x₃ = core_num.

Superscripts or separate index on configs:
x⁽ʲ⁾ or xⱼ = the j-th config in the dataset. j ranges from 1 to N (all configs) or 1 to n (measured configs).

### Sets

M = set of measured config indices (configs we have benchmarked). |M| = n.
U = set of unmeasured config indices (configs we have NOT benchmarked). |U| = N − n.
N = total configs. D = number of variable tiling parameters (1–6). K = number of "good" configs (duration ≤ 1.05 × d_best).

### Matrices and vectors

| Symbol | Shape | Description |
|--------|-------|-------------|
| X | (N, D) | raw feature matrix, all configs |
| Xₘ | (n, D) | raw features of measured configs |
| Xᵤ | (N−n, D) | raw features of unmeasured configs |
| Xₘ_norm | (n, D) | normalized features of measured configs |
| Xᵤ_norm | (N−n, D) | normalized features of unmeasured configs |
| Fₘ | (n, p) | basis-expanded features of measured configs |
| Fᵤ | (N−n, p) | basis-expanded features of unmeasured configs |
| d | (n,) | measured durations |
| d̂ | (N−n,) | predicted durations for unmeasured configs |
| β̂ | (p,) | estimated regression coefficients |
| G | (p, p) | Gram matrix FₘᵀFₘ + λI |
| H | (n, n) | hat matrix Fₘ G⁻¹ Fₘᵀ |
| p | scalar | number of basis features (varies by basis type) |

Convention: plain symbols (d, X, F) are observed/constructed from data. Symbols with hat (d̂, β̂, R̂²) are estimated by the algorithm. Subscript m = measured set, u = unmeasured set.

### Error metrics map

Two distinct error metrics appear in the algorithm at different stages:

| Metric | Formula | Used in | Purpose |
|--------|---------|---------|---------|
| SS_res (L²) | Σ(dᵢ − d̂ᵢ)² | R̂² computation (§9) | model quality → τ̂ adaptation |
| L₁ distance | Σ\|hₜ − hₜ₋₁\| | convergence (§13) | distribution stability → early stop |
| ρ_s (Spearman) | rank correlation | mode selection (§9) | basis competition criterion |

Key distinction: R̂² measures how well the model fits training data (squared error between observed d and predicted d̂ on the measured set). L₁ measures how much the distribution of measured durations changed between rounds (histogram comparison, no model involved). ρ_s measures rank agreement between actual and cross-validated predictions (used to pick the best basis model). They are independent calculations used at different stages.

---

## §1. Problem Formulation

Given N tiling configurations for a Triton kernel, find the fastest one by measuring as few as possible. Each measurement costs compile time + benchmark time. We want to measure ~35% of N and still find a config within 5% of the true optimum.

Formally: let d(·) be the unknown duration function. We observe d($j$) for a subset M ⊂ {1..N} of our choosing, sequentially. The goal is to find $i^*$ such that d($i^*$) ≤ 1.05 · minⱼ d($j$), while |M| ≤ 0.35N.

This is a sequential experiment design problem, closely related to best-arm identification in multi-armed bandits, but with side information (the feature vectors of each config).

### Evaluation metrics

Catch rate: fraction of simulation trials where the algorithm found a config within 5% of the true optimum. Higher is better. 100% = always finds a near-optimal config.

Savings: fraction of configs not measured. savings = 1 − |M|/N. Higher is better.

Regret: how much worse the selected config is vs the true optimum. regret = (d($i^*$) − min d) / min d × 100%. Lower is better.

K (needle count): number of configs with d ≤ 1.05 × min d. Low K = needle in a haystack. High K = easy landscape.

---

## §2. Feature Extraction

**Component 1.** Always runs first. No conditions, no gate.

```text
inputs:  configs (list of N objects with .kwargs dict)
outputs: X (N, D) — feature matrix
         names (list of D strings) — feature names, sorted
feeds:   stratified R1 sampling (X)
         viability assessment (D)
         budget allocation (D)
         normalization (X)
```

Each config has a kwargs dict mapping parameter names to numeric values. We scan all configs, collect keys that are numeric, and keep only columns where the value varies (more than one unique value across configs). Categorical flags (compile_mode, multibuffer, split_k, etc.) are skipped.

Output: X ∈ ℝ^(N×D), where N = number of configs, D = number of varying features.

The old approach used a hardcoded list of 7 parameter names. This missed entire families (T0BLOCK, Z1BLOCK, Y1BLOCK, etc.). 28 of 35 test kernels had incorrect D under the hardcoded scheme. Auto-detection was the single largest impact fix (exp38).

---

## §3. R1 Sizing: The Hypergeometric Distribution

**Theoretical foundation for R1 sample size.** Determines how many configs to benchmark blind before the model kicks in.

### 3.1 General theory

The hypergeometric distribution models the number of successes in draws from a finite population without replacement. If a population of N items contains K "good" items, and we draw n without putting them back, the probability of getting exactly k good items is:

```text
P(X = k) = C(K,k) · C(N−K, n−k) / C(N,n)
```

where C(a,b) = a! / [b!(a−b)!] is the binomial coefficient.

Reference: [1] Rice (2006), Mathematical Statistics and Data Analysis, 3rd ed. Ch. 2: hypergeometric as sampling without replacement.

Key distinction from the binomial: the binomial assumes each draw is independent (with replacement). The hypergeometric accounts for the shrinking pool — after drawing a good item, fewer good items remain.

### 3.2 The special case: P(miss all good)

We care about k=0 specifically: probability of drawing zero good items.

```text
P(miss all) = C(N−K, n) / C(N, n)
```

Expanding into a computationally stable product form:

```text
P(miss) = ∏ᵢ₌₀ⁿ⁻¹ (N − K − i) / (N − i)
```

Each factor is: "of the remaining unchosen configs, what fraction are bad?" At draw i=0: (N−K)/N are bad. At draw i=1: (N−K−1)/(N−1) are bad (one bad config was already drawn). And so on.

### 3.3 Our application

We set r₁ = 15% of N. The hypergeometric tells us what this implies:

Hard kernel (K=2, N=143, add_layer_norm_0): r₁ = 21. P(miss) ≈ 0.74, so P(catch) ≈ 26%. R1 alone has a 1-in-4 chance.

Easy kernel (K=47, N=1800, addmm_tanh_9): r₁ = 270. P(miss) ≈ 0.00003. Caught with certainty.

Closed form for K=2: P(miss) = (N−n)(N−n−1) / [N(N−1)]. Requires n ≈ 90% of N to guarantee P(miss) ≤ 0.01. This is why R1 alone cannot solve K=2 kernels and why the estimator is essential.

---

## §4. Stratified R1 Sampling

**Component 2.** Always runs first round. No conditions.

```text
inputs:  X (N, D) — feature matrix
         n_sample — number of configs to sample (r₁)
         rng — random state
outputs: r1_idx (list of r₁ indices)
feeds:   measure function → measured durations
         viability assessment (n = |r1_idx|)
         mode selection (training data)
```

R1 (the blind exploration phase, 15% of N) uses most-unique-first stratification:

1. Find the dimension with the most unique values.
2. For each unique value in that dimension, randomly pick one config with that value.
3. Move to next-most-unique dimension. Repeat until budget exhausted.
4. Fill remaining budget randomly.

Why most-unique-first? If x₁ has 50 unique values and x₃ has 3, covering x₁ first gives the polynomial better training data because that dimension drives the performance landscape. The 16 picks for x₁ randomly cover most of x₃'s 3 values anyway. Experiment 48: least-unique-first drops one kernel from 99% to 77.5%.

---

## §5. Budget Allocation

**Component 3.** Gate / decision point. Runs once at start.

```text
inputs:  N — total configs
         D — number of features
         r₁ — R1 sample size
outputs: total_budget — maximum configs to measure
feeds:   per-round batch size calculation
         total stopping criterion
```

Base budget: 35% of N. High budget: 50% for kernels with D ≥ 4 AND n/p_linear < 5.

Why a step function at D=4, n/p<5? Experiment 49 showed that a smooth budget function (linear interpolation between 35% and 50%) kills clone_6 (99.5→91.0%). clone_6 has N=199, D=5 — right at the boundary. The smooth function gives it 35.6% budget, but it needs the full 50% because it has only K=4 needles in 199 configs. The step function at D=4 correctly gives it 50%.

---

## §6. Viability Assessment

**Component 4.** Binary gate: model path vs coverage fallback. Checked once after R1.

```text
inputs:  n — number of measured samples (|M|)
         D — number of features
outputs: path ∈ {'model', 'coverage'}
feeds:   coverage fallback (if path = 'coverage')
         mode selection (if path = 'model')
```

Computes p_linear = 1 + D. If n / p_linear < 2, even the simplest linear model has barely more equations than unknowns — OLS is unstable and predictions are noise. This is the high-variance regime of the bias-variance tradeoff ([2] Hastie et al. Ch. 7.3): too many parameters relative to data means the model fits noise rather than signal.

Reference: [11] Bühlmann & van de Geer (2011). Statistics for High-Dimensional Data. Ridge in the p > n regime.

**v6 difference:** in v6, viability also chose the polynomial complexity (linear → quad → full) based on n/p thresholds for each mode. That branching is removed in v7 — the LOOCV mode competition (§9) handles basis selection autonomously.

---

## §7. Coverage Fallback

**Component 5.** Activates when viability fails (n/p_linear < 2).

```text
inputs:  N — total configs
         measured — current measured set
outputs: best_idx — index of best config found
         best_dur — duration of best config
feeds:   algorithm return (terminates the kernel)
```

Random sampling at 50% budget. No model, no predictions. This saved softmax_5 (N=26, D=3) from 72% to 96% (exp42) — with only 26 configs and 3 features, any polynomial model is pure noise.

---

## §8. Feature Normalization & Basis Construction

**Component 6.** Runs every round on model path. Normalization computed once, basis constructed per candidate.

```text
inputs:  X (N, D) — raw feature matrix
         M, U — measured/unmeasured index sets
         mode ∈ {'linear', 'quad', 'full', 'fourier', 'cubic'} — basis type
outputs: Fₘ (n, p) — measured feature matrix
         Fᵤ (N−n, p) — unmeasured feature matrix
feeds:   ridge regression (Fₘ, Fᵤ)
```

### 8.1 Normalization

Before basis features are built, each raw feature column is min-max normalized to [0,1]. Without normalization, features at different scales (Y0BLOCK in [32,1024] vs core_num in [1,8]) produce a Gram matrix with a large condition number, making the normal equations numerically unstable ([4] Strang Ch. 11.2):

```text
xᵢⱼ_norm = (xᵢⱼ − minⱼ) / (maxⱼ − minⱼ)
```

If maxⱼ = minⱼ (constant column, already filtered out in feature extraction), set range = 1.0.

Normalization uses all N configs' statistics. Unmeasured configs are normalized using the same min/max, so all values stay in [0,1]. The same normalization applies to Fₘ and Fᵤ construction.

### 8.2 Linear basis

```text
f(x) = [1, x₁, x₂, ..., x_D]
p = 1 + D
```

Shape: Fₘ is (n, 1+D), Fᵤ is (N−n, 1+D).

The minimal viable model. Works well when one dimension dominates (e.g., core_num for D=1 kernels). Always in the candidate set.

### 8.3 Quadratic basis

```text
f(x) = [1, x₁, ..., x_D, x₁², ..., x_D²]
p = 1 + 2D
```

Shape: Fₘ is (n, 1+2D), Fᵤ is (N−n, 1+2D).

Captures curvature but no interactions. Useful for kernels where block sizes have sweet spots (too small = overhead, too large = waste). The squared term x² captures U-shapes that linear terms miss.

### 8.4 Full polynomial basis (with cross terms)

```text
f(x) = [1, x₁, ..., x_D, x₁², ..., x_D², x₁x₂, x₁x₃, ..., x_{D-1}x_D]
p = 1 + 2D + D(D−1)/2
```

Shape: Fₘ is (n, p), Fᵤ is (N−n, p). At D=6: p = 1 + 12 + 15 = 28.

The Config O/M workhorse. The cross-term xᵢxⱼ captures joint effects: "Y0BLOCK matters more when core_num is high."

Reference: [6] Box & Draper (2007). Response Surfaces, Mixtures, and Ridge Analyses, 2nd ed. Second-order polynomial models are the standard tool in response surface methodology.

### 8.5 Fourier basis (new in exp59)

```text
f(x) = [1, x₁, sin(πx₁), cos(πx₁), sin(2πx₁), cos(2πx₁),
            x₂, sin(πx₂), cos(πx₂), sin(2πx₂), cos(2πx₂),
            ...,
            x_D, sin(πx_D), cos(πx_D), sin(2πx_D), cos(2πx_D)]
p = 1 + 5D
```

Shape: Fₘ is (n, 1+5D), Fᵤ is (N−n, 1+5D). At D=6: p = 31.

### 8.5.1 General theory: Fourier series

Any periodic function f(x) on [0,1] can be represented as a sum of sines and cosines:

```text
f(x) = a₀ + Σₖ [aₖ cos(2πkx) + bₖ sin(2πkx)]
```

Reference: [14] Tolstov (1976). Fourier Series. Dover. Ch. 1: convergence of trigonometric series.

Truncating to the first two harmonics (k=1,2) gives a finite basis that captures the dominant periodic structure without overfitting to noise. The raw x term is included alongside the trigonometric terms to capture monotone trends that the periodic terms miss.

### 8.5.2 Our application

NPU performance is periodic in tile sizes — tiles aligned to cache line boundaries or SRAM bank boundaries show regular performance peaks. A tile of 256 might hit a sweet spot, 512 another, etc. Polynomial basis can't capture this without impractically high degree (need degree ≥ 4 to approximate two full periods). Fourier captures it with 5 terms per dimension.

Fourier wins on 8 of 35 kernels, including 3 hard kernels where polynomial fails. At D=6, p=31 — comparable to full polynomial (p=28) but orthogonal basis functions avoid collinearity issues that plague high-degree polynomials.

Higher harmonics (sin(3πx), cos(3πx)) were tested in exp64–65 and caused regressions due to near-collinearity with existing terms when data is sparse. Two harmonics is the empirical sweet spot.

### 8.6 Cubic polynomial basis (new in exp61)

```text
f(x) = [1,                              // intercept
         x₁, ..., x_D,                  // linear
         x₁², ..., x_D²,               // squared
         x₁x₂, ..., x_{D-1}x_D,       // 2-way cross
         x₁³, ..., x_D³,              // cubic
         x₁²x₂, x₁²x₃, ...,          // squared × linear (D(D−1) terms)
         x₁x₂x₃, ...]                 // 3-way cross (C(D,3) terms)
```

Parameter count:

```text
p_cubic = 1 + D + D + C(D,2) + D + D(D−1) + C(D,3)
       = 1 + D + D + D(D−1)/2 + D + D(D−1) + D(D−1)(D−2)/6
```

At D=3: p = 1+3+3+3+3+6+1 = 20. At D=6: p = 1+6+6+15+6+30+20 = 84.

Shape: Fₘ is (n, p_cubic), Fᵤ is (N−n, p_cubic).

The cubic basis captures 3-way interactions (xᵢxⱼxₖ) that matter for high-D kernels where the performance landscape has complex multi-parameter interactions. The x²y terms capture asymmetric interactions: "the curvature in Y0BLOCK depends on core_num."

### 8.7 Parameter count summary

| Basis | p formula | D=1 | D=2 | D=3 | D=4 | D=5 | D=6 |
|-------|-----------|-----|-----|-----|-----|-----|-----|
| linear | 1+D | 2 | 3 | 4 | 5 | 6 | 7 |
| quad | 1+2D | 3 | 5 | 7 | 9 | 11 | 13 |
| full | 1+2D+C(D,2) | 3 | 6 | 10 | 15 | 21 | 28 |
| fourier | 1+5D | 6 | 11 | 16 | 21 | 26 | 31 |
| cubic | see above | 4 | 11 | 20 | 35 | 56 | 84 |

---

## §9. Mode Selection via Spearman-Ranked LOOCV

**Component 7.** The core estimator. Runs every round on model path.

```text
inputs:  Xₘ_norm (n, D) — normalized measured features
         d (n,) — measured durations
         Xᵤ_norm (N−n, D) — normalized unmeasured features
         D — number of features
outputs: d̂ (N−n,) — predicted durations for unmeasured
         R̂² (scalar) — coefficient of determination of winning model
         best_mode (string) — name of winning basis type
feeds:   softmax batch selection (d̂)
         marginal voting (d̂ competes for batch allocation)
         convergence check (d̂ for margin criterion)
         temperature adaptation (R̂²)
```

This component replaces v6's fixed polynomial mode assignment. Instead of choosing a mode by n/p ratio, we compete up to 5 basis types via leave-one-out cross-validation, scored by Spearman rank correlation.

### 9.1 Step 1: Build candidate set

Always include: linear, quad, full, fourier. Include cubic only if p_cubic < n/2.

The n/2 gate for cubic is the identifiability condition — see §9.5.

### 9.2 Step 2: Fit each candidate model (ridge regression)

For each basis type b in the candidate set:

**Build feature matrix.** Construct Fₘ⁽ᵇ⁾ ∈ ℝ^(n × pᵇ) using the basis expansion from §8.

**Compute Gram matrix with ridge.**

```text
G⁽ᵇ⁾ = Fₘ⁽ᵇ⁾ᵀ Fₘ⁽ᵇ⁾ + λI
```

Shapes: Fₘ⁽ᵇ⁾ᵀ is (pᵇ, n), Fₘ⁽ᵇ⁾ is (n, pᵇ), so G⁽ᵇ⁾ is (pᵇ, pᵇ). λ = 0.1 for all bases (validated as globally Pareto-optimal in exp26).

**Invert Gram matrix.**

```text
G⁽ᵇ⁾⁻¹ = (Fₘ⁽ᵇ⁾ᵀ Fₘ⁽ᵇ⁾ + λI)⁻¹     shape: (pᵇ, pᵇ)
```

Adding λI guarantees invertibility even when features are collinear. Ridge bumps all eigenvalues of FᵀF up by λ, so 1/(σᵢ+λ) replaces 1/σᵢ — weak directions are clamped, strong directions barely change.

Reference: [3] Hoerl & Kennard (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. The original ridge regression paper.

**Solve for coefficients.**

```text
β̂⁽ᵇ⁾ = G⁽ᵇ⁾⁻¹ Fₘ⁽ᵇ⁾ᵀ d     shape: (pᵇ,)
```

Shapes walkthrough: G⁻¹ is (p, p), Fₘᵀ is (p, n), d is (n,). So Fₘᵀd is (p,) and G⁻¹(Fₘᵀd) is (p,). Each element β̂[k] corresponds to one basis feature.

### 9.2.1 General theory: the normal equation

Given n observations in feature matrix F (n × p) and response vector d (n × 1), ordinary least squares minimizes ||Fβ − d||². The solution is:

```text
β = (FᵀF)⁻¹ Fᵀd
```

F is rectangular (n × p, with n > p). You cannot invert a rectangular matrix. The trick: multiply both sides by Fᵀ (shape p × n):

```text
FᵀF β = Fᵀd
```

Now FᵀF is (p × p) — square. This is the Gram matrix. And Fᵀd is (p,) — the right-hand side.

What does FᵀF mean? Entry (i,j) = dot product of column i and column j of F across all n measured configs. It answers: "how much do basis features i and j overlap in the training data?" If two features are perfectly collinear, the Gram matrix is singular.

What does Fᵀd mean? Entry i = dot product of column i with the observed duration vector. It answers: "how much does this basis feature correlate with the thing we're trying to predict?"

Reference: [4] Strang (2006). Linear Algebra and Its Applications, 4th ed. Ch. 3 (Orthogonality and Least Squares).

Gauss-Markov theorem: among all linear unbiased estimators, OLS has the smallest variance. This justifies the normal equations as the starting point. Ridge regression deliberately introduces bias to reduce variance — trading the Gauss-Markov guarantee for lower total error when the system is ill-conditioned.

Reference: [4] Strang Ch. 4.3; [2] Hastie et al. Ch. 3.2.

### 9.2.2 General theory: ill-conditioning and ridge

FᵀF has an eigendecomposition (spectral theorem for symmetric matrices):

```text
FᵀF = QΛQᵀ     where Λ = diag(σ₁, σ₂, ..., σₚ)
```

Reference: [4] Strang Ch. 5. The spectral theorem guarantees real eigenvalues and orthogonal eigenvectors for symmetric matrices.

If any eigenvalue σᵢ is small, 1/σᵢ is huge — that direction in feature space gets amplified enormously. Example: σ₁=500, σ₂=300, σ₃=0.001. Then 1/σ₃ = 1000, dominating the solution.

Reference: [5] Golub & Van Loan (2013). Matrix Computations, 4th ed. Condition numbers and numerical stability.

Ridge regression adds λ to every eigenvalue:

```text
β = (FᵀF + λI)⁻¹ Fᵀd
```

In eigenvalue language: 1/(σᵢ+λ) replaces 1/σᵢ. Bias-variance tradeoff: ridge introduces bias (coefficients are systematically shrunk toward zero) but reduces variance (the solution is stable). For ranking unmeasured configs, low variance matters more than low bias — we need the ranking correct, not the absolute predictions.

Reference: [2] Hastie et al. Ch. 3.4.

### 9.2.3 Why λ=0.1 is globally optimal

Exp26 tested 15 adaptive λ strategies (discrete lookup by n/p ratio, continuous formulas, round-adaptive, fixed sweep). λ=0.1 is the only config with zero regressions across all 35 kernels.

The eigenvalue argument predicts D=6 needs stronger λ and D=1 could tolerate weaker. But D=1 kernels have only 5 discrete core_num values — a quadratic through 5 points overfits at low λ regardless of n/p. The discreteness confounds the n/p heuristic. Uniform λ=0.1 sits at the Pareto optimum.

### 9.3 Step 3: LOOCV via hat matrix

For each candidate basis b, compute leave-one-out cross-validation predictions without refitting.

#### 9.3.1 General theory: leave-one-out cross-validation

LOOCV removes one observation i, refits the model on the remaining n−1, and predicts the held-out observation. Repeating for all n gives n predictions. The LOOCV error is:

```text
LOOCV_MSE = (1/n) Σᵢ (dᵢ − d̂₋ᵢ)²
```

where d̂₋ᵢ is the prediction for observation i from the model trained without i.

Naively, this requires n separate regressions — prohibitively expensive. The hat matrix shortcut avoids all n refits.

Reference: [2] Hastie et al. Ch. 7.10: cross-validation. [15] Allen (1974). "The Relationship Between Variable Selection and Data Augmentation and a Method for Prediction." Technometrics 16(1):125–127. The original PRESS (Predicted Residual Sum of Squares) result.

#### 9.3.2 The hat matrix

The hat matrix H "puts a hat on" y — it maps observed values to fitted values:

```text
ŷ = Hy     where H = F(FᵀF + λI)⁻¹Fᵀ
```

Shapes: F is (n, p), (FᵀF+λI)⁻¹ is (p, p), Fᵀ is (p, n). So H is (n, n).

H is symmetric and (for ridge) satisfies 0 < Hᵢᵢ < 1. The diagonal element Hᵢᵢ is the leverage of observation i — how much it influences its own prediction. High leverage = the observation is unusual in feature space and strongly pulls the fit toward itself.

#### 9.3.3 The LOOCV shortcut

The key identity: the leave-one-out prediction for observation i equals:

```text
d̂₋ᵢ = dᵢ − eᵢ / (1 − Hᵢᵢ)
```

where eᵢ = dᵢ − d̂ᵢ is the ordinary residual. Equivalently:

```text
loocv_predᵢ = dᵢ − eᵢ / (1 − Hᵢᵢ)
```

Proof sketch: removing observation i from the training set changes the fit proportionally to how much i influences it (leverage Hᵢᵢ). The denominator (1 − Hᵢᵢ) corrects the residual for this leverage effect. When Hᵢᵢ is high (i is influential), the correction amplifies the residual — the LOO prediction moves further from dᵢ.

Reference: [2] Hastie et al. Ch. 7.10. The derivation follows from the Sherman-Morrison-Woodbury formula applied to the rank-1 update of removing one observation.

#### 9.3.4 Our application

In code:

```text
H = Fₘ @ G⁻¹ @ Fₘᵀ                          shape: (n, n)
h_diag = diag(H)                               shape: (n,)
residuals = d − Fₘ @ β̂                        shape: (n,)
denom = max(1 − h_diag, 0.01)                  shape: (n,), clamped for stability
loocv_pred = d − residuals / denom              shape: (n,)
```

The clamping at 0.01 prevents division by zero when a data point has leverage near 1.0 (it lies in a region of feature space where it's the only observation). Cost: one matrix multiply and element-wise operations — no refitting needed. Each basis type gets a vector of n LOOCV predictions at no extra cost beyond the single matrix inversion in step 2.

### 9.4 Step 4: Rank by Spearman

For each basis b, compute Spearman rank correlation between actual durations and LOOCV predictions:

```text
ρ_s⁽ᵇ⁾ = spearman(d, loocv_pred⁽ᵇ⁾)
```

The basis with the highest ρ_s wins.

#### 9.4.1 General theory: Spearman rank correlation

Spearman's ρ converts both variables to ranks, then computes Pearson correlation on the ranks:

```text
ρ_s = corr(rank(a), rank(b))
```

Equivalently (when no ties):

```text
ρ_s = 1 − 6·Σ(rₐ − r_b)² / [n(n²−1)]
```

where rₐ, r_b are the ranks. |ρ_s| close to 1 = perfect monotonic agreement. |ρ_s| near 0 = no monotonic relationship.

Reference: [13] Spearman (1904). The Proof and Measurement of Association Between Two Things. The original rank correlation paper. Also [2] Hastie et al. Ch. 14.7 for its use in feature screening.

#### 9.4.2 Why Spearman, not MSE (exp57–58)

The key insight: we only need the *ranking* of unmeasured configs to be correct for softmax selection to work, not the absolute predicted durations.

Example: model A predicts [100, 200, 300] for true durations [10, 20, 30] — terrible MSE (10× scale error), perfect ranking, correct softmax behavior (it selects the config predicted fastest, which IS the true fastest).

Model B predicts [15, 25, 12] for true [10, 20, 30] — much better MSE, but wrong ranking (predicts config 3 is fastest when config 1 actually is). Softmax follows the wrong ranking.

MSE penalizes scale errors that don't affect selection quality. Spearman measures exactly what matters — rank agreement. Exp57–58 showed switching from MSE to Spearman as the LOOCV criterion gives a clear improvement.

### 9.5 The identifiability gate for cubic (p < n/2)

The cubic basis only enters the competition when p_cubic < n/2. This prevents overfitting.

#### 9.5.1 General theory: identifiability

A model is identifiable when the data contains enough information to distinguish the true parameters from alternatives. With p parameters and n observations, the effective degrees of freedom for error estimation is n − p. When p > n/2, more than half the degrees of freedom are consumed by parameters — the model is fitting noise as much as signal.

Reference: [2] Hastie et al. Ch. 7.3: effective degrees of freedom and the bias-variance tradeoff.

#### 9.5.2 Our application

Without the gate, cubic wins the LOOCV competition on small n via overfitting: LOOCV residuals look small because the hat matrix diagonal Hᵢᵢ approaches 1.0 (each observation fully determines its own prediction), and the (1 − Hᵢᵢ) correction cannot fully compensate when almost all observations are high-leverage.

With the gate, cubic only enters when there's enough data for genuine 3rd-order structure to be distinguishable from noise. On our data: cubic enters and wins on 4 kernels where it captures real 3-way interactions, stays out on kernels where it would overfit.

Exp63b tested a tighter gate (p < n/3) — too restrictive, cubic never enters on medium-sized kernels where it helps. Exp63d tested applying the gate to all modes — hurts, because linear and quad naturally satisfy p < n/2 at our sample sizes. The gate is cubic-specific.

### 9.6 Winning model prediction

After the competition selects a winner:

```text
Fᵤ = build_features(Xᵤ_norm, D, best_mode)     shape: (N−n, p_best)
d̂ = Fᵤ @ β̂_best                                 shape: (N−n,)
```

R̂² is computed on training data for the winning model:

```text
SS_res = Σ(dᵢ − Fₘ @ β̂_best)²
SS_tot = Σ(dᵢ − d̄)²
R̂² = max(0, min(1, 1 − SS_res / (SS_tot + ε)))
```

where ε = 10⁻¹² for numerical safety. R̂² feeds into temperature adaptation (§10), not into mode selection.

### 9.7 Walkthrough: 3-parameter kernel (D=3, n=18, N=119)

**Round 1 after R1 (n=18 measured, 101 unmeasured):**

Build 4 candidates (cubic excluded: p_cubic=20, n/2=9, 20 > 9 → gated):

- linear: Fₘ is (18, 4), p=4, G is (4,4)
- quad: Fₘ is (18, 7), p=7, G is (7,7)
- full: Fₘ is (18, 10), p=10, G is (10,10)
- fourier: Fₘ is (18, 16), p=16, G is (16,16)

For each: solve β̂, compute H diagonal, get loocv_pred, compute ρ_s.

Suppose results: ρ_s(linear)=0.72, ρ_s(quad)=0.78, ρ_s(full)=0.81, ρ_s(fourier)=0.85.

Winner: fourier. Predict: d̂ = Fᵤ_fourier @ β̂_fourier, shape (101,).

**Round 3 (n=42 measured, 77 unmeasured):**

Now cubic enters: p_cubic=20, n/2=21, 20 < 21 → allowed.

5 candidates compete. Suppose ρ_s(cubic)=0.91, beats all others. Winner changes to cubic.

This is the key behavior: the LOOCV competition re-runs every round, and the winning basis can change as data accumulates.

---

## §10. Temperature Adaptation

**Component 8.** Always runs on model path.

```text
inputs:  R̂² (scalar) — from mode selection winning model
         k_est — number of needle-range configs measured so far
outputs: τ̂ (scalar) — softmax temperature
feeds:   softmax batch selection (τ̂)
```

```text
τ̂ = τ_min + τ_range × (1 − R̂²) × min(1, k_est / 4)
```

where τ_min = 0.3, τ_range = 0.4, k_est = |{j ∈ M : d($j$) ≤ 1.05 × min(d)}|.

Three interacting factors:

1. R̂² (model quality). Bad model (R̂² ≈ 0) → τ̂ high → explore widely. Good model (R̂² ≈ 1) → τ̂ low → exploit predictions.

2. k_est (needle rarity). k_est = estimated number of near-optimal configs found so far. k_est ≤ 3: min(1, k/4) < 1 → squashes the range toward τ_min for tight exploitation ("needles are rare, focus search"). k_est ≥ 4: factor hits 1.0, full R̂²-adaptation.

3. τ_min (floor). Even with a perfect model and rare needles, we keep some stochasticity to avoid deterministic dead ends.

This unified formula (exp49) replaced 3 separate if-branches with zero regressions.

Note: R̂² here comes from the winning basis model's training-set fit, not from the LOOCV Spearman score. Temperature adapts to how well the chosen model explains the data in absolute terms, not to cross-validated rank correlation.

---

## §11. Softmax Batch Selection

**Component 9a.** Always runs on model path.

```text
inputs:  d̂ (N−n,) — predicted durations from mode selection
         τ̂ (scalar) — temperature from adaptation
         batch_size — number of configs to select
         rng — random state
outputs: selected_indices (array of local indices into unmeasured set)
feeds:   measure function (configs to benchmark next)
```

### 11.1 General theory: softmax (Boltzmann) function

The softmax function converts a vector of scores s = [s₁, ..., sₘ] into a probability distribution:

```text
pᵢ = exp(sᵢ / τ) / Σⱼ exp(sⱼ / τ)
```

where τ > 0 is the temperature. τ → 0: concentrates on argmax (greedy). τ → ∞: uniform.

Reference: [7] Sutton & Barto (2018). Reinforcement Learning, 2nd ed. Ch. 2.3: softmax action selection in bandits. The temperature parameter comes from the Boltzmann distribution in statistical mechanics.

Numerical stability: subtract max(s) before exponentiating. Doesn't change probabilities but keeps exponents ≤ 0.

Reference: [8] Goodfellow, Bengio, Courville (2016). Deep Learning. Ch. 4.1: numerical stability of softmax.

### 11.2 Our application

Input: d̂ vector from §9 (one predicted duration per unmeasured config). We want to select configs with LOW duration.

```text
step 1: scores sᵢ = −d̂ᵢ                          (lower duration = higher score)
step 2: shift: sᵢ ← sᵢ − max(s)                  (numerical stability)
step 3: exponentiate: pᵢ = exp(sᵢ / τ̂)           shape: (N−n,)
step 4: clamp: pᵢ = max(pᵢ, 10⁻³⁰⁰)              (underflow floor)
step 5: normalize: pᵢ ← pᵢ / Σpⱼ                 shape: (N−n,), sums to 1
step 6: sample batch_size configs without replacement from this distribution
```

### 11.3 Underflow protection

When τ̂ is small and d̂ range is large, exp(−d̂/τ̂) can collapse to 0 for most configs. We clamp at 10⁻³⁰⁰ to prevent all-zero probability vectors. If partial underflow occurs (some but not all probabilities are non-zero), we use a hybrid approach: softmax selection for the non-zero pool, random fill for the zero pool.

### 11.4 Why softmax, not hard top-k

Hard top-k picks the same configs deterministically. If the model's ranking has a 10-position error, the batch misses the needle. Softmax gives the predicted-best HIGH probability but not certainty. Experiment 44 showed top-k was catastrophic: bmm_mul_4 dropped from 96% to 56%.

Reference: [9] Chapelle & Li (2011). Connection between softmax selection and Thompson sampling in bandits.

### 11.5 Batch allocation

If D < 3: softmax gets 100% of the batch. If D ≥ 3: softmax gets 50%, marginal voting (§12) gets the other 50%. The two halves are merged via set union, capped to batch_size.

---

## §12. Marginal Voting

**Component 9b.** Conditional — activates when D ≥ 3 (MARGINAL_D threshold).

```text
inputs:  X_measured (n, D) — raw (not normalized) features of measured configs
         d (n,) — measured durations
         X_unmeasured (N−n, D) — raw features of unmeasured configs
         D — number of features
outputs: marginal_scores (N−n,) — additive scores for unmeasured
feeds:   softmax selection with fixed τ = τ_min = 0.3
         merged into batch (50% allocation)
```

### 12.1 The factored additive model

```text
score(x) = Σ_{d=1}^{D} f_d(x_d)
```

where f_d(v) = mean measured duration of configs with feature d equal to v:

```text
f_d(v) = mean({d(j) : j ∈ M, X[j,d] = v})
```

If value v hasn't been seen in the measured set, f_d(v) = mean(d) (global mean fallback).

Shape: for each unmeasured config, sum D lookup values to get one score. Output: (N−n,) vector.

### 12.2 Why this complements the basis model

The polynomial/fourier/cubic models capture interactions (xᵢxⱼ) but need enough data to estimate cross-term coefficients. The marginal scorer captures "X1BLOCK=32 is always slow" without needing interactions — it's a strictly additive model. For high-D kernels where the basis model is underdetermined, the marginal path provides a safety net.

The marginal channel uses fixed τ = τ_min = 0.3 (always exploitative) because it has no model confidence to adapt — it's a simple lookup.

Experiment 36 showed this improved bmm_mul_4 from 88% to 94%.

---

## §13. Convergence Check

**Component 10.** Always runs on model path, after each round.

```text
inputs:  d (n,) — measured durations (from common data)
         d̂ (N−n,) — predicted durations (from mode selection)
         best_measured — min(d)
         prev_hist — histogram from previous round (or None)
outputs: stop (boolean) — whether to terminate
         curr_hist — current histogram (for next round)
feeds:   loop decision: stop → return best, continue → next round
```

Two independent early-stop criteria (either triggers stop):

### 13.1 Signal 1: Estimation margin

```text
est_margin = min(d̂_unmeasured) / min(d_measured)
```

If est_margin > 2.5, stop. The model predicts nothing unmeasured is even close to what we've found. Catches easy kernels where the optimum appears in R1.

Reads from: mode selection output (d̂) + common data (d).

### 13.2 Signal 2: Histogram L₁ distance

```text
bins = linspace(min(d), max(d), 11)
hist_now = histogram(d, bins=bins, density=True)    shape: (10,)
hist_now = hist_now / sum(hist_now)                  normalized
hist_L1 = Σᵢ |hist_now[i] − prev_hist[i]|
```

If hist_L1 < 0.01, less than 1% of probability mass shifted between rounds — the distribution of measured durations has stabilized.

Reads from: common data (d, measured durations only). Does NOT use the estimator.

### 13.3 General theory: L₁ distance (total variation)

For two discrete distributions P and Q over the same bins:

```text
d_L1(P, Q) = Σᵢ |Pᵢ − Qᵢ|
```

This equals twice the total variation distance. Intuitive meaning: total probability mass that "moved" between bins.

Reference: [10] Levin, Peres & Wilmer (2009). Markov Chains and Mixing Times. Ch. 4: total variation distance.

### 13.4 Why L₁, not L₂

L₁ measures total change. L₂ is dominated by the largest single-bin change. For "did the distribution change overall," L₁ is more informative. In our data (11 bins): L₁/L₂ ratio median 2.07 (range 1.42–3.05). L₁<0.01 catches 16 correct stops vs L₂<0.005's 14, both zero false.

### 13.5 Threshold selection

Thresholds found by brute-force sweep (exp25b): margin>2.5 OR L₁<0.01. Union: 118 correct stops, zero false, across 700 runs. Hard kernels correctly run all 5 rounds.

### 13.6 Where error metrics enter the flow

Per-round flow annotated with which metric is active:

1. R1: blind stratified sample. No model, no error metrics.
2. Mode selection: fit models, LOOCV → Spearman ρ_s (mode competition). Compute R̂² (diagnostic).
3. Temperature adaptation: τ̂ from R̂². R̂²'s influence ends here.
4. Batch selection: softmax(−d̂ / τ̂). Measure selected configs.
5. Convergence: (a) margin from d̂ and d. (b) histogram L₁ from d only.
6. If not converged → loop to step 2 with expanded M.

Summary: ρ_s lives in step 2 (mode competition). R̂² lives in steps 2–3. L₁ lives in step 5. They never interact directly.

---

## §14. R² Diagnostic

**Not a decision component.** Computed inside mode selection, used only by temperature adaptation.

```text
R̂² = 1 − Σ(dᵢ − d̂ᵢ)² / Σ(dᵢ − d̄)²
```

Computed on measured configs (in-sample), for the winning basis model. R̂² = 1.0: perfect fit. R̂² = 0.0: no better than predicting the mean.

Reference: [12] Draper & Smith (1998). Applied Regression Analysis, 3rd ed. Wiley.

Key values from our data: clone_5 (D=1) R̂²≈0.45, no_t_2 (D=2) R̂²≈0.05, add_layer_norm_0 (D=6) R̂²≈0.90. Softmax selection doesn't need accurate d̂ values, just approximately correct ranking. R̂²=0.45 still produces 90% catch.

We tested using R̂² to switch estimators in exp23 — it regressed. R̂² is diagnostic only.

---

## §15. The Feedback Loop

After measuring a batch, the algorithm loops back to mode selection (§9) with the expanded measured set. Key properties:

- The LOOCV competition re-runs every round — the winning basis can change as data accumulates (e.g., linear → fourier → cubic as n grows past the identifiability threshold)
- The cubic gate may open in later rounds when n crosses the p_cubic/2 threshold
- R̂² typically improves each round, pushing τ̂ down (more exploitation)
- Convergence criteria are re-checked every round — easy kernels exit early

The loop runs up to FASTA_MAX_ROUNDS=5 or until budget is exhausted or convergence triggers, whichever comes first.

---

## §16. Component Data Flow Summary

### Execution order per kernel

```text
[1] Feature Extraction
     X (N,D), names
         │
         ├──→ [2] Stratified R1 Sampling ──→ measure(r1_idx) ──→ d_r1
         ├──→ [3] Budget Allocation ──→ total_budget
         └──→ [4] Viability Assessment
                  │
                  ├── path='coverage' ──→ [5] Coverage Fallback ──→ return best
                  │
                  └── path='model' ──→ LOOP (up to 4 rounds):
                                         │
                                         ├──→ [6] Normalization + Basis Construction
                                         │        Fₘ (n,p) for each candidate basis
                                         │        Fᵤ (N-n,p) for winning basis
                                         │
                                         ├──→ [7] Mode Selection (§9)
                                         │        fit all candidates → LOOCV → Spearman
                                         │        outputs: d̂ (N-n,), R̂², best_mode
                                         │        │
                                         │        ├──→ [8] Temperature Adaptation
                                         │        │        τ̂ from R̂² and k_est
                                         │        │
                                         │        ├──→ [9a] Softmax Selection
                                         │        │        p = softmax(-d̂/τ̂)
                                         │        │        sample batch (50-100%)
                                         │        │
                                         │        ├──→ [9b] Marginal Voting (if D≥3)
                                         │        │        additive scores → softmax
                                         │        │        sample batch (0-50%)
                                         │        │
                                         │        └──→ [10] Convergence Check
                                         │                 margin from d̂, L₁ from d
                                         │                 │
                                         │                 ├── stop=True → return best
                                         │                 └── stop=False → measure batch
                                         │                                  expand M
                                         │                                  ↑ loop back
                                         └──────────────────────────────────┘
```

### Data shapes through the pipeline (D=3, N=119 example)

| Stage | Object | Shape | Notes |
|-------|--------|-------|-------|
| Feature extraction | X | (119, 3) | raw features |
| R1 sampling | r1_idx | (18,) | 15% of 119 |
| After R1 | Xₘ | (18, 3) | measured features |
| After R1 | Xᵤ | (101, 3) | unmeasured features |
| After R1 | d | (18,) | measured durations |
| Normalization | Xₘ_norm | (18, 3) | [0,1] scaled |
| Normalization | Xᵤ_norm | (101, 3) | [0,1] scaled |
| Basis (linear) | Fₘ | (18, 4) | p=4 |
| Basis (quad) | Fₘ | (18, 7) | p=7 |
| Basis (full) | Fₘ | (18, 10) | p=10 |
| Basis (fourier) | Fₘ | (18, 16) | p=16 |
| Basis (cubic) | — | gated out | p=20 > n/2=9 |
| Gram matrix | G | (p, p) | per candidate |
| Gram inverse | G⁻¹ | (p, p) | per candidate |
| Coefficients | β̂ | (p,) | per candidate |
| Hat matrix | H | (18, 18) | per candidate |
| Hat diagonal | h_diag | (18,) | leverage values |
| LOOCV pred | loocv_pred | (18,) | per candidate |
| Spearman | ρ_s | scalar | per candidate |
| Winner prediction | d̂ | (101,) | for unmeasured |
| R̂² | scalar | | winning model |
| τ̂ | scalar | | temperature |
| Softmax probs | p | (101,) | selection weights |
| Batch indices | sel | (~7,) | per round |
| After round 1 | d | (25,) | expanded |
| After round 1 | Xₘ | (25, 3) | expanded |
| After round 1 | Xᵤ | (94, 3) | shrunk |

### Input/output connectivity matrix

| Component | Reads from | Writes to |
|-----------|-----------|-----------|
| Feature extraction | configs | X, D, names |
| R1 sampling | X, r₁ size | r1_idx |
| Budget allocation | N, D, r₁ | total_budget |
| Viability | n, D | path (model/coverage) |
| Coverage fallback | unmeasured set | best_idx (terminates) |
| Basis construction | Xₘ_norm, Xᵤ_norm, mode | Fₘ, Fᵤ |
| Mode selection | Fₘ, d, Fᵤ for each candidate | d̂, R̂², best_mode |
| Temperature | R̂², k_est | τ̂ |
| Softmax selection | d̂, τ̂, batch_size | selected indices |
| Marginal voting | X_measured, d, X_unmeasured, D | marginal scores → indices |
| Convergence | d̂ (from estimator), d (measured), prev_hist | stop boolean |

---

## §17. Parameter Summary

All independently validated constants:

| Parameter | Value | Controls | Validated in |
|-----------|-------|----------|-------------|
| FASTA_R1_PCT | 0.15 | blind exploration fraction | exp25: 10% too aggressive, 25% wastes |
| R1_FLOOR_PCT | 0.10 | minimum R1 fraction | lower bound for small kernels |
| FASTA_BASE_BUDGET | 0.35 | total measurement budget | exp11: savings-catch sweet spot |
| FASTA_HIGH_BUDGET | 0.50 | budget for borderline kernels | exp45: closed last regressions |
| HIGH_D_THRESH | 4 | D threshold for high budget | exp47 |
| HIGH_NP_THRESH | 5 | n/p threshold for high budget | exp49: smooth function fails |
| MIN_VIABLE_NP | 2 | minimum n/p for any model | exp42: coverage gate |
| RIDGE_LAMBDA | 0.1 | regularization strength | exp26: global Pareto optimum |
| TAU_MIN | 0.3 | softmax τ̂ floor | exp44: greedy is catastrophic |
| TAU_RANGE | 0.4 | softmax τ̂ range | exp49: unified formula |
| MARGINAL_D | 3 | D threshold for marginal voting | exp36 |
| FASTA_MAX_ROUNDS | 5 | maximum R2 iterations | exp20c: peaks at 5 |
| MARGIN_CONV | 2.5 | convergence margin threshold | exp25b |
| L1_CONV | 0.01 | histogram L₁ threshold | exp27 |
| HIST_BINS | 11 | histogram resolution | exp22 |
| K_TOLERANCE | 1.05 | definition of "near-optimal" | exp15 |
| UNDERFLOW_FLOOR | 10⁻³⁰⁰ | softmax numerical safety | — |

The cubic identifiability gate (p < n/2) is not a tunable constant — it follows from the bias-variance tradeoff (§9.5).

---

## §A. API and Integration

### A.1 `fasta_algo` public functions

15 functions. All are pure math — no I/O, no logging, no framework dependencies. A production integration wraps these into a caller-driven loop (see A.2b).

**Feature preparation:**

| Function | Signature | Returns | Shape |
|----------|-----------|---------|-------|
| `extract_features(configs)` | list of config objects | X (N, D), names (list[str]) | scans kwargs, keeps numeric varying columns |
| `build_features(X, D, mode)` | dispatcher | F (len(X), p) | mode ∈ {linear, quad, full, fourier, cubic} |
| `build_poly_features(X, D, mode)` | linear/quad/full construction | F (len(X), p) | |
| `build_fourier_features(X, D)` | fourier construction | F (len(X), 1+5D) | |
| `build_cubic_features(X, D)` | cubic construction | F (len(X), p_cubic) | |
| `cubic_param_count(D)` | parameter count | int | used by identifiability gate |

**Sampling and viability:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `stratified_sample(X, n_sample, rng)` | most-unique-first R1 sampling | list of indices |
| `assess_viability(n_samples, D)` | binary gate: model vs coverage | (path, poly_mode) |

**Core estimator:**

| Function | Signature | Returns | Shape |
|----------|-----------|---------|-------|
| `select_mode_and_predict(Xm_norm, dm, Xu_norm, D)` | compete bases via Spearman LOOCV | (d̂, R̂², best_mode) | d̂: (len(Xu_norm),) |
| `numpy_spearman(a, b)` | Spearman rank correlation | float in [−1, 1] | returns 0.0 if n < 3 |

**Batch selection:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `softmax_select(scores, tau, batch_size, rng)` | softmax sampling without replacement | array of local indices |
| `marginal_scores(X_measured, d_measured, X_unmeasured, D)` | per-dim additive scoring | (len(X_unmeasured),) |

**Temperature and convergence:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_tau(r_sq, k_est)` | τ̂ from model quality + needle rarity | float |
| `should_stop(dm_array, best_measured, d_hat_unmeasured, prev_hist)` | margin + histogram L₁ | (stop: bool, hist: array) |

**Entry point:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `run(X, measure, rng)` | full algorithm loop | (best_idx, best_dur, n_rounds, n_measured, last_mode) |

`measure` is either a callable `measure(indices) → list[float]` or an ndarray (N,) with ground truth durations. When it's an array, `run()` wraps it as a lookup internally.

### A.2 What happens inside `run()`

`run()` drives the full loop: R1 → viability → model/coverage path → iterate → return best.

```python
run(X, measure, rng):

    phase 1 — setup (runs once)
        r1_size = max(15%·N, 10%·N), clamped to N
        p_lin = 1 + D
        if D ≥ 4 AND r1_size / p_lin < 5:
            total_budget = 50%·N
        else:
            total_budget = 35%·N
        r1_idx = stratified_sample(X, r1_size, rng)
        r1_durations = measure(r1_idx)                   # blind benchmark
        M = set(r1_idx), U = {0..N−1} − M
        measured_d = {idx: dur for r1}

    phase 2 — viability gate (runs once)
        path, _ = assess_viability(|M|, D)
        if path == 'coverage':
            sample remaining budget randomly from U
            measure those, return best                    # no model, done

    phase 3 — normalize (runs once)
        xmin, xmax per column across all N configs
        Xn = (X − xmin) / (xmax − xmin)                  shape: (N, D), all in [0,1]

    phase 4 — model loop
        for round 1..4:
            refine(state, latest_durations)               # see A.2b
        return best
```

In the target design, `run()` calls `refine()` internally — the same `refine()` that production uses. This eliminates duplicated loop logic:

```python
run(X, measure, rng):
    state = init_state(X, rng)              # phase 1–3: features, R1, normalize, viability
    batch_idx = state.r1_indices
    while batch_idx:
        durations = measure(batch_idx)
        batch_idx = refine(state, durations) # phase 4: one iteration
    return state.best_idx, state.best_dur, ...
```

`refine()` becomes a module-level function taking explicit state. Both `run()` and production call it. One code path for the algorithm.

### A.2b What happens inside `refine()`

`refine()` is one iteration of the model loop. `run()` calls it internally with durations from its `measure()` callback. A production integration calls it directly with compile+benchmark results. Same function, two callers.

```text
refine(state, durations):
    step 1 — absorb results
        for each (index, duration) in zip(state.last_batch, durations):
            add to M, remove from U
            update best
        n = |M|

    step 2 — terminal check
        if U is empty → return []

    step 3 — branch by path
        if path == 'coverage':
            pick remaining budget randomly from U → return final batch
            (one call, then done)

        if path == 'model':
            continue to step 4

    step 4 — model iteration
        if round ≥ FASTA_MAX_ROUNDS (5) → return []

        4a. rebuild from current M, U
            Xm = Xn[sorted(M)]                          shape: (n, D) — all measured so far
            dm = durations[sorted(M)]                    shape: (n,)
            Xu = Xn[sorted(U)]                           shape: (N−n, D)

        4b. compete basis models (§9)
            d̂, R̂², mode = select_mode_and_predict(Xm, dm, Xu, D)

            internally: build candidate set {linear, quad, full, fourier}
                        + cubic if p_cubic < n/2
            for each candidate:
                Fm = build_features(Xm, D, mode)         shape: (n, p)
                G = Fm'Fm + λI                           shape: (p, p)
                β̂ = G⁻¹ Fm' dm                          shape: (p,)
                H = Fm G⁻¹ Fm'                           shape: (n, n)
                loocv_pred = dm − residuals / (1 − diag(H))
                ρ_s = spearman(dm, loocv_pred)
            winner = argmax(ρ_s)
            d̂ = build_features(Xu, D, winner) @ β̂_winner

        4c. check convergence (§13)
            if should_stop(dm, best, d̂, prev_hist) → return []

        4d. compute batch_size from remaining budget
            batch_size = total_budget / FASTA_MAX_ROUNDS
            clamped to min(batch_size, |U|, remaining_budget)

        4e. select next batch (§11 + §12)
            τ̂ = compute_tau(R̂², k_est)
            if D ≥ 3:
                50% → softmax_select(d̂, τ̂)              model channel
                50% → softmax_select(marginal, τ_min)     marginal channel
                merge via set union, cap to batch_size
            else:
                100% → softmax_select(d̂, τ̂)

        return batch indices                              (or [] if budget exhausted)
```

The first `refine()` call receives R1 durations. Steps 4a–4e run for the first time, producing the first guided batch. Each subsequent call grows M (step 1), refits the model from scratch on all accumulated data (step 4a–4b), and selects the next batch. After at most 4 calls (rounds 2–5), or earlier if convergence fires or budget runs out, it returns `[]`.

Compilation failures: the caller passes `float('inf')` as the duration for configs that failed to compile. From `refine()`'s perspective, those are very slow configs that softmax won't select again.

### A.3 The growing measured matrix: what the estimator refits each round

The estimator trains on all accumulated measurements — Fₘ grows every round. Here is the concrete progression for a D=3, N=119 kernel:

| Round | Event | n (measured) | Fₘ shape | dm shape | Unmeasured | What's new in the training set |
|-------|-------|-------------|----------|----------|------------|-------------------------------|
| R1 | blind sample | 18 | (18, p) | (18,) | 101 | initial 15% stratified sample |
| R2-1 | 1st guided | 25 | (25, p) | (25,) | 94 | +7 softmax-selected configs |
| R2-2 | 2nd guided | 32 | (32, p) | (32,) | 87 | +7 more, biased toward predicted-fast |
| R2-3 | 3rd guided | 39 | (39, p) | (39,) | 80 | +7 more |
| R2-4 | 4th guided (or stop) | 42 | (42, p) | (42,) | 77 | +3 (budget exhausted at 35%) |

Each round, `select_mode_and_predict` receives `Xm = Xn[sorted(M)]` — all configs measured so far, not just the latest batch. The Gram matrix FₘᵀFₘ is built from all n rows. β̂ is solved from scratch — no incremental update, no warm-start from the previous round's coefficients.

The model at round R2-3 (n=39) sees the same 18 R1 configs that the model at round R2-1 (n=25) saw, plus 14 additional guided samples. R1 provides broad coverage of the parameter space. Guided samples fill in the fast region. Together they give the model both landscape shape and local precision near the optimum.

### A.4 Why train on measured only, and why earlier approaches failed

The regression trains on Fₘ (n rows, measured configs with known durations) and predicts on Fᵤ (N−n rows, unmeasured configs with unknown durations). We don't have durations for unmeasured configs, so they can't be training data.

Two things the estimator does use from the full N-config set:

1. **Normalization.** Min-max scaling to [0,1] uses all N configs' statistics. Measured and unmeasured share the same feature space.
2. **Prediction targets.** Fᵤ is built from the same basis expansion as Fₘ, applied to unmeasured configs' (known) tiling parameters. The features x are known for all N — only the durations d are unknown for unmeasured ones.

The tension is at low n. After R1 with D=6, we have n≈18 samples. The full polynomial has p=28 parameters — more unknowns than equations. Three approaches were tried:

**Approach 1: force the richest model, let ridge handle it (exp52d).** Always use full cross-terms from round 1 regardless of n/p. Tested with λ=0.1, λ=1.0, and decaying λ. Failed — at n=18, p=28, even strong regularization can't extract ranking signal from an underdetermined system. Ridge shrinks coefficients toward zero, which means predictions converge toward the mean duration. The model is "stable" (low variance) but predicts everything as roughly equal (high bias in ranking), which is useless for softmax selection.

**Approach 2: downgrade basis complexity based on n/p ratio (Config O/M).** Viability assessment picks linear when n/p < 2 for quad, quad when n/p < 3 for full. This works but is rigid — some kernels genuinely need full cross-terms even at moderate n/p, while others are better served by fourier even at high n/p. The n/p threshold is a proxy for model quality, and a coarse one.

**Approach 3 (current): compete multiple bases, let LOOCV decide (exp55→63c).** Don't choose the basis by n/p arithmetic. Fit all viable bases, measure each one's ranking accuracy via LOOCV, pick the winner. At n=18: linear (p=4, n/p=4.5) and quad (p=7, n/p=2.6) are well-conditioned and compete on ranking quality. Full (p=10, n/p=1.8) is marginal but enters and might win if cross-terms genuinely matter. Fourier (p=16, n/p=1.1) enters and might win if periodic structure dominates. Cubic (p=20) is gated out (20 > 18/2). By round R2-3 (n=39), cubic enters (20 < 39/2) and all 5 bases compete.

The key difference from approach 1: approach 1 forces one fixed complex model across all rounds. Approach 3 lets the data decide, per round, which complexity matches the available sample size.

### A.5 When and why the winning mode changes between rounds

The LOOCV competition re-runs every round. The winner can change for three reasons:

**1. Sample size crosses an identifiability threshold.** Cubic requires p < n/2. At D=3, p_cubic=20. After R1 (n=18), cubic is gated out (20 > 9). After round R2-2 (n=32), it enters (20 < 16). If cubic genuinely captures 3-way structure, it wins from that round onward. This is the most common mode switch — a basis that was absent becomes available.

**2. Measured set composition changes.** R1 is stratified — roughly uniform coverage. Guided rounds add configs biased toward the predicted-fast region (softmax selected those). A basis that captures broad periodic structure (fourier) might win on R1 data, while one that captures local curvature (full polynomial with interactions) might win once the fast-region detail fills in.

**3. LOOCV stability improves with more data.** With n=18, Spearman ρ_s has high variance — a basis might win by noise. At n=40, the 40-fold LOOCV produces a more stable ranking and the winner reflects genuine model quality.

In practice on 35 kernels: most settle on one mode by round 2 and stay. The ~5 kernels that switch are the ones where the landscape complexity depends on sample size, or where the cubic gate opens mid-run.

### A.6 Cost and risk of carrying 5 basis types

**Cost:** near zero. Each LOOCV evaluation is one matrix inversion O(p³) plus element-wise operations. At our sizes (p≤7 for linear, p≤31 for fourier, p≤84 for cubic when gated), the full 5-basis competition takes <0.5ms — less than a single config benchmark (~5ms).

**Risk — vote splitting.** If two similar bases are in the pool, they can split LOOCV scores, letting an inferior third basis win. This happened in exp65 (multiple fourier variants with different harmonic counts — near-collinear bases produced similar ρ_s scores, and the wrong one occasionally won). With 5 structurally distinct basis types, the LOOCV scores separate cleanly.

**Effect of removing modes:**

| Configuration | Catch rate | Regressions | What's lost |
|--------------|-----------|-------------|-------------|
| all 5 (exp63c) | 99.69% | 0 | — |
| drop cubic | ~99.4% | 0 | 4 high-D kernels with 3-way interactions |
| drop fourier | ~99.2% | 0 | 3 hard kernels with periodic cache-boundary effects |
| polynomial only (Config M) | 99.1% | 0 | combined: the kernels above |
| linear only | ~97% | several | most kernels need at least curvature or interactions |

The gains from each additional basis are small globally but concentrated on hard kernels — the cases where the algorithm is already closest to failure.

---

## References

[1] Rice, J.A. (2006). Mathematical Statistics and Data Analysis, 3rd ed. Duxbury/Thomson. Ch. 2: hypergeometric distribution.

[2] Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning, 2nd ed. Springer. Ch. 3.4.1: ridge regression. Ch. 7.3: bias-variance. Ch. 7.10: cross-validation. Ch. 14.7: rank correlation. Free: https://hastie.su.domains/ElemStatLearn/

[3] Hoerl, A.E. & Kennard, R.W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics 12(1):55–67.

[4] Strang, G. (2006). Linear Algebra and Its Applications, 4th ed. Thomson. Ch. 3 (Least Squares), Ch. 5 (Eigenvalues), Ch. 11.2 (Condition Numbers).

[5] Golub, G.H. & Van Loan, C.F. (2013). Matrix Computations, 4th ed. Johns Hopkins. Condition numbers and numerical stability.

[6] Box, G.E.P. & Draper, N.R. (2007). Response Surfaces, Mixtures, and Ridge Analyses, 2nd ed. Wiley. Second-order polynomial models.

[7] Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning, 2nd ed. MIT Press. Ch. 2.3: softmax action selection.

[8] Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press. Ch. 4.1: numerical stability of softmax.

[9] Chapelle, O. & Li, L. (2011). An Empirical Evaluation of Thompson Sampling. NIPS 2011:2249–2257.

[10] Levin, D.A., Peres, Y., Wilmer, E.L. (2009). Markov Chains and Mixing Times. AMS. Ch. 4: total variation distance.

[11] Bühlmann, P. & van de Geer, S. (2011). Statistics for High-Dimensional Data. Springer. Ridge in the p > n regime.

[12] Draper, N.R. & Smith, H. (1998). Applied Regression Analysis, 3rd ed. Wiley. R² and coefficient of determination.

[13] Spearman, C. (1904). The Proof and Measurement of Association Between Two Things. American Journal of Psychology 15(1):72–101.

[14] Tolstov, G.P. (1976). Fourier Series. Dover. Ch. 1: convergence of trigonometric series.

[15] Allen, D.M. (1974). The Relationship Between Variable Selection and Data Augmentation and a Method for Prediction. Technometrics 16(1):125–127. The original PRESS statistic / LOOCV shortcut.
