# Dynamic Filter Algorithm Parameters

This document describes the configuration parameters used in the dynamic filter algorithm for efficient kernels tiling config selection via adaptive surrogate modeling.

> **Note:** All FASTA_ parameters are environment-overridable using the `FASTA_<NAME>=value` syntax (e.g., `export FASTA_R1_PCT=0.5`).

---

## Sampling & Budget Parameters

### `FASTA_R1_PCT`

- **Type:** float
- **Default:** `0.3`
- **Description:** The percentage of configs to sample in the initial R1 stratified sample phase. Higher values improve initial filter accuracy and parallel compilation throughput, but leave less budget for R2 exploration. See [Tuning Guide](#tuning-guide) for trade-offs.

### `R1_FLOOR_PCT`

- **Type:** float
- **Default:** `0.10`
- **Description:** Minimum floor percentage for R1 sampling. Ensures at least this fraction of configs are sampled even if `FASTA_R1_PCT` calculation yields fewer.

### `FASTA_BASE_BUDGET`

- **Type:** float
- **Default:** `0.35`
- **Description:** Base measurement budget as a fraction of total configs (N). Used for kernels with moderate complexity. See [Tuning Guide](#tuning-guide) for trade-offs with other budget parameters.

### `FASTA_HIGH_BUDGET`

- **Type:** float
- **Default:** `0.4`
- **Description:** Higher measurement budget for small, low-dimensional kernels. Controls the total R2 exploration budget. Higher values increase exploration at the cost of more measurements. See [Tuning Guide](#tuning-guide).

### `FASTA_LOW_BUDGET`

- **Type:** float
- **Default:** `0.25`
- **Description:** Lower measurement budget for large, high-dimensional kernels. Trade-off between exploration and measurement cost. See [Tuning Guide](#tuning-guide).

### `R1_PCT_LOW`

- **Type:** float
- **Default:** `0.12`
- **Description:** R1 sampling percentage for low-budget kernels. Used when routing determines a kernel should use `FASTA_LOW_BUDGET`.

---

## Kernel Classification Thresholds

### `HIGH_D_THRESH`

- **Type:** int
- **Default:** `4`
- **Description:** Dimensionality threshold for classifying a kernel as high-dimensional. Kernels with D >= this value are considered high-dimensional.

### `HIGH_NP_THRESH`

- **Type:** int
- **Default:** `5`
- **Description:** Number of programs threshold for high-dimensional classification. Used in conjunction with `HIGH_D_THRESH`.

### `HARD_D_THRESH`

- **Type:** int
- **Default:** `3`
- **Description:** Dimensionality threshold for classifying a kernel as "hard". Small kernels with D < this threshold may get `FASTA_HIGH_BUDGET`.

### `HARD_N_THRESH`

- **Type:** int
- **Default:** `150`
- **Description:** Config count threshold for "hard" kernel classification. Small kernels with N < this threshold may get `FASTA_HIGH_BUDGET`.

### `EASY_D_THRESH`

- **Type:** int
- **Default:** `4`
- **Description:** Dimensionality threshold for classifying a kernel as "easy". Large kernels with D >= this threshold may get `FASTA_LOW_BUDGET`.

### `EASY_N_THRESH`

- **Type:** int
- **Default:** `150`
- **Description:** Config count threshold for "easy" kernel classification. Large kernels with N >= this threshold may get `FASTA_LOW_BUDGET`.

### `MIN_VIABLE_NP`

- **Type:** int
- **Default:** `2`
- **Description:** Minimum number of programs required for a kernel to be considered viable for optimization.

---

## Surrogate Model Parameters

### `TAU_MIN`

- **Type:** float
- **Default:** `0.3`
- **Description:** Minimum tau value for Spearman rank correlation. Used in model selection to determine if a surrogate model is useful.

### `TAU_RANGE`

- **Type:** float
- **Default:** `0.4`
- **Description:** Range for tau-based model selection. The algorithm considers models with tau in the range `[TAU_MIN, TAU_MIN + TAU_RANGE]`.

### `MARGINAL_D`

- **Type:** int
- **Default:** `3`
- **Description:** Marginal dimensionality threshold. Used to determine when to add higher-order terms to the surrogate model.

### `RIDGE_LAMBDA`

- **Type:** float
- **Default:** `0.1`
- **Description:** Ridge regression regularization parameter. Prevents overfitting in surrogate models by penalizing large coefficients.

---

## Convergence Parameters

### `FASTA_MAX_ROUNDS`

- **Type:** int
- **Default:** `2`
- **Description:** Maximum number of optimization rounds. Controls how many iterations of measure-model-propose cycles to perform. Increase when the algorithm struggles to converge and more R2 iterations are required. See [Tuning Guide](#tuning-guide).

### `MARGIN_CONV`

- **Type:** float
- **Default:** `2.0`
- **Description:** Margin threshold for convergence. When the best config's predicted margin exceeds this value, convergence is triggered.

### `L1_CONV`

- **Type:** float
- **Default:** `0.01`
- **Description:** L1 norm threshold for convergence. When the L1 norm of changes falls below this value, the algorithm converges.

---

## Numerical Stability Parameters

### `HIST_BINS`

- **Type:** int
- **Default:** `11`
- **Description:** Number of bins for histogram-based computations. Used in distribution estimation and binning operations.

### `K_TOLERANCE`

- **Type:** float
- **Default:** `1.05`
- **Description:** Tolerance multiplier for kernel count. Allows slight variations in the expected number of good kernels (e.g., K * 1.05).

### `UNDERFLOW_FLOOR`

- **Type:** float
- **Default:** `1e-300`
- **Description:** Minimum value to prevent numerical underflow. Used to avoid log(0) or division by zero in probability computations.

---

## Budget Routing Logic

The algorithm routes kernels to different budgets based on their characteristics:

| Kernel Type | Condition | Budget |
|-------------|-----------|--------|
| Small Low-D | D < `HARD_D_THRESH` AND N < `HARD_N_THRESH` | `FASTA_HIGH_BUDGET` |
| Large High-D | D >= `EASY_D_THRESH` AND N >= `EASY_N_THRESH` | `FASTA_LOW_BUDGET` |
| Moderate | Otherwise | `FASTA_BASE_BUDGET` |

---

---

## Tuning Guide

The dynamic filter algorithm operates in two phases:

1. **R1 Phase (Initial Sampling):** Compiles and measures a stratified sample of configs in parallel
2. **R2 Phase (Iterative Refinement):** Uses surrogate modeling to propose and evaluate new configs

### Key Trade-offs

The relationship between `FASTA_R1_PCT`, `FASTA_HIGH_BUDGET`, and `FASTA_MAX_ROUNDS` determines how the measurement budget is allocated:

```text
Total R2 budget = FASTA_HIGH_BUDGET - FASTA_R1_PCT
R2 batch size per iteration = (FASTA_HIGH_BUDGET - FASTA_R1_PCT) / FASTA_MAX_ROUNDS
```

Note: The total R2 budget is fixed (FASTA_HIGH_BUDGET - FASTA_R1_PCT), and FASTA_MAX_ROUNDS only controls how it's split across iterations.

| Parameter | Effect When Increased | Trade-off |
|-----------|----------------------|-----------|
| `FASTA_R1_PCT` | More configs compiled in parallel during R1; better initial filter accuracy | Less total budget for R2 exploration (fixed FASTA_HIGH_BUDGET - FASTA_R1_PCT) |
| `FASTA_HIGH_BUDGET` | More total budget for R2 exploration | Higher total measurement cost |
| `FASTA_MAX_ROUNDS` | Finer-grained exploration across more iterations (same total R2 budget split differently) | More iterations may slow convergence |

### Tuning Strategies

#### High Compilation Throughput (Faster Compile Phase)

```text
FASTA_R1_PCT = 0.4-0.5
FASTA_HIGH_BUDGET = 0.5-0.6
FASTA_MAX_ROUNDS = 2
```

- Higher R1 means more configs are compiled in parallel at `.compile()` time
- Better initial filter accuracy from more samples
- Total R2 budget: 0.5-0.6 - 0.4-0.5 = ~0.1-0.2
- Suitable when compilation time dominates

#### High Exploration (Better Kernel Discovery)

```text
FASTA_R1_PCT = 0.2-0.25
FASTA_HIGH_BUDGET = 0.45-0.5
FASTA_MAX_ROUNDS = 3-4
```

- Lower R1 leaves more total budget for R2 exploration (0.45-0.5 - 0.2-0.25 = ~0.2-0.3)
- More iterations allow finer-grained exploitation of promising regions
- Suitable when the optimal config is rare and needs extensive search

#### Convergence Issues

```text
FASTA_R1_PCT = 0.3
FASTA_HIGH_BUDGET = 0.4
FASTA_MAX_ROUNDS = 3-4
```

- Increase `FASTA_MAX_ROUNDS` when the algorithm struggles to converge (same total R2 budget of 10%, split across more iterations)
- More R2 iterations give the surrogate model more opportunities to refine predictions
- Monitor `MARGIN_CONV` and `L1_CONV` to detect convergence

### Default Configuration Rationale

The defaults (`FASTA_R1_PCT=0.3`, `FASTA_HIGH_BUDGET=0.4`, `FASTA_MAX_ROUNDS=2`) provide:

- ~30% of configs measured in R1 (fully parallel compilation)
- ~10% total for all R2 iterations combined (FASTA_HIGH_BUDGET - FASTA_R1_PCT)
- Total budget: ~40% of configs measured
- With FASTA_MAX_ROUNDS=3, the 10% R2 budget would be split: 5% per iteration
- Balanced trade-off between initial accuracy and iterative refinement

---
