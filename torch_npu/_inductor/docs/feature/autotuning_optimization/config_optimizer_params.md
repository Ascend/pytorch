# Config Optimizer Parameters

This document describes the configuration parameters used in the FASTA config optimizer, which performs NPU-aware pruning and diversity filtering on generated tiling configs.

> **Note:** All FASTA_ parameters are environment-overridable using the `FASTA_<NAME>=value` syntax (e.g., `export FASTA_CONFIG_OPTIMIZER=1`).

---

## Activation Parameter

### `FASTA_CONFIG_OPTIMIZER`

- **Type:** bool (string comparison)
- **Default:** `"0"` (disabled)
- **Description:** Master switch to enable/disable the config optimizer. Set to `"1"` to activate the optimization pipeline.

---

## Stage 2: NPU-Aware Pruning Parameters

### `MAX_CIRCLE_NUM`

- **Type:** int
- **Default:** `4`
- **Description:** Maximum allowed `circle_num` value. Configs with `circle_num > MAX_CIRCLE_NUM` are filtered out. This controls the maximum number of loop iterations required, helping avoid configs with excessive computational overhead.

### `MIN_SUB_NUMEL`

- **Type:** int
- **Default:** `32`
- **Description:** Minimum tile size in elements. Configs with `sub_numel < MIN_SUB_NUMEL` are filtered out. This ensures tiles are large enough to provide efficient computation on the NPU.

---

## Stage 3: Diversity Filter Parameters

### `MAX_CONFIGS`

- **Type:** int
- **Default:** `50`
- **Description:** Maximum number of configs to retain after diversity filtering. If more configs pass the pruning stages, they are capped at this value via uniform `sub_numel` sampling to maintain diversity.

---

## Processing Pipeline

The config optimizer runs in three stages:

```text
Generated Configs
       │
       ▼
┌─────────────────────────┐
│  Stage 2: NPU-Aware     │
│  Pruning                │
│  - circle_num <= 4      │
│  - sub_numel >= 32      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Stage 3: Diversity     │
│  Filter                 │
│  - Deduplicate by       │
│    (circle_num,         │
│    sub_block_pattern)   │
│  - Cap at MAX_CONFIGS   │
└───────────┬─────────────┘
            │
            ▼
   Optimized Configs
```

---

## Tuning Guidelines

| Parameter | When to Increase | When to Decrease |
|-----------|------------------|------------------|
| `MAX_CIRCLE_NUM` | When you need more aggressive tiling options | When NPU memory/compute constraints are tight |
| `MIN_SUB_NUMEL` | When larger tiles are more efficient | When you need finer-grained search space |
| `MAX_CONFIGS` | When you have more measurement budget | When measurement budget is limited |

**Note:** Increasing `MAX_CONFIGS` increases the total measurement budget required by the dynamic filter algorithm downstream.
