# TORCHINDUCTOR\_NPU\_BACKEND

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:58.854Z pushedAt=2026-06-16T03:14:22.405Z -->

## Feature Description

This environment variable configures the backend optimization strategy in graph mode (Inductor), supporting switching between modes such as Triton, MLIR, and DVM.

- When set to `default` or not configured: uses the default Triton mode.
- When set to `mlir`: uses the MLIR mode.
- When configured as `dvm`: Uses DVM mode.

This environment variable is configured as `default`.

## Configuration Example

Using the default Triton mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="default"
```

Using MLIR mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
```

Using DVM mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="dvm"
```

## Usage Constraints

- This environment variable must be set before `import torch`, otherwise it will not take effect.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
