# TORCHINDUCTOR\_NPU\_BACKEND

## Feature Description

This environment variable configures the backend optimization strategy in graph mode (Inductor), supporting switching between modes such as Triton, MLIR, and DVM.

- When set to `default` or not configured: uses the default Triton mode.
- When set to `mlir`: uses the MLIR mode.
- When set to `dvm`: uses the DVM mode.

By default, this environment variable is configured as `default`.

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

This environment variable must be set before `import torch`. Otherwise, it does not take effect.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term> (Supports only the Triton mode and DVM mode in the Inductor backend compiler and does not support the MLIR mode.)
