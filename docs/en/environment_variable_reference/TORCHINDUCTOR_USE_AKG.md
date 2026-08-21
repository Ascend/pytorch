# TORCHINDUCTOR\_USE\_AKG

## Feature Description

This environment variable configures whether to enable AKG (Auto Kernel Generator) backend optimization in MLIR (Multi-Level Intermediate Representation) mode under the torch.compile graph mode (Inductor). AKG performs scheduling optimizations such as fusion and tiling based on Affine Dialect, improving fusion capability and reducing scheduling and execution overhead.

- When not configured or set to `0`: With MLIR enabled, the default MLIR optimization flow continues to be used.
- When set to `1`: With MLIR enabled, AKG compilation optimization is used.

This environment variable defaults to `0`.

## Configuration Example

After enabling MLIR, enable AKG:

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
export TORCHINDUCTOR_USE_AKG=1
```

Keep the default MLIR optimization flow:

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
export TORCHINDUCTOR_USE_AKG=0
```

## Usage Constraints

This environment variable takes effect only in the MLIR mode of the torch.compile graph mode (Inductor).

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
