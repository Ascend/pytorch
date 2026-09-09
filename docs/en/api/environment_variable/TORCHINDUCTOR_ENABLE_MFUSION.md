# (beta) TORCHINDUCTOR\_ENABLE\_MFUSION

## Feature Description

This environment variable controls whether to enable the MFusion fusion optimization feature. MFusion is a graph fusion optimization technique for NPU platforms that can automatically fuse multiple operators into a single kernel, thereby reducing data transfer overhead and improving overall computational performance.

- When set to "0" or not configured: Disables MFusion fusion optimization.
- When set to "1": Enables MFusion fusion optimization.

This environment variable is set to "0" by default.

## Configuration Example

Disable MFusion (default behavior):

```bash
export TORCHINDUCTOR_ENABLE_MFUSION="0"
```

Enable MFusion:

```bash
export TORCHINDUCTOR_ENABLE_MFUSION="1"
```

## Usage Constraints

- This feature takes effect only when the torch.compile graph compilation backend is "Inductor".
- This feature takes effect only in PyTorch 2.7.1 and 2.9.0.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
