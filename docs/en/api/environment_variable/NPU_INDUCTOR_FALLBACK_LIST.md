# NPU_INDUCTOR_FALLBACK_LIST

## Feature Description

This environment variable specifies the list of operators that need to fall back to the native PyTorch implementation. When certain operators encounter issues during execution on the NPU backend or when the native operator implementation is required, you can configure them through this environment variable.

- The default value is `None`, meaning no operator fallback is performed.
- Specifying operators: a comma-separated list of operator names (for example, `aten.div,aten.add.Tensor`), and the corresponding operators fall back to the native PyTorch implementation.
- Setting to `allfallback`: all operators fall back to the native PyTorch implementation.

## Configuration Example

- Specify specific operators for fallback:

```python
import os
os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "aten.div,aten.add.Tensor"
```

- Specify all operators for fallback:

```python
import os
os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "allfallback"
```

## Usage Constraints

- This environment variable takes effect only in Inductor backend scenarios.
- Operator names must use the complete aten operator naming format (for example, `aten.div.Tensor` and `aten.add.Tensor`).
- Multiple operators are separated by commas. Spaces are not supported.
- The fallback operation causes the corresponding operators to lose NPU hardware acceleration capability. Use it with caution.
- You are advised to enable this only during debugging or troubleshooting. It is not recommended for production environments.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Ascend 950DT</term> (Supports only the Triton mode and DVM mode in the Inductor backend compiler and does not support the MLIR mode.)
