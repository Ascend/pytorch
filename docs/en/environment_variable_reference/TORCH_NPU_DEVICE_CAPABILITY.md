# TORCH\_NPU\_DEVICE\_CAPABILITY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:14.387Z pushedAt=2026-06-16T03:14:22.395Z -->

## Function Description

This environment variable configures the return value of `torch_npu.npu.get_device_capability()`. It is only used for compatibility with the native PyTorch `torch.cuda.get_device_capability()` interface and does not represent the actual capability of the NPU hardware.

- When not configured, `torch_npu.npu.get_device_capability()` returns `None`.

- When configured, `torch_npu.npu.get_device_capability()` returns the value of the environment variable TORCH\_NPU\_DEVICE\_CAPABILITY. The configuration format follows the major.minor format, for example, 8.0, 9.0.

This environment variable is not configured by default.

## Configuration Example

```bash
export TORCH_NPU_DEVICE_CAPABILITY=8.0
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
