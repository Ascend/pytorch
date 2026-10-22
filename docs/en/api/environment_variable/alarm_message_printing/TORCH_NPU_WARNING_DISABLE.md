# TORCH\_NPU\_WARNING\_DISABLE

## Function Description

This environment variable configures whether to print TorchNPU warning messages.

- When not configured or the configured value is not 1, warning message printing is enabled, and warning messages are printed normally on the screen of the first node.
- When the configured value is 1, warning message printing is disabled, and warning messages are not printed on the screen.

This environment variable is not configured by default.

> [!CAUTION]  
>
> - Disables warning message printing. It only applies to TorchNPU warning messages and does not affect warnings in native PyTorch, third-party libraries, or user scripts.
> - This environment variable has exactly the same function as the deprecated [TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md). When both are configured, this environment variable takes precedence.

## Configuration Example

Disable warning message printing:

```bash
export TORCH_NPU_WARNING_DISABLE=1
```

Re-enable warning message printing:

```bash
unset TORCH_NPU_WARNING_DISABLE
```

## Usage Constraints

This environment variable takes effect only in PyTorch 2.1.0 and later versions.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2 inference server</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
