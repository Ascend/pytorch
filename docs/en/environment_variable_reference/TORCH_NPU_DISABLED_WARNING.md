# TORCH\_NPU\_DISABLED\_WARNING

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:26.883Z pushedAt=2026-06-16T03:14:22.397Z -->

## Function Description

This environment variable configures whether to print warning messages of Ascend Extension for PyTorch.

- When not configured or the configured value is not 1, warning message printing is enabled, and warning messages are printed normally on the screen of the first node.
- When the configured value is 1, warning message printing is disabled, and warning messages are not printed on the screen.

This environment variable is not configured by default.

> [!CAUTION]  
>
> Disabling warning message printing only applies to warning messages from Ascend Extension for PyTorch. It does not affect warnings from native Torch, third-party libraries, or user scripts.

## Configuration Example

Disable warning message printing:

```bash
export TORCH_NPU_DISABLED_WARNING=1
```

Re-enable warning message printing:

```bash
unset TORCH_NPU_DISABLED_WARNING
```

## Usage Constraints

This environment variable takes effect only in PyTorch 2.1.0 and later versions.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
- <term>Atlas inference series</term>
