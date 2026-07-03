# TORCH\_NPUGRAPH\_GC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:48.575Z pushedAt=2026-06-16T03:14:22.401Z -->

## Feature Description

This environment variable controls whether to actively trigger Python GC (Garbage Collection) during graph capture mode (NPUGraph Capture).

- When configured as "0", NPUGraph Capture will not actively trigger Python GC.

- When configured as "1", NPUGraph Capture will actively trigger Python GC.

Default value: "0".

## Configuration Example

```bash
export TORCH_NPUGRAPH_GC=1
```

## Usage Constraints

- The `TORCH_NPUGRAPH_GC` environment variable is read by the PyTorch module and can be configured as "0" or "1". For other values, behavior varies across different PyTorch versions and may change in the future, so configuring them is not recommended.

    >       For versions prior to PyTorch 2.7.1, setting a value other than "0" or "1" will fall back to the default value "0".
    >       For PyTorch 2.7.1 and later versions, setting a value other than "0" or "1" will fall back to the default value "1".

- Setting TORCH\_NPUGRAPH\_GC to "1" will cause a performance degradation during NPUGraph Capture.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas inference series</term>
