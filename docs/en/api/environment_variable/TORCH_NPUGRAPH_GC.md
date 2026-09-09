# TORCH\_NPUGRAPH\_GC

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

- The `TORCH_NPUGRAPH_GC` environment variable is read by the PyTorch module and can be configured as "0" or "1". For other values, behavior varies across different PyTorch versions and may change in the future. You are advised not to configure such values.
    >       For versions prior to PyTorch 2.7.1, setting a value other than "0" or "1" falls back to the default value "0".
    >       For PyTorch 2.7.1 and later versions, setting a value other than "0" or "1" falls back to the default value "1".

- Setting TORCH\_NPUGRAPH\_GC to "1" causes a performance degradation during NPUGraph Capture.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
