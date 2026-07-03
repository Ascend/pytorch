# NPU\_ASD\_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:35.044Z pushedAt=2026-06-16T03:14:22.267Z -->

## Function Description

This environment variable controls whether to enable the feature value detection feature of Ascend Extension for PyTorch.

- When not set or set to "0", it indicates that feature value detection is disabled. The default value of this environment variable is "0".
- When set to "1", it indicates that feature value detection is enabled, only abnormal logs are printed, and no alarm is generated.
- When set to "2", it indicates that feature value detection is enabled and an alarm is raised.
- When set to "3", it indicates that feature value detection is enabled, an alarm is raised, and process data is recorded in the device-side info-level log.

## Configuration Example

```bash
export NPU_ASD_ENABLE=2
```

## Usage Constraints

- This environment variable is not supported in PyTorch graph mode (TorchAir) scenarios.
- Feature value detection requires calculating the statistical values of activation gradients, which incurs additional memory usage. This may lead to OOM when memory is tight.
- This environment variable applies to Ascend Extension for PyTorch version 7.0.0 and earlier.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas inference series</term>
