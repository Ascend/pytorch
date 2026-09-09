# NPU\_ASD\_ENABLE

## Function Description

This environment variable controls whether to enable the feature value detection feature.

- When not set or set to "0", it indicates that feature value detection is disabled. The default value of this environment variable is "0".
- When set to "1", it indicates that feature value detection is enabled, only abnormal logs are printed, and no alarm is generated.
- When set to "2", it indicates that feature value detection is enabled and an alarm is raised.
- When set to "3", it indicates that feature value detection is enabled, an alarm is raised, and process data is recorded in the device-side info-level log.

## Configuration Example

```bash
export NPU_ASD_ENABLE=2
```

## Usage Constraints

- This environment variable is not supported in TorchAir scenarios.

- Feature value detection requires calculating the statistical values of activation gradients, which incurs additional memory usage. This may lead to OOM when memory is tight.

- This environment variable applies to TorchNPU 7.0.0 and earlier.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
