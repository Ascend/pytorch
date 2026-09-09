# NPU\_ASD\_UPPER\_THRESH

## Feature Description

This environment variable configures the absolute threshold of the feature value detection function. The format is an integer data pair, with a minimum value of 3.

The first element controls the first-level threshold. When the absolute value of a feature value exceeds the first-level threshold, training is terminated and an alarm is reported. The second element controls the second-level threshold. When the absolute value of a feature value exceeds the second-level threshold and `ASCEND_GLOBAL_LOG_LEVEL` is set to `0`, `1`, or `2`, a Warning-level log is printed as an advance warning. Decreasing the threshold can detect abnormal data with smaller fluctuations and increase the detection rate, while increasing the threshold has the opposite effect.

The default threshold is 1000000,10000.

> [!NOTE]
>
> The factory default thresholds are the optimal values. You are advised not to modify them. If the following situations occur, adjust the thresholds based on the actual scenario and pay attention to the related impacts.
>
> - Scenarios requiring larger thresholds: if an alarm occurs and you confirm that the value fluctuation is normal and does not affect training, increase the thresholds.
>     - If `val` exceeds `NPU_ASD_UPPER_THRESH` and triggers an alarm, increase the `NPU_ASD_UPPER_THRESH` threshold based on the `val` value (recommended: `val`\*2).
>     - If the jump amplitude exceeds `NPU_ASD_SIGMA_THRESH` and triggers an alarm, increase the `NPU_ASD_SIGMA_THRESH` threshold based on the ratio of `(val-pre_val)` to `(max-min)` (recommended: `(val-pre_val)`/`(max-min)`\*2).
> 
>     Related impact: Increasing the thresholds causes the detection rate to decrease to some extent, but also reduces the false positive rate.
> - Scenarios requiring smaller thresholds: if frequent loss spikes or grad norm spikes affect training and spikes still occur after restarting but no alarm is triggered, gradually decrease the thresholds by a certain ratio (for example, 10).  
>     Related impact: Decreasing the thresholds can improve the detection rate, but it is also prone to causing false positives.

## Configuration Example

```bash
export NPU_ASD_UPPER_THRESH=1000000,10000
```

## Usage Constraints

- This environment variable is not supported in TorchAir scenarios.

- This environment variable is applicable to TorchNPU 7.0.0 and earlier versions.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
