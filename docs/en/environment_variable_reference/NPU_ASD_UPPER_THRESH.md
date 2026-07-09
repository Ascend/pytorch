# NPU_ASD_UPPER_THRESH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:53.369Z pushedAt=2026-06-16T03:14:22.286Z -->

## Function Description

This environment variable configures the absolute threshold for the feature value detection function. The format is an integer data pair, with a minimum value of 3.

The first element controls the first-level threshold. When the absolute value of a feature value exceeds the first-level threshold, training is terminated and an alarm is reported. The second element controls the second-level threshold. When the absolute value of a feature value exceeds the second-level threshold and `ASCEND_GLOBAL_LOG_LEVEL` is set to "0", "1", or "2", a Warning-level log is printed as an alert. Reducing the threshold can detect abnormal data with smaller fluctuations, increasing the detection rate, while increasing the threshold has the opposite effect.

The default threshold is 1000000,10000.

> [!NOTE]
>
> The factory default thresholds are the optimal values, and customer modification is not recommended. If the following situations occur, you can adjust the thresholds based on the actual scenario and pay attention to the related impacts.
>
> - Scenarios requiring threshold increase: If an alarm occurs and it is confirmed that the value fluctuation is normal and does not affect training, increase the threshold.
>     - If `val` exceeds `NPU_ASD_UPPER_THRESH` and triggers an alarm, increase the `NPU_ASD_UPPER_THRESH` threshold based on the `val` value (recommended: `val`*2).
>     - If the jump amplitude exceeds `NPU_ASD_SIGMA_THRESH` and triggers an alarm, increase the `NPU_ASD_SIGMA_THRESH` threshold based on the ratio of `(val-pre_val)` to `(max-min)` (recommended: `(val-pre_val)`/`(max-min)`*2).
> 
>     Related impact: Increasing the threshold will reduce the detection rate, but the false positive rate will also decrease.
> - Scenarios requiring threshold decrease: If loss spike/grad norm spike occurs frequently and affects training, and the spike persists after restarting but no alarm is triggered, gradually decrease the threshold by a certain ratio (for example, 10).  
>     Related impact: Decreasing the threshold can improve the detection rate, but it may also easily cause false positives.

## Configuration Example

```bash
export NPU_ASD_UPPER_THRESH=1000000,10000
```

## Usage Constraints

- This environment variable is not supported in PyTorch graph mode (TorchAir) scenarios.

- This environment variable is applicable to Ascend Extension for PyTorch 7.0.0 and earlier versions.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
