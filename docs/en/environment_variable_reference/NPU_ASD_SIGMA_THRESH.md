# NPU_ASD_SIGMA_THRESH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:40.005Z pushedAt=2026-06-16T03:14:22.273Z -->

## Function Description

This environment variable configures the relative threshold for the feature value detection function. The format is an integer data pair, with a minimum value of 3.

The algorithm detects the jump amplitude of feature values. The first element controls the first-level threshold: when the feature value jump amplitude exceeds the first-level threshold, training is terminated and an alarm is reported. The second element controls the second-level threshold: when the feature value jump amplitude exceeds the second-level threshold and `ASCEND_GLOBAL_LOG_LEVEL` is set to "0", "1", or "2", a Warning-level log is printed as an alert. Decreasing the thresholds can detect abnormal data with smaller fluctuations, increasing the detection rate; increasing the thresholds has the opposite effect.

The default threshold is 100000,5000.

> [!NOTE]  
>
> The factory default thresholds are the optimal values, and users are not recommended to modify them. If the following situations occur, you can adjust the thresholds based on the actual scenario and pay attention to the related impacts.
>
> - Scenarios requiring larger thresholds: If an alarm occurs and you confirm that the value fluctuation is normal and does not affect training, increase the thresholds.
> - If the `val` exceeds `NPU_ASD_UPPER_THRESH` and triggers an alarm, increase the threshold `NPU_ASD_UPPER_THRESH` based on the `val` value (recommended: `val`\*2);
> - If the jump amplitude exceeds `NPU_ASD_SIGMA_THRESH` and triggers an alarm, increase the threshold `NPU_ASD_SIGMA_THRESH` based on the ratio of `(val-pre_val)` to `(max-min)` (recommended: `(val-pre_val)`/`(max-min)`*2).
>
> - Related impact: Increasing the thresholds will reduce the detection rate to some extent, but the false positive rate will also decrease.
> - Scenarios requiring smaller thresholds: If frequent loss spikes/grad norm spikes affect training, and spikes still occur after restarting but no alarm is triggered, gradually decrease the thresholds by a certain ratio (for example, 10).  
> - Related impact: Decreasing the thresholds can improve the detection rate, but it is also prone to causing false positives.

## Configuration Example

```bash
export NPU_ASD_SIGMA_THRESH=100000,5000
```

## Usage Constraints

- This environment variable is not supported in PyTorch graph mode (TorchAir) scenarios.

- This environment variable is applicable to Ascend Extension for PyTorch 7.0.0 and earlier versions.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
