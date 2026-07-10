# Feature Value Detection

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:22:21.585Z pushedAt=2026-07-08T10:47:16.866Z -->

## Introduction

Feature value detection, fully named "online feature value detection during training," is primarily designed for long-duration stable training scenarios and targets high-bit flips. It performs anomaly detection based on gradient feature values to identify precision issues.

The detection principle and process are as follows:

1. During training, collect the tensors that need to be detected and compute their feature values.
2. The feature values are received by an asynchronous thread for anomaly detection.
3. "**Cooldown suppression**" is used to determine whether a new gradient anomaly has occurred. If so, "A grad-norm spike may happen" is printed once.

    Cooldown suppression: Since the current solution does not perform fault suppression, anomalous gradients on activation values can affect multiple parameters, leading to multiple anomalies on a single card within a short period. Therefore, a cooldown mechanism is introduced: for a certain period after the first anomaly, any new anomalies are treated as the same anomaly and are not counted toward the "three strikes and out" check. A single anomaly prints the "A grad-norm spike may happen" event log at the INFO level, which is disabled by default and can be enabled by setting "export TORCH_NPU_LOGS=silent".

    The "cooldown suppression time window" is controlled by the configuration item `cooldown`, with a default value of 5 minutes.

4. "**Three strikes and out**" is used to determine whether a feature value anomaly has occurred. If so, "feature detection detects abnormal results" is printed once.

    Three strikes and out: After cooldown suppression, if the event is confirmed as a new anomaly, it must further undergo the "three strikes and out" check to determine whether the total number of anomalies within the current three-strikes time window has reached the count limit. If the limit is reached, a feature value anomaly is determined to have occurred, and a Warning log reading "feature detection detects abnormal results" is printed.

    The "three strikes and out time window" is controlled by the configuration item `strikes_window`, with a default value of 480 min. The "three strikes and out anomaly count" limit is configured via the configuration item `strikes_num`, with a default value of 3.

5. When a feature value anomaly occurs, if `with_checksum` is configured as true, all cards are notified and "**checksum linkage**" is synchronously enabled.

    Checksum linkage: When a feature value anomaly is triggered by the three-strikes-and-out mechanism, if the checksum linkage function is enabled in the configuration, checksum linkage detection is synchronously enabled on all cards. Additional performance overhead is incurred during the detection period. The specific principle is to perform checksum API verification on the inputs and outputs of torch.matmul and torch.Tensor.matmul. If a verification result of True exists within the enabled time window, a Warning log reading "The result of Matmul checksum is abnormal" is printed.

    The "single checksum linkage enable time window" reuses the "cooldown suppression time window," with a default value of 5 min. Checksum is allowed to be enabled only once within the "checksum linkage cooldown time window." The "checksum linkage cooldown time window" is controlled by checksum\_cooldown, with a default value of 180 min.

## Use Cases

Detects whether gradient feature values are abnormal during model training.

## Usage Guide

Use the `NPU_ASD_CONFIG` environment variable to enable or disable feature value detection and the checksum linkage function.

- `enable`: Can be configured as true or false, with a default value of false. Whether feature value detection is enabled.
- `with_checksum`: Optional configuration, which can be set to true or false. The default value is false. Whether to enable the checksum linkage function.
- `cooldown`: Positive integer, with a minimum value of 1 and a default value of 5, in minutes. Cooldown suppression time window. A single checksum linkage activation time window reuses the cooldown suppression time window. Configure as needed.
- `strikes_num`: Positive integer, with a minimum value of 1 and a default value of 3. The anomaly count limit for the three strikes and out mechanism. Configure as needed.
- `strikes_window`: Positive integer, with a minimum value of 1 and a default value of 480, in minutes. The time window for the three strikes and out mechanism. Configure as needed.
- `checksum_cooldown`: Positive integer, with a minimum value of 1 and a default value of 180, in minutes. The checksum linkage cooldown time window. Configure as needed.
- `upper_thresh1`: positive integer, minimum value 3, default value 1000000. The first-level threshold. If the feature value exceeds this absolute threshold, it is considered a gradient anomaly. The default detection threshold requires no configuration; if the threshold needs to be modified, it can be changed by configuring this environment variable.
- `upper_thresh2`: positive integer, minimum value 3, default value 100. The second-level threshold. If the feature value exceeds the second-level threshold, it is considered a suspected anomaly and will not be updated into the historical mean. The default detection threshold requires no configuration; if the threshold needs to be modified, it can be changed via this environment variable.
- `grad_sample_interval`: positive integer, minimum value 1, default value 3. The interval for gradient detection, indicating how many gradients are skipped between each detection. A smaller value yields a higher detection rate, but performance degradation becomes more severe and may exceed 2%.

For details on using this environment variable, refer to the "[NPU_ASD_CONFIG](../environment_variable_reference/NPU_ASD_CONFIG.md)" section in *Environment Variable Reference*.

## Usage Example

```shell
export NPU_ASD_CONFIG=enable:true,with_checksum:true,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180,upper_thresh1:1000000,upper_thresh2:100,grad_sample_interval:3
```

## Constraints

- Currently, only gradient anomalies occurring during model training with data types **BF16** or **FP32** can be identified.
- The checksum linkage function only supports the **BF16** data type.
