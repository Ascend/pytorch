# NPU\_ASD\_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:33.493Z pushedAt=2026-06-16T03:14:22.260Z -->

## Feature Description

This environment variable controls whether to enable the feature value detection feature of Ascend Extension for PyTorch. For details about this feature, see the "[Feature Value Detection](../framework_feature_guide_pytorch/feature_value_detection.md)" section in the *PyTorch Feature Guide*.

This environment variable supports the following optional parameters:

- `enable`: optional configuration, true or false. Default value is false. Whether to enable feature value detection.
- `with_checksum`: optional configuration, true or false. Default value is false. Whether to enable the checksum linkage function.
- `cooldown`: positive integer, minimum value is 1, default value is 5, unit: minutes. Cooldown suppression time window, the single activation time window for checksum linkage. Configure as needed.
- `strikes_num`: positive integer, minimum value is 1, default value is 3. Three strikes and out anomaly count limit. Configure as needed.
- `strikes_window`: positive integer, minimum value is 1, default value is 480, unit: minutes. Three strikes and out detection time window. Configure as needed.
- `checksum_cooldown`: positive integer, minimum value is 1, default value is 180, unit: minutes. Checksum linkage cooldown time window. Configure as needed.
- `upper_thresh1`: positive integer, minimum value is 3, default value is 1000000. Level-1 threshold. A feature value exceeding this absolute threshold is considered a gradient anomaly. The default detection threshold requires no configuration. If the threshold needs to be modified, it can be changed by configuring the environment variable.
- `upper_thresh2`: positive integer, minimum value is 3, default value is 100. Secondary threshold. A feature value exceeding the secondary threshold is considered a suspected anomaly and will not be updated into the historical mean. The default detection threshold requires no configuration. If you need to modify the threshold, you can do so through this environment variable.
- `grad_sample_interval`: positive integer, minimum value is 1, default value is 3. The interval for gradient detection, specifying how many gradients to skip between each detection. A smaller configuration yields a higher detection rate, but performance degradation will be more severe, potentially exceeding 2%.

## Configuration Example

```bash
export NPU_ASD_CONFIG=enable:true,with_checksum:true,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180,upper_thresh1:1000000,upper_thresh2:100,grad_sample_interval:3
```

## Usage Constraints

- This environment variable is not supported in PyTorch graph mode (TorchAir) scenarios.
- Feature value detection requires computing statistical values of activation gradients, which incurs additional memory usage. There may be an extra memory consumption of up to 1.5 GB. If the user's memory is insufficient, this may lead to OOM (Out of Memory).
- This environment variable is applicable to Ascend Extension for PyTorch 7.1.0 and later versions. For Ascend Extension for PyTorch 7.0.0 and earlier versions, you can use [NPU\_ASD\_ENABLE](NPU_ASD_ENABLE.md) to enable feature value detection. For specific operations, refer to the corresponding version documentation of Ascend Extension for PyTorch.
- Currently, it can only identify gradient anomalies that occur during model training with data types of **BF16** or **FP32**.
- The checksum linkage only supports the **BF16** data type.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
