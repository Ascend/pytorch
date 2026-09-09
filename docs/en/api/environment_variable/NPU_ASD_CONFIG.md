# NPU\_ASD\_CONFIG

## Feature Description

This environment variable controls whether to enable the feature value detection function of TorchNPU. For details about this function, see the "[Feature Value Detection](../framework_feature_guide_pytorch/feature_value_detection.md)" section in *Framework Features*.

This environment variable supports the following optional parameters:

- `enable`: configurable as `true` or `false`, with the default value `false`. Indicates whether feature value detection is enabled.
- `with_checksum`: configurable as `true` or `false`, with the default value `false`. Indicates whether the checksum linkage function is enabled.
- `cooldown`: positive integer, minimum value 1, default value 5, unit: minute. Cooldown suppression time window, that is, the time window for a single activation of the checksum linkage. Configure it as needed.
- `strikes_num`: positive integer, minimum value 1, default value 3. Limit on the number of anomalies in the three-strikes-and-out rule. Configure it as needed.
- `strikes_window`: positive integer, minimum value 1, default value 480, unit: minute. Detection time window of the three-strikes-and-out rule. Configure it as needed.
- `checksum_cooldown`: positive integer, minimum value 1, default value 180, unit: minute. Cooldown time window of the checksum linkage. Configure it as needed.
- `upper_thresh1`: positive integer, minimum value 3, default value 1000000. Level-1 threshold. A feature value exceeding the absolute threshold is considered a gradient anomaly. The default detection threshold does not need to be configured. If you need to modify the threshold, you can do so by configuring the environment variable.
- `upper_thresh2`: positive integer, minimum value 3, default value 100. Level-2 threshold. A feature value exceeding the level-2 threshold is considered a suspected anomaly and is not updated to the historical mean. The default detection threshold does not need to be configured. If you need to modify the threshold, you can do so through this environment variable.
- `grad_sample_interval`: positive integer, minimum value 1, default value 3. Interval of gradient detection, indicating how many gradients each detection covers. The smaller the configuration, the higher the detection rate, but the greater the performance degradation, which may exceed 2%.

## Configuration Example

```bash
export NPU_ASD_CONFIG=enable:true,with_checksum:true,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180,upper_thresh1:1000000,upper_thresh2:100,grad_sample_interval:3
```

## Usage Constraints

- This environment variable is not supported in TorchAir scenarios.
- Feature value detection requires computing the statistical values of activation value gradients, which incurs additional memory usage. The additional memory consumption may be up to 1.5 GB. If your memory is insufficient, OOM (Out of Memory) may occur.
- This environment variable is applicable to TorchNPU 7.1.0 and later versions. For TorchNPU 7.0.0 and earlier versions, you can use [NPU\_ASD\_ENABLE](NPU_ASD_ENABLE.md) to enable feature value detection. For specific operations, refer to the documentation of the corresponding TorchNPU version.
- Currently, it can only identify gradient anomalies that occur during model training with data types of **BF16** or **FP32**.
- The checksum linkage supports only the **BF16** data type.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term>
