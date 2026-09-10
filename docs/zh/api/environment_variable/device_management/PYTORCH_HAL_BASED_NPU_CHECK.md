# PYTORCH\_HAL\_BASED\_NPU\_CHECK

## 功能描述

通过此环境变量可配置`torch_npu.npu.is_available()`使用的NPU可用性检测方式。

- 配置为“1”时：`torch_npu.npu.is_available()`通过HAL接口探测设备数量（与`torch_npu.npu.device_count()`采用相同路径）。
  - 动态加载`libascend_hal.so`。
  - 解析`ASCEND_RT_VISIBLE_DEVICES`环境变量。
  - 若HAL探测失败，则回退至Runtime方式检测。
- 配置为其他值或者未配置时：`torch_npu.npu.is_available()`通过Runtime接口查询设备数量。
  - 首次查询到的非零设备数将被缓存。
  - 有效降低NPU初始化前频繁调用`is_available()`的开销。

此环境变量默认未配置。

该变量的命名与取值语义设计上参考了PyTorch的[PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)，在功能模式和使用逻辑上保持一致，便于开发者迁移和理解。

## 配置示例

启用基于HAL的检测路径：

```bash
export PYTORCH_HAL_BASED_NPU_CHECK=1
```

## 使用约束

此环境变量需在首次调用`torch_npu.npu.is_available()`前设置。

该环境变量仅在值严格等于“1”时生效，若配置为其他值（如“true”、“on”等），均视为未启用，不触发相应功能。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
