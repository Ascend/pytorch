# TORCH\_NPU\_DEVICE\_CAPABILITY

## 功能描述

通过此环境变量可配置`torch_npu.npu.get_device_capability()`的返回值，仅用于兼容原生PyTorch的`torch.cuda.get_device_capability()`接口，不代表NPU硬件实际能力。

- 未配置时，`torch_npu.npu.get_device_capability()`返回值为`None`。
- 配置时，`torch_npu.npu.get_device_capability()`返回值为环境变量`TORCH_NPU_DEVICE_CAPABILITY`的值，配置格式遵循major.minor格式，例如8.0、9.0。

此环境变量默认为未配置。

## 配置示例

```bash
export TORCH_NPU_DEVICE_CAPABILITY=8.0
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
