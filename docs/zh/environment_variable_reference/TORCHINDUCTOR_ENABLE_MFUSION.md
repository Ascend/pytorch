# TORCHINDUCTOR_ENABLE_MFUSION

## 功能描述

通过此环境变量可控制是否启用 MFusion 融合优化功能。MFusion 是一种针对 NPU 平台的图融合优化技术，能够自动融合多个算子为单一kernel，从而减少数据传输开销，提升整体计算性能。

- 配置为"0"或未配置时：禁用 MFusion 融合优化。
- 配置为"1"时：启用 MFusion 融合优化。

此环境变量默认配置为"0"。

## 配置示例

禁用 MFusion（默认行为）：

```bash
export TORCHINDUCTOR_ENABLE_MFUSION="0"
```

启用 MFusion：

```bash
export TORCHINDUCTOR_ENABLE_MFUSION="1"
```

## 使用约束

- 该功能仅在 torch.compile 图编译后端为 "Inductor" 生效。
- 该功能仅在 torch_npu 2.7.1 和 2.9.0 分支生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
