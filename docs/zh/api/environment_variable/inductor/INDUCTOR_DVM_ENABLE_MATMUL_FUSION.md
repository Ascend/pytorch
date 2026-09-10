# INDUCTOR\_DVM\_ENABLE\_MATMUL\_FUSION

## 功能描述

通过此环境变量可开启DVM MatMul template融合功能。开启后，Inductor在DVM模式（`TORCHINDUCTOR_NPU_BACKEND="dvm"`）下会将矩阵乘算子（`mm`、`bmm`、`addmm`、`baddbmm`）通过DVM template进行融合，生成高效的融合kernel，提升矩阵乘性能。

- 默认值为`0`，关闭DVM MatMul template融合。
- 配置为`1`：开启DVM MatMul template融合。

> [!NOTE]
>
> - 此环境变量为`torch_npu`特有。
> - 开启后会注册DVM mm template lowering，将矩阵乘算子融合为DVM template kernel。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

开启DVM MatMul template融合：

```bash
export INDUCTOR_DVM_ENABLE_MATMUL_FUSION=1
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 仅在DVM模式（`TORCHINDUCTOR_NPU_BACKEND="dvm"`）下生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
