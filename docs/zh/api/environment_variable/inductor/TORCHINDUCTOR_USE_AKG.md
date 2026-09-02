# TORCHINDUCTOR\_USE\_AKG

## 功能描述

通过此环境变量可配置torch.compile图模式（Inductor）下MLIR（Multi-Level Intermediate Representation）模式启用AKG（Auto Kernel Generator）后端优化。AKG基于Affine Dialect进行融合切分等调度优化，提升融合能力，降低调度和执行开销。

- 未配置或配置为`0`时：在启用MLIR的情况下，继续使用默认的MLIR优化流程。
- 配置为`1`时：在启用MLIR的情况下，使用AKG编译优化。

此环境变量默认配置为`0`。

## 配置示例

在启用MLIR后，再开启AKG：

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
export TORCHINDUCTOR_USE_AKG=1
```

保持MLIR默认优化流程：

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
export TORCHINDUCTOR_USE_AKG=0
```

## 使用约束

该环境变量仅在torch.compile图模式（Inductor）的MLIR模式下生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
