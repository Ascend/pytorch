# TORCHINDUCTOR\_NPU\_BACKEND

## 功能描述

通过此环境变量可配置图模式（Inductor）下的后端优化策略，支持在Triton、MLIR、DVM等模式之间切换。

-   配置为"default"或未配置时：使用默认的Triton模式。
-   配置为"mlir"时：使用MLIR模式。
-   配置为"dvm"时：使用DVM模式。


此环境变量默认配置为"default"。

## 配置示例

使用默认的Triton模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="default"
```

使用MLIR模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="mlir"
```

使用DVM模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="dvm"
```

## 使用约束

-   此环境变量必须在`import torch`之前设置，否则不生效。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
