# TORCH\_NPU\_DISABLED\_WARNING

## 功能描述

通过此环境变量可配置是否打印Ascend Extension for PyTorch的告警信息。

-   未配置或配置值不为1时，开启告警信息打印，告警信息会正常打印在首节点的屏幕上。
-   配置值为1时，关闭告警信息打印，告警信息不会打印在屏幕上。

此环境变量默认未配置。

> [!CAUTION]  
> 关闭告警信息打印，仅针对Ascend Extension for PyTorch的告警信息，不会影响原生Torch、第三方库或用户脚本中的告警。

## 配置示例

关闭告警信息打印：

```
export TORCH_NPU_DISABLED_WARNING=1
```

重新开启告警信息打印：

```
unset TORCH_NPU_DISABLED_WARNING
```

## 使用约束

该环境变量仅在PyTorch2.1.0及以上版本生效。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 800I A2 推理产品</term>
-   <term>Atlas 推理系列产品</term>

