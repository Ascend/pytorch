# NPU\_ASD\_ENABLE

## 功能描述

通过此环境变量可控制是否开启Ascend Extension for PyTorch的特征值检测功能。

特征值检测功能具体参见《PyTorch 训练模型迁移调优指南》的“[特征值检测](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/trainingmigrguide/PT_LMTMOG_0024.html)”章节。

-   未设置或者设置为“0”时，表示关闭特征值检测。此环境变量默认值为“0”。
-   设置为“1”时，表示开启特征值检测，只打印异常日志，不告警。
-   设置为“2”时，表示开启特征值检测，并告警。
-   设置为“3”时，表示开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据。

## 配置示例

```
export NPU_ASD_ENABLE=2
```

## 使用约束

-   此环境变量不支持在PyTorch图模式（TorchAir）场景下使用。

-   特征值检测需要计算激活值梯度的统计值，会产生额外的显存占用，用户显存紧张情况下可能导致OOM。

-   此环境变量适用于Ascend Extension for PyTorch 7.0.0及之前版本。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

