# TORCHINDUCTOR\_MAX\_AUTOTUNE

## 功能描述

通过此环境变量可控制是否开启max autotune功能。开启后，Inductor会对候选config进行更全面的自动调优，以获取最优的kernel配置。

- 默认值为`0`，关闭max autotune。
- 配置为`1`：开启max autotune。

> [!NOTE]
>
> - 开启max autotune会增加编译时间，但可能获得更优的kernel性能。
> - 矩阵乘算子可调优的候选实现由`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`控制。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

开启max autotune：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
