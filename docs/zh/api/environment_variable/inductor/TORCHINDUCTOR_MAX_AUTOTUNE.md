# TORCHINDUCTOR\_MAX\_AUTOTUNE

## 功能描述

通过此环境变量可控制是否开启max autotune功能。开启后，Inductor会对候选config进行更全面的自动调优，以获取最优的kernel配置。

- 默认值为“0”：关闭max autotune。
- 配置为“1”：开启max autotune。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 开启max autotune会增加编译时间，但可能获得更优的kernel性能。
> - 矩阵乘算子可调优的候选实现由`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`控制。

该变量对应PyTorch的[TORCHINDUCTOR_MAX_AUTOTUNE](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)，配置方式一致。

## 配置示例

开启max autotune：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id2 -->
