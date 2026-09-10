# TORCHINDUCTOR\_ENABLE\_WELFORD

## 功能描述

通过此环境变量可控制是否启用Welford算法计算方差与均值类归约。启用后，`torch.var_mean`、`torch.var`、`torch.std`等算子在Triton模式（`TORCHINDUCTOR_NPU_BACKEND="default"`）下使用Welford单遍算法lowering（SIMT Welford），将归约epilogue融合进Welford kernel，减少中间结果落盘，提升数值稳定性与性能。

- 默认值为`0`，使用默认的两步方差（two-step variance）或逐项展开路径。
- 配置为`1`：启用Welford lowering。

> [!NOTE]
>
> - 归约元素较少（小于`unroll_reductions_threshold`）时仍会走两步方差展开路径，不受此变量影响。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。PyTorch的Inductor默认展开方差计算。

## 配置示例

```bash
export TORCHINDUCTOR_ENABLE_WELFORD=1
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 仅在Triton模式（`TORCHINDUCTOR_NPU_BACKEND="default"`）下生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
