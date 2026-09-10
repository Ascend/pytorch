# TORCHINDUCTOR\_ENABLE\_LAYERNORM\_V4

## 功能描述

通过此环境变量可控制是否启用LayerNormV4专属实现。启用后，宽度为512的LayerNorm在Welford路径下使用专属的LayerNormV4实现：归约epilogue融合进Welford kernel后，LayerNormV4更快且避免了SIMD归约与SIMT后处理的切分。

- 默认值为`0`，不启用LayerNormV4实现。
- 配置为`1`：启用LayerNormV4实现。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCHINDUCTOR_ENABLE_WELFORD=1
export TORCHINDUCTOR_ENABLE_LAYERNORM_V4=1
```

## 使用约束

需在导入`torch_npu`之前设置。生效需同时满足：A5系列芯片、`TORCHINDUCTOR_ENABLE_WELFORD=1`、输入dtype为`float16`或`bfloat16`，且归一化维度为512。

## 支持的型号

- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
