# INDUCTOR\_ASCEND\_FX\_GRAPH\_CACHE

## 功能描述

通过此环境变量可配置Inductor降层时FX图缓存的目录路径。设置后，系统会将追踪的FX图缓存到指定目录，便于后续复用和分析。

- 默认值未设置，不缓存FX图。
- 设置路径：缓存FX图到指定目录。

PyTorch通过`TORCHINDUCTOR_FX_GRAPH_CACHE`控制FX图缓存。

> [!NOTE]
>
> 配置用途不同：TorchNPU的`INDUCTOR_ASCEND_FX_GRAPH_CACHE`用于指定独立的缓存路径。

## 配置示例

```bash
export INDUCTOR_ASCEND_FX_GRAPH_CACHE=/data/fx_graph_cache
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 需确保目标目录可写。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
