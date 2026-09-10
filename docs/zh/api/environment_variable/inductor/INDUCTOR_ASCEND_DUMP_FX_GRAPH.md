# INDUCTOR\_ASCEND\_DUMP\_FX\_GRAPH

## 功能描述

通过此环境变量可控制在Inductor降层时是否导出FX图。启用后，系统会在降层过程中将FX图导出为文件，便于开发者分析图优化和降层过程。

- 默认值为`False`，不导出FX图。
- 配置为`True`：导出FX图。

> [!NOTE]
>
> - 当`INDUCTOR_ASCEND_CHECK_ACCURACY`启用时，会自动启用FX图dump。
> - dump文件路径由`TORCHINDUCTOR_CACHE_DIR`或`INDUCTOR_ASCEND_FX_GRAPH_CACHE`控制。

PyTorch通过`torch._inductor.config.trace.enabled`和`torch._inductor.config.trace.log_url`控制FX图导出，TorchNPU通过此环境变量提供相关配置。

## 配置示例

```bash
export INDUCTOR_ASCEND_DUMP_FX_GRAPH=True
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
