# TORCH\_NPUGRAPH\_GC

## 功能描述

通过此环境变量可控制图捕获模式（NPUGraph Capture）过程中是否主动触发Python GC（Garbage Collection）。

-   配置为"0"时，NPUGraph Capture不会主动触发Python GC。
-   配置为"1"时，NPUGraph Capture会主动触发Python GC。

默认值："0"。

## 配置示例

```
export TORCH_NPUGRAPH_GC=1
```

## 使用约束

-   TORCH\_NPUGRAPH\_GC环境变量读取依赖PyTorch模块，可配置为"0"或"1"，不建议配置其他值。
-   TORCH\_NPUGRAPH\_GC设置为"1"时，会导致NPUGraph Capture性能下降。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>