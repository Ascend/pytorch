# PARALLEL\_SCHEDULER\_NODES\_MIN

## 功能描述

通过此环境变量可配置并行调度器分组的 Minimum Scheduler Node Count（最小调度节点数）。当Cube组或Vector组的节点数小于内部阈值（该环境变量的1/5）时，并行调度器放弃本次分组重排，保持原有调度顺序，避免为很小的算子组引入额外调度开销。

- 默认值为`20`，组内节点数少于4时不分组。
- 设置正整数：按所配置值的1/5作为分组下限。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ENABLE_PARALLEL_SCHEDULER=true
export PARALLEL_SCHEDULER_NODES_MIN=40
```

## 使用约束

需在导入`torch_npu`之前设置，且仅在`ENABLE_PARALLEL_SCHEDULER=true`时生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
