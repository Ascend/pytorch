# ENABLE\_PARALLEL\_SCHEDULER

## 功能描述

通过此环境变量可控制是否启用并行调度器FX pass。启用后，`torch_npu`在图优化阶段按算子类型（Cube/Vector）对调度节点分组重排，使矩阵乘类（Cube）与向量类（Vector）算子切分到不同流水分组，提升计算与通信、大算子与小算子间的并行度。

- 默认值为`false`，不启用并行调度器。
- 配置为`true`：启用并行调度器。

> [!NOTE]
>
> - 仅对NPU wrapper代码生成（NPUPythonWrapperCodeGen）生效。
> - 分组最小节点数由`PARALLEL_SCHEDULER_NODES_MIN`控制。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ENABLE_PARALLEL_SCHEDULER=true
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
