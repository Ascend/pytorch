# ENABLE\_PARALLEL\_SCHEDULER

## 功能描述

通过此环境变量可控制是否启用并行调度器FX pass。启用后，`torch_npu`在图优化阶段按算子类型（Cube/Vector）对调度节点分组重排，使矩阵乘类（Cube）与向量类（Vector）算子切分到不同流水分组，提升计算与通信、大算子与小算子间的并行度。

- 默认值为`false`：不启用并行调度器。
- 配置为`true`：启用并行调度器。
- 配置为“1”等其他值：不启用并行调度器，仅`true`（大小写不敏感）生效。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 当参与调度的节点总数不超过内置阈值（默认20个）时，不执行并行调度，仍按原有调度执行。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ENABLE_PARALLEL_SCHEDULER=true
```

## 使用约束

仅对NPU wrapper代码生成（NPUPythonWrapperCodeGen）生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Ascend 950DT</term>
