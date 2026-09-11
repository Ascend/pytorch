# 概述

C++算子开发人员需要遵循指引完成自定义算子接入TaskQueue流水线的适配，确保自定义算子在启用TaskQueue时能够正确入队并获取性能提升。适配完成后，模型开发人员可通过 `TASK_QUEUE_ENABLE` 控制算子行为，其使用方式与内置算子保持一致。
在 [TaskQueue并行下发](taskqueue_parallel_delivery.md) 介绍 `TASK_QUEUE_ENABLE` 控制机制的基础上，提供自定义算子接入TaskQueue的两条路径：标准接入（基于`EXEC_NPU_CMD_EXT`，仅适用于aclnn）与底层接入（基于`RunOpApiV2`，适用于AscendC kernel或非aclnn CANN接口）。

## 使用场景

自定义算子根据其最终调用的接口分为两条接入路径：

- [标准接入：EXEC_NPU_CMD_EXT](standard_access.md)：专为aclnn系列算子适配TaskQueue流水线而设计，该接口封装了底层的`RunOpApiV2`函数，提供开箱即用的简易化调用方式，可快速完成自定义算子接入，减少开发复杂度，适配最简洁。

- [底层接入：RunOpApiV2](lower_layer_access.md)：该接入方式适用于AscendC kernel launch等非aclnn场景，直接调用`RunOpApiV2`函数，不经过封装，适用于对性能有更高要求、需手动管理资源（如stream、lambda生命周期）的场景，性能更可控。

| 接入路径 | 适用场景 | 关键差异 |
|---------|---------|---------|
| 标准接入：EXEC_NPU_CMD_EXT | 算子最终调用aclnn接口 | 宏自动托管stream与lambda捕获，适配最简洁 |
| 底层接入：RunOpApiV2| AscendC kernel launch / 非aclnn CANN接口 | 需手动管理stream、lambda生命周期，性能更可控 |

![接入路径选择](../../../figures/taskqueue_op_developer_overview_fig_01.png)
