# INDUCTOR\_ASCEND\_ENABLE\_COSTMODEL

## 功能描述

通过此环境变量可控制是否启用Inductor-Ascend的CostModel预筛选功能。

开启后，Inductor-Ascend会在Triton后端precompile之前，为每个候选config生成TTIR，并调用Triton-Ascend的CostModel后端预测耗时。预测结果用于重排并筛选候选config，减少后续编译和实测profiling的数量。

- 配置为“0”、“false”、“no”或“未配置”时： 关闭CostModel预筛选（默认值）。
- 配置为“1”、“true”或“yes”时： 开启CostModel预筛选。

> [!NOTE]
>
> - CostModel用于预筛选config，不替代后续编译和实测profiling。最终可用config仍以precompile和后续autotune结果为准。
> - 如果CostModel后端不可用、返回结果异常或没有有效预测结果，会跳过CostModel预筛选，继续使用原始config集合。

## 配置示例

```bash
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
```

## 使用约束

- 该功能仅影响Triton后端存在多个候选config的场景。config数量小于等于1时不会调用CostModel。
- 启用该功能需要当前环境已安装包含CostModel后端的Triton-Ascend包。

## 支持的型号

<term>Ascend 950DT</term>
