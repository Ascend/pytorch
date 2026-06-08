# INDUCTOR_ASCEND_ENABLE_COSTMODEL

## 功能

控制是否启用Inductor-Ascend的CostModel预筛选功能，默认值为0。

开启后，Inductor-Ascend会在Triton后端precompile之前，为每个候选config生成TTIR，并调用Triton-Ascend的CostModel后端预测耗时。预测结果用于重排并筛选候选config，减少后续编译和实测profiling的数量。

| 值 | 说明 |
|---|---|
| 未设置、0、false、no | 关闭CostModel预筛选（默认值） |
| 1、true、yes | 开启CostModel预筛选 |

## 配置示例

```shell
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
```

## 使用约束

- 该功能仅影响Triton后端存在多个候选config的场景。config数量小于等于1时不会调用CostModel。
- CostModel用于预筛选config，不替代后续编译和实测profiling。最终可用config仍以precompile和后续autotune结果为准。
- 如果CostModel后端不可用、返回结果异常或没有有效预测结果，会跳过CostModel预筛选，继续使用原始config集合。
- 启用该功能需要当前环境已安装包含CostModel后端的Triton-Ascend包。

## 支持型号

- <term>Atlas A5 系列产品</term>
