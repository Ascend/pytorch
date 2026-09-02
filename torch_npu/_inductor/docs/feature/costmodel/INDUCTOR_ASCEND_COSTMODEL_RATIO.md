# INDUCTOR_ASCEND_COSTMODEL_RATIO

## 功能

控制CostModel预筛选后保留的config比例，默认值为0.25。

启用CostModel后，Inductor-Ascend会根据CostModel返回的预测耗时对候选config排序，并保留耗时最短的一部分config进入后续precompile流程。该变量用于控制保留比例，取值范围为(0, 1]。当取值小于等于0或大于1时，会回退到默认值0.25。

| 值 | 说明 |
|---|---|
| 未设置 | 使用默认值0.25 |
| (0, 1) | 按比例保留CostModel预测结果中耗时较短的config |
| 1 | 不进行CostModel预筛选 |

## 配置示例

```shell
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
export INDUCTOR_ASCEND_COSTMODEL_RATIO=0.25
```

## 使用约束

- 该变量仅在`INDUCTOR_ASCEND_ENABLE_COSTMODEL=1`时生效。
- 保留比例越小，后续编译和实测profiling的config数量越少，首次编译开销通常越低，但可能错过实际最优config。
- 如果CostModel筛选出的config均无法编译通过，Inductor-Ascend会使用被CostModel筛掉的config进行兜底编译。

## 支持型号

- <term>Atlas A5 系列产品</term>
