# INDUCTOR\_ASCEND\_COSTMODEL\_RATIO

## 功能描述

通过此环境变量可控制CostModel预筛选后保留的config比例。启用CostModel后，Inductor-Ascend会根据CostModel返回的预测耗时对候选config排序，并保留耗时最短的一部分config进入后续precompile流程。取值范围为(0, 1]。当取值小于等于0或大于1时，会回退到默认值“0.25”。

- 未配置时：使用默认值“0.25”，保留CostModel预测耗时最短的25%候选config进入后续precompile流程。
- 配置为(0,1)时：按比例保留CostModel预测结果中耗时较短的config。
- 配置为“1”时：不进行CostModel预筛选。

> [!NOTE]
>
> - 保留比例越小，后续编译和实测profiling的config数量越少，首次编译开销通常越低，但可能错过实际最优config。
> - 如果CostModel筛选出的config均无法编译通过，Inductor-Ascend会使用被CostModel筛掉的config进行兜底编译。

## 配置示例

```bash
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
export INDUCTOR_ASCEND_COSTMODEL_RATIO=0.25
```

## 使用约束

该变量仅在`INDUCTOR_ASCEND_ENABLE_COSTMODEL=1`时生效。

## 支持的型号

<!-- npu="950" id1 -->
<term>Ascend 950DT</term>
<!-- end id1 -->
