# INDUCTOR\_ASCEND\_SYMBOLIC\_GROUP\_AUTOTUNE

## 功能描述

通过此环境变量可控制是否启用动态shape分组autotune（grouped autotune）。

开启后，对于存在动态shape轴的Triton kernel，Inductor-Ascend会按shape特征（如维度长度）将运行时shape划分到不同分组（bucket），每组使用代表shape进行一次autotune benchmark，运行时根据实际shape所在的分组选择对应config。相比每个shape单独autotune，可显著减少动态shape场景下的重复编译和调优开销，实现“一次调优，多种shape复用”。

- 默认值为“0”：关闭分组autotune。
- 配置为“1”、“true”或“yes”：开启分组autotune。

## 配置示例

```bash
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1
```

## 使用约束

- 仅对存在动态shape轴的Triton kernel生效，静态shape kernel的行为不受影响。
- 仅对pointwise、reduction、persistent_reduction模板kernel生效，参与调优的模板类型可通过`INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES`控制。
- 该变量当前为灰度开关。当分组计划不支持、或分组benchmark的显存占用超过预算（由`INDUCTOR_ASCEND_SYMBOLIC_GROUP_MAX_BENCHMARK_MEMORY_RATIO`控制）时，会自动回退到普通autotune流程。

## 支持的型号

<term>Ascend 950DT</term>
