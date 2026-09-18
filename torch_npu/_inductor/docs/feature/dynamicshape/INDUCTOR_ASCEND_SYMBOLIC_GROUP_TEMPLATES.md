# INDUCTOR\_ASCEND\_SYMBOLIC\_GROUP\_TEMPLATES

## 功能描述

通过此环境变量可配置参与动态shape分组autotune的模板类型列表，多个模板以逗号分隔，默认值为`pointwise,reduction,persistent_reduction`。

设置`INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1`启用分组autotune后，仅列表中的模板kernel会走分组autotune流程，其余模板kernel仍走普通autotune流程，用于灰度控制参与分组调优的kernel范围。

- 未配置时：使用默认值"pointwise,reduction,persistent_reduction"，即三类模板kernel均参与分组autotune，相当于不限制范围。
- 配置为"pointwise"：逐元素类模板kernel参与分组autotune。
- 配置为"reduction"：归约类模板kernel参与分组autotune。
- 配置为"persistent_reduction"：persistent归约类模板kernel参与分组autotune。

## 配置示例

仅对pointwise模板kernel启用分组autotune：

```bash
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES=pointwise
```

## 使用约束

- 该变量仅在`INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1`时生效。
- 取值需为`pointwise`、`reduction`、`persistent_reduction`的任意组合，其他取值会被忽略。
- 该变量当前为灰度开关，用于控制分组autotune的rollout范围。

## 支持的型号

<term>Ascend 950PR&950DT系列产品</term>
