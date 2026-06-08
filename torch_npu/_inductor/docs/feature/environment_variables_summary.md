# 环境变量列表

本手册描述开发者在使用inductor-ascend过程中可使用的环境变量。

**表 1**  环境变量列表

| 环境变量类型 | 环境变量名称 | 简介 |
|---|---|---|
| CostModel | INDUCTOR_ASCEND_ENABLE_COSTMODEL | 控制是否启用CostModel预筛选，默认值为0 |
| CostModel | INDUCTOR_ASCEND_COSTMODEL_RATIO | 控制CostModel预筛选后保留的config比例，默认值为0.25 |
| 其他 | INDUCTOR_ASCEND_CHECK_ACCURACY | 开启triton后端精度对比工具，dump单算子用例 |
| 其他 | INDUCTOR_ASCEND_DUMP_FX_GRAPH | dump可执行的单算子用例，用于调试和问题排查 |
| 其他 | INDUCTOR_ASCEND_LOG_LEVEL | 设置Inductor-Ascend日志等级，控制日志输出的详细程度 |
| 其他 | TORCHINDUCTOR_NDDMA | 启用Triton-Ascend load随路转置能力 |
