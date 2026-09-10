# TORCHINDUCTOR\_MAX\_AUTOTUNE\_GEMM\_BACKENDS

## 功能描述

通过此环境变量可配置max autotune过程中矩阵乘（matmul）类算子参与调优的候选实现列表。此处的"实现"指matmul kernel的生成来源（如ATen算子库、Triton模板、Catlass模板），不同于`TORCHINDUCTOR_NPU_BACKEND`配置的编译模式（Triton、DVM等）。仅在`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

- 默认值为`"ATEN,TRITON,CPP"`，即尝试ATen算子库、Triton模板和CPP模板三类候选实现。
- 若需尝试Catlass模板，请在配置中添加`"CATLASS"`。

> [!NOTE]
>
> - 此环境变量以逗号分隔实现名称，不支持空格。
> - 添加`"CATLASS"`后，需同时配置`TORCHINDUCTOR_NPU_CATLASS_DIR`指定Catlass模板库路径。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

尝试CATLASS和ATen候选实现：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CATLASS,ATEN"
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 仅在`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
