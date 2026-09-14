# TORCHINDUCTOR\_MAX\_AUTOTUNE\_GEMM\_BACKENDS

## 功能描述

通过此环境变量可配置max autotune过程中矩阵乘（matmul）类算子参与调优的候选实现列表。此处的"实现"指matmul kernel的生成来源（如ATen算子库、Triton模板、Catlass模板），不同于`TORCHINDUCTOR_NPU_BACKEND`配置的编译模式（Triton、DVM等）。仅在`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

- 默认值为`"ATEN,TRITON,CPP"`：NPU上实际参与调优的为ATen算子库与Triton模板两类候选实现（`CPP`仅面向CPU设备）。
- 配置为逗号分隔的实现名称列表：自定义候选实现组合（可选项如下）。

候选实现可选项如下：

| 实现名称 | 说明 |
|:---|:---|
| `ATEN` | ATen算子库 |
| `TRITON` | Inductor中的Triton模板 |
| `CATLASS` | Catlass模板，TorchNPU特有，需配置`TORCHINDUCTOR_NPU_CATLASS_DIR` |

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 此环境变量以逗号分隔实现名称，不支持空格分隔，大小写不敏感。
> - 添加`"CATLASS"`后，需同时配置`TORCHINDUCTOR_NPU_CATLASS_DIR`指定Catlass模板库路径。

该变量对应PyTorch的[TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)，配置方式一致。

## 配置示例

尝试CATLASS和ATen候选实现：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CATLASS,ATEN"
```

仅使用Triton模板调优：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
```

## 使用约束

仅在`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

## 支持的型号

<term>Ascend 950DT</term>
