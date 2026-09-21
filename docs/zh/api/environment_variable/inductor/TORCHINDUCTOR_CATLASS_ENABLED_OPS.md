# TORCHINDUCTOR\_CATLASS\_ENABLED\_OPS

## 功能描述

通过此环境变量可配置Catlass模板库支持的矩阵乘类型算子列表。设置后，Inductor将对指定类型的matmul算子尝试使用Catlass模板库进行模板调优。

- 默认值为`"mm,addmm,bmm"`：对`mm`、`addmm`、`bmm`算子启用Catlass。
- 配置为`"ALL"`：对全部支持的矩阵乘算子启用Catlass。
- 配置为空字符串：不对任何算子启用Catlass。
- 配置为逗号分隔的算子名称列表：对列表中的矩阵乘算子启用Catlass（可选项如下）。

可配置的算子名称如下：

| 算子名称 | 说明 |
|:---|:---|
| `mm` | 二维矩阵乘。 |
| `addmm` | 带偏置的二维矩阵乘。 |
| `bmm` | 批量矩阵乘。 |
| `grouped_mm` | 分组矩阵乘，TorchNPU特有。 |

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 此环境变量以逗号分隔算子名称，不支持空格分隔，大小写不敏感。
> - 需同时配置`TORCHINDUCTOR_NPU_CATLASS_DIR`（指定Catlass模板库路径）和`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

该变量对应PyTorch的[TORCHINDUCTOR_CUTLASS_ENABLED_OPS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)，配置方式一致。默认值不同：TorchNPU默认值为`"mm,addmm,bmm"`，PyTorch默认值为`all`。

## 配置示例

扩展支持grouped_mm：

```bash
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"
```

对全部支持的矩阵乘算子启用Catlass：

```bash
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="ALL"
```

## 使用约束

无

## 支持的型号

<!-- npu="950" id1 -->
<term>Ascend 950DT</term>
<!-- end id1 -->
