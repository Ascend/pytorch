# TORCHINDUCTOR\_CATLASS\_ENABLED\_OPS

## 功能描述

通过此环境变量可配置Catlass模板库支持的矩阵乘类型算子列表。设置后，Inductor将对指定类型的matmul算子尝试使用Catlass模板库进行模板调优。

- 默认值为`"mm,addmm,bmm"`，即对`mm`、`addmm`、`bmm`算子启用Catlass。
- 可添加`grouped_mm`扩展支持范围。

> [!NOTE]
>
> - 此环境变量以逗号分隔算子名称，不支持空格。

该变量对应PyTorch的`TORCHINDUCTOR_CUTLASS_ENABLED_OPS`。配置方式一致，默认值为`"mm,addmm,bmm"`。

## 配置示例

扩展支持grouped_mm：

```bash
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 需同时配置`TORCHINDUCTOR_NPU_CATLASS_DIR`（指定Catlass模板库路径）和`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
