# TORCHINDUCTOR\_NPU\_CATLASS\_DIR

## 功能描述

通过此环境变量可配置Catlass模板库的路径。Catlass是NPU平台上的矩阵乘模板加速库，Inductor在autotune时可通过Catlass模板库对matmul类算子进行模板调优。

- 默认值未配置，使用`torch`安装目录下`../third_party/catlass`路径（源码编译场景），若该路径不存在则不加载Catlass模板库。
- 设置有效路径：从配置的路径加载Catlass模板库。

> [!NOTE]
>
> - 若路径配置错误，会有WARNING信息提示，并自动跳过Catlass模板库。
> - 使用Catlass模板库需同时配置`TORCHINDUCTOR_MAX_AUTOTUNE=1`和`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`包含`"CATLASS"`。

该变量对应PyTorch的`TORCHINDUCTOR_CUTLASS_DIR`。

> [!NOTE]
>
> 模板库不同：PyTorch配置CUTLASS（CUDA Templates for Linear Algebra Subroutines）库路径，TorchNPU配置Catlass库路径。

## 配置示例

```bash
export TORCHINDUCTOR_NPU_CATLASS_DIR="/path/to/catlass/dir"
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 需确保路径有效且Catlass模板库已正确安装。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
