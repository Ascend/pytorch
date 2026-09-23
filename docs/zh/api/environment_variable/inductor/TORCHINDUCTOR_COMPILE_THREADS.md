# TORCHINDUCTOR\_COMPILE\_THREADS

## 功能描述

通过此环境变量可配置并发编译的进程数量。Inductor在编译时会使用多个进程并行编译kernel，以加速编译过程。

- 默认值为`min(32, CPU核心数)`：并发编译进程数取32与机器CPU核心数的较小值。
- 配置为“1”：在主进程内串行编译，不启用编译子进程，可用于编译问题定位。
- 配置为大于1的正整数：使用指定的进程数进行并发编译。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 增大线程数可加速编译，但会增加内存占用。
> - 该变量无内置上限，配置过大会在首次编译时同时拉起过多编译子进程，可能导致内存耗尽或进程创建失败，建议不超过机器CPU核心数。
> - Ascend C模式下并发编译进程数上限为32。
> - 在NPU上，此环境变量还作为`TORCHNPU_PRECOMPILE_THREADS`默认值的推导输入（`CPU核心数 // max(该值, 2)`，且不低于32）。

该变量对应PyTorch的[TORCHINDUCTOR_COMPILE_THREADS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)，配置方式一致，默认值为`min(32, CPU核心数)`。

## 配置示例

```bash
export TORCHINDUCTOR_COMPILE_THREADS=16
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id2 -->
