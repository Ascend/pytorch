# TORCHINDUCTOR\_COMPILE\_THREADS

## 功能描述

通过此环境变量可配置并发编译的进程数量。Inductor在编译时会使用多个进程并行编译kernel，以加速编译过程。

- 默认值为`32`，使用32个进程进行并发编译。
- 配置为正整数：使用指定的进程数进行并发编译。

> [!NOTE]
>
> - 增大线程数可加速编译，但会增加内存占用。
> - 建议不超过机器CPU核心数。

该变量沿用PyTorch的同名环境变量，配置方式一致，默认值为`32`。

## 配置示例

```bash
export TORCHINDUCTOR_COMPILE_THREADS=16
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
