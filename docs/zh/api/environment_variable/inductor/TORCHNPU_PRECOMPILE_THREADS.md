# TORCHNPU\_PRECOMPILE\_THREADS

## 功能描述

通过此环境变量可配置`torch_npu` Inductor的预编译线程数。预编译线程用于加速图编译过程中的并行编译任务。

- 默认值为`max(os.cpu_count() // max(compile_threads, 2), 32)`（且不低于32），其中`compile_threads`由`TORCHINDUCTOR_COMPILE_THREADS`控制。
- 设置正整数：使用指定的线程数；设置为非法值（非整数）时程序会记录异常日志并退出。

> [!NOTE]
>
> - 此环境变量在`torch_npu._inductor`模块初始化时读取。

PyTorch通过`TORCHINDUCTOR_COMPILE_THREADS`控制编译并发度。

> [!NOTE]
>
> TorchNPU在编译并发度配置之外，提供`TORCHNPU_PRECOMPILE_THREADS`单独控制预编译线程数。

## 配置示例

```bash
export TORCHNPU_PRECOMPILE_THREADS=16
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
