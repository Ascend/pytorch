# TORCHNPU_PRECOMPILE_THREADS

## 功能

设置预编译线程数。未设置时，默认值为`max(os.cpu_count() // max(compile_threads, 2), 32)`，其中`compile_threads`是Inductor的编译进程数。

## 配置示例

```shell
export TORCHNPU_PRECOMPILE_THREADS=32
```

## 使用约束

无

## 支持型号

- <term>Atlas A2 推理系列产品</term>
- <term>Atlas A3 推理系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>
