# TORCHNPU_PRECOMPILE_THREADS

## 功能

控制多线程编译线程数量，默认为max(os.cpu_count() // max(TORCHINDUCTOR_COMPILE_THREADS, 2), 32)，即与社区编译进程数相关且不小于32，大于1时使用并发编译。

## 配置示例

```shell
export TORCHNPU_PRECOMPILE_THREADS=32
```

## 使用约束

无

## 支持型号

- <term>Atlas A2推理系列产品</term>
- <term>Atlas A3推理系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>
