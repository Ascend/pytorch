## TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING （同社区）

## 功能描述

该环境变量与inductor整体的profiling环境变量保持一致，用于管理autotune过程中是否使用profiling。

"0"为不使用profiling，"1"为使用profiling。

默认配置为TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="0"。

## 配置示例

开启autotune过程中的profiling

```shell
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="1"
```

关闭autotune过程中的profiling

```shell
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="0"
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>