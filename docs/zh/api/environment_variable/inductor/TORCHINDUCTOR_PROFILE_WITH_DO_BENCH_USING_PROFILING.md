# TORCHINDUCTOR\_PROFILE\_WITH\_DO\_BENCH\_USING\_PROFILING

## 功能描述

通过此环境变量可控制autotune过程中是否使用profiling进行性能测量。开启后，Inductor会使用profiling获取更精确的kernel执行时间，而非基于事件计时的估算。

- 默认值为`0`，不使用profiling，基于事件计时估算性能。
- 配置为`1`：使用profiling测量kernel执行时间。

> [!NOTE]
>
> - 使用profiling可获得更精确的调优结果，但会增加autotune的时间开销。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

开启基于profiling的autotune：

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING=1
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 仅在`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
