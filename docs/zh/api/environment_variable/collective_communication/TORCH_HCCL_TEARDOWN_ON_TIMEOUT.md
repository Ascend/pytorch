# TORCH\_HCCL\_TEARDOWN\_ON\_TIMEOUT

## 功能描述

当使用HCCL作为通信后端时，此环境变量用于控制特定异常（即经过WatchDog逐Work超时处理，且异步错误处理模式为TearDown或SkipCleanUp的异常）在关闭“一般重抛”后，是否继续重抛并导致进程退出。TorchNPU 2.15版本默认值为0。

- 配置为“0”：不额外启用这条超时重抛规则；是否退出仍取决于其他错误策略。

- 配置为“1”：启用这条规则。符合条件时，即使`TORCH_HCCL_RETHROW_HCCL_ERRORS=0`，WatchDog外层仍继续重抛超时异常。

**异步错误处理模式**在“一般重抛”关闭时，该变量对Work超时分支的影响:

| 异步错误处理模式 | 该变量为0 | 该变量为1 |
| --- | --- | --- |
| `NoHandling` | 不额外强制重抛| 不通过这条规则强制重抛|
| `TearDown` | 不额外强制重抛| 继续重抛符合条件的超时异常|
| `CleanUpOnly` | 不额外强制重抛| 不通过这条规则强制重抛|
| `SkipCleanUp` | 不额外强制重抛| 继续重抛符合条件的超时异常|

表格仅说明这条规则，不保证其他异常或心跳监控不会终止进程。

它不是任意超时的统一开关。建链超时、用户线程阻塞等待超时、直接进入其他异常路径的错误，不应都视为这条逐Work规则。

`TORCH_HCCL_RETHROW_HCCL_ERRORS`默认1，因此关闭该变量不等于关闭默认的故障退出。

> [!NOTE]
>
> 应在进程组创建前设置，建议在各Rank启动前统一配置。

## 配置示例

关闭一般重抛，但保留模式`SkipCleanUp`的Work超时重抛：

```bash
export TORCH_HCCL_ASYNC_ERROR_HANDLING=3
export TORCH_HCCL_RETHROW_HCCL_ERRORS=0
export TORCH_HCCL_TEARDOWN_ON_TIMEOUT=1
```

## 使用约束

该环境变量仅在TorchNPU 2.15.0及之后版本生效。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
