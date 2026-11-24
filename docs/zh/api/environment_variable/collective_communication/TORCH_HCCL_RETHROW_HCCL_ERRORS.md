# TORCH\_HCCL\_RETHROW\_HCCL\_ERRORS

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制WatchDog外层捕获标准C++异常（`std::exception`）后，是否继续重抛。对应异常来自Work检查、通信器预检查或WatchDog中的其他调用。TorchNPU 2.15版本默认值为1。

- 配置为“0”：不通过这个开关继续重抛对应的标准异常，WatchDog随后结束，不重新进入检查循环。

- 配置为“1”：继续重抛对应异常，默认会导致训练进程异常终止。

该变量不能保证进程不退出。启用[TORCH_HCCL_TEARDOWN_ON_TIMEOUT](TORCH_HCCL_TEARDOWN_ON_TIMEOUT.md)时，符合条件的Work超时仍可重抛；非标准异常的未知异常分支也有独立重抛处理。

当前实现识别到`driver shutting down`时，会记录运行时退出并结束WatchDog，不按普通标准异常分支重抛。

该变量不改变Work错误分支是否清理通信器，清理策略由[TORCH_HCCL_ASYNC_ERROR_HANDLING](TORCH_HCCL_ASYNC_ERROR_HANDLING.md)决定。

该变量只影响WatchDog对应的外层异常处理，不改变用户线程调用Work接口时的异常传播。

> [!NOTE]
>
> - 设为0不表示错误已经恢复，也不表示WatchDog继续监控。后台线程结束后，后续Work可能不再得到正常的后台检查和结果通知。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置。
> - 不应通过关闭重抛来假设故障训练可以继续正常运行。进程组、通信器和后台检查是否仍可用，需要由上层恢复流程判断。

## 配置示例

保持默认重抛行为：

```bash
export TORCH_HCCL_RETHROW_HCCL_ERRORS=1
```

在上层已有明确错误接管设计的排障流程中，可关闭对应标准异常的重抛：

```bash
export TORCH_HCCL_RETHROW_HCCL_ERRORS=0
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
