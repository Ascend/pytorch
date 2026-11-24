# TORCH\_HCCL\_ASYNC\_ERROR\_HANDLING

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制WatchDog在Work异常分支中的通信器清理和异常抛出策略。TorchNPU 2.15版本默认值为3。

| 取值 | 模式 | 处理方式 |
| --- | --- | --- |
| 0 | `NoHandling` | 不在该分支主动清理通信器或抛出异常；仍跟踪符合条件的Work，记录错误和结果 |
| 1 | `TearDown` | 尝试清理相关通信器，并抛出Work异常 |
| 2 | `CleanUpOnly` | 尝试清理相关通信器，不在该分支抛出Work异常 |
| 3 | `SkipCleanUp` | 跳过该分支的通信器清理，抛出Work异常 |

各取值的异常需经WatchDog外层重抛策略处理后才能决定最终结果：默认配置下会重抛异常，取值1和3会使进程异常终止，但取值0和2也无法确保所有故障下进程都不退出。心跳监控、通信器预检查等异常路径有独立处理规则。

在非阻塞等待模式下启用Desync诊断（TORCH_DISTRIBUTED_DEBUG=DETAIL亦可启用）且取值为0时，进程组在初始化时会将自身的异步错误处理策略改为模式3（`SkipCleanUp`）。

取值1和3可能调整内部默认WatchDog超时时间；此时建议new_group和init_process_group的timeout大于HCCL_CONNECT_TIMEOUT和HCCL_EXEC_TIMEOUT，以便定位HCCL超时原因，详见《CANN HCCL集合通信库》中“[HCCL_CONNECT_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/zh/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)”和“[HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)”章节。


> [!NOTE]  
>
> - 当前版本同时兼容旧名称`HCCL_ASYNC_ERROR_HANDLING`。
> - 当`TORCH_HCCL_ASYNC_ERROR_HANDLING`和`HCCL_ASYNC_ERROR_HANDLING`同时配置时，优先使用`TORCH_HCCL_ASYNC_ERROR_HANDLING`。
> - 该配置需在首次使用前设置，建议在启动各Rank训练进程前统一完成。由于配置会被缓存，训练中途修改环境变量无法切换已有进程组的策略。
> - 取值为2不是训练自动恢复功能，清理通信器后继续使用原进程组可能存在风险。

## 配置示例

默认值时，跳过清理通信器配置如下：

```bash
export TORCH_HCCL_ASYNC_ERROR_HANDLING=3
```

如需在Work异常分支先尝试清理通信器再抛出：

```bash
export TORCH_HCCL_ASYNC_ERROR_HANDLING=1
```

## 使用约束

该环境变量在TorchNPU 2.15.0之前版本默认值为1。

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
