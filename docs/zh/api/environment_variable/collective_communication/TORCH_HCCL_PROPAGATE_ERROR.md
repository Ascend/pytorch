# TORCH\_HCCL\_PROPAGATE\_ERROR

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制是否通过进程组Store传播和检查远端错误来源。TorchNPU 2.15版本默认值为0。

- 配置为“0”：不启用这条远端错误传播和检查路径。

- 配置为“1”：WatchDog在Work错误处理中发送错误来源；其他启用本选项的rank通过Store检查并记录远端错误。


接收端仅在进程组错误状态仍为`SUCCESS`时检查这条远端错误信号，发现后设置为`REMOTE_ERROR`并打印来源rank。

WatchDog即使没有待处理Work，也可以在轮询中检查远端错误；该路径不依赖心跳监控线程启动。

> [!NOTE]
>
> - 信号按进程组名称区分。当前实现写入的错误来源是该进程组内的rank，不应直接当作设备编号、`LOCAL_RANK`或任意子组的全局rank使用。
> - `REMOTE_ERROR`表示获知其他rank报错，不证明本地设备发生了同样的故障。排查时应优先寻找来源rank的最早错误。
> - 应在进程组创建前设置，并建议在各Rank上统一配置。
> - 传播和检查依赖进程组Store可用。进程在发送信号前退出，或Store访问失败时，不能保证远端获知错误。

## 配置示例

在参与进程组的各rank启动前配置：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_PROPAGATE_ERROR=1
```

接收端发现远端错误时，日志可能包含：

```text
remote error detected from rank: 1
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
