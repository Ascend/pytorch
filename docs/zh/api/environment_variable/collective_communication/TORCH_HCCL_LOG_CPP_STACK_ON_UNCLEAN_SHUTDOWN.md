# TORCH\_HCCL\_LOG\_CPP\_STACK\_ON\_UNCLEAN\_SHUTDOWN

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制心跳监控线程进入故障诊断流程后，是否尝试调用已注册的C++栈采集器并输出诊断日志。TorchNPU 2.15版本默认值为1。

- 配置为“0”：跳过这一步C++栈采集。

- 配置为“1”：在采集器已注册时尝试执行C++栈采集。

默认值为1不代表正常训练会持续打印C++栈。只有监控线程实际运行并进入故障诊断时才可能执行这一步。

关闭该变量只跳过这一项诊断，不关闭心跳监控、其他转储路径或故障终止处理。

> [!NOTE]
>
> - 未注册C++栈采集器时，即使该变量为1也不会通过这一步得到C++栈输出。
> - 不限制C++栈采集的耗时。如果C++栈采集卡住，诊断过程仍可能一直等待。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置。
> - 日志是否可见还受日志级别和输出配置影响，不能仅因没有看到栈日志就判断监控线程没有运行。

## 配置示例

开启心跳监控，并允许故障诊断时尝试采集C++栈：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=1
```

如需跳过这一步采集：

```bash
export TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0
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
