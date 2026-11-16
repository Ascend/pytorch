# TORCH\_INCLUDE\_STACK\_TRACE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制监控线程执行常规故障转储或管道转储时，是否输出FR条目的调用栈。

FR条目的调用栈指各次通信发起时的调用栈（通信条目中的`frames`字段）。该变量不影响独立的Python线程调用栈采集。在相应转储请求处理时会读取该环境变量，建议在启动训练进程前设置。

- 0：不输出FR条目的调用栈。
- 1：输出已采集的FR条目调用栈。

默认值：1。

> [!NOTE]
>
> - 该变量控制转储时的调用栈输出，不控制记录阶段是否采集C++调用栈。C++调用栈采集由[TORCH\_FR\_CPP\_STACK](TORCH_FR_CPP_STACK.md)控制。
> - 首次故障转储等待超时后的重试，以及`TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1`触发的额外快照，均不输出FR条目调用栈，不受该变量设置为1的影响。

## 配置示例

不在常规故障转储或管道转储中输出FR条目调用栈：

```bash
export TORCH_INCLUDE_STACK_TRACE=0
```

## 使用约束

无

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
