# TORCH\_HCCL\_DEBUG\_INFO\_PIPE\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可指定手动触发FR转储的命名管道前缀。监控线程为默认进程组创建`<prefix><rank>.pipe`，向管道写入任意非空内容可请求当前rank异步采集诊断信息。

默认值：空字符串，不创建命名管道。

> [!NOTE]
>
> 该变量指定触发入口，FR数据文件的输出位置由`TORCH_FR_DUMP_TEMP_FILE`等输出路径配置决定。管道请求只触发当前rank，采集后训练可继续执行。

## 配置示例

在启动训练进程前设置：

```bash
mkdir -p ./trace_dir
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_HCCL_DEBUG_INFO_PIPE_FILE="$PWD/fr_request_rank_"
export TORCH_FR_DUMP_TEMP_FILE="$PWD/trace_dir/comm_lib_trace_rank_"
```

训练进程创建管道后，在同一机器的另一个终端中向rank 0请求采集：

```bash
test -p ./fr_request_rank_0.pipe && printf 'dump\n' > ./fr_request_rank_0.pipe
```

## 使用约束

- 该变量在初始化HCCL进程组时读取，需使用支持命名管道的运行环境，并开启监控且保证FR缓冲区有效值大于0。
- 自定义前缀的父目录必须提前创建并可写。默认进程组负责创建管道；该进程的多个HCCL进程组共享FR记录。
- 不同任务应使用不同管道前缀，初始化时会移除同名旧管道并重新创建。
- 管道异步采集不会等待转储完成，也不广播到其他rank；如需多rank快照，应分别向各rank所在机器上的管道发送请求。
- 调用栈和活动条目过滤由[TORCH\_INCLUDE\_STACK\_TRACE](TORCH_INCLUDE_STACK_TRACE.md)、[TORCH\_INCLUDE\_ONLY\_ACTIVE](TORCH_INCLUDE_ONLY_ACTIVE.md)控制。
- 设置`TORCH_HCCL_DUMP_ON_TIMEOUT=0`不会禁用管道，但会调整监控循环的轮询间隔，可能导致请求响应延迟。
- 同步通信提交失败后，或在`TORCH_HCCL_ASYNC_ERROR_HANDLING=0`（NoHandling）模式下主动采集时，可能受通信事件生命周期限制导致进程异常退出；请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。

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
