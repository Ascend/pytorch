# TORCH\_FR\_CPP\_STACK

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制Flight Recorder是否采集通信操作的C++调用栈。

采集和解析C++调用栈会带来额外开销，建议在需要定位通信调用位置时开启。

- 0：不采集C++调用栈。
- 1：采集C++调用栈。

默认值：0。

> [!NOTE]
>
> - 配置在当前进程首次访问FR时固定，之后修改不生效。应在启动训练进程前设置。
> - 该变量控制C++调用栈的采集，不用于关闭Python调用栈或控制转储文件是否输出调用栈。
> - 原生HCCL记录器不读取社区NCCL配置`TORCH_NCCL_TRACE_CPP_STACK`。
> - 输出调用栈还取决于转储入口的设置。监控和管道转储可通过[TORCH\_INCLUDE\_STACK\_TRACE](TORCH_INCLUDE_STACK_TRACE.md)控制输出；无调用栈重试和异常出口的额外转储会保留通信记录，但不包含发起通信的调用栈，即转储文件中的通信条目不包含`frames`字段。

## 配置示例

开启FR记录并采集通信操作C++调用栈的配置示例如下：

```bash
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_FR_CPP_STACK=1
```

## 使用约束

- 该变量仅在[TORCH\_FR\_BUFFER\_SIZE](TORCH_FR_BUFFER_SIZE.md)的有效值大于 0（即开启 FR 记录）时生效。
- 当前版本同时兼容旧名称`TORCH_HCCL_TRACE_CPP_STACK`。两个变量同时配置时，优先使用`TORCH_FR_CPP_STACK`。

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
