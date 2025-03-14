# 飞行记录器超时类问题分析

训练任务卡住是阻塞AI大规模分布式集群训练任务的主要和关键问题，当前需要等待集合通信超时才能感知，影响集群可用性。框架需要支持检测训练任务卡住问题，做到提前识别并保存必要的诊断信息，提高问题定位效率和集群设备可用性。当HeartbeatMonitor长时间未检测到心跳时，即可认为训练任务已经卡住，需要触发诊断信息保存。

本工具提供torch npu上飞行记录器flight recorder记录日志的读取解析能力，并根据解析后的日志提供超时类问题的初步分析能力，主要支持以下三种情况的超时类问题的识别和分析

|问题| 具体内容 | 
| --- | --- |
|类型一  | 同通信域内的某张卡计算超时，导致其他卡等待触发飞行记录器和hccl time out | 
|类型二  | 同通信域内的通信算子之后的非通信任务耗时过长|
|类型三  | 同通信域内的某个通信算子进行通信时执行超时 |

## 使用方法

### 1 飞行记录器开启方法

按照如下方法设置环境变量开启飞行记录器

```
export TORCH_HCCL_ENABLE_MONITORING=1 #用于检测是否开启卡住问题检测
export TORCH_HCCL_DUMP_ON_TIMEOUT=1 # 用于控制是否保存诊断信息
export TORCH_HCCL_TRACE_BUFFER_SIZE=1 # 用于控制保存的集合通信状态数量
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=20 # 用于控制心跳超时时间，即训练业务多久未下发集合通信算子时需要判定为卡住，默认10分钟，单位s。（需要小于HCCL_EXEC_TIMEOUT，避免集合通信先报超时错误）
export TORCH_HCCL_DEBUG_INFO_TEMP_FILE=/tmp/  #保存诊断信息的文件路径
```

### 2 工具使用方法

```
python analysis_flight.py --path /tmp/ --world_size 8
```

脚本从命令行参数传入 `path` 和 `world_size` 的值，并记录日志。如果未提供命令行参数，则使用默认值。

* `path`：从命令行第一个参数获取，如果未提供则使用 `default_path`, default_path从TORCH_HCCL_DEBUG_INFO_TEMP_FILE获取。
* `world_size`：从命令行第二个参数获取，如果未提供则使用 `default_world_size`，默认为8。

| 参数名| 含义 | 使用限制 |
| --- | --- | --- | 
| path | 飞行记录器的日志 | 可选。数据类型：string 默认为环境变量中的TORCH_HCCL_DEBUG_INFO_TEMP_FILE,若设置日志格式指定有前缀，则需要在路径中加入前缀 | 
| world_size | 同一个通信域中的卡数 | 可选。数据类型：int 默认为8 |

### 3 输出示例

```
2025-02-19 08:10:07,160 - INFO - Path: /tmp/
2025-02-19 08:10:07,160 - INFO - World Size: 8
2025-02-19 08:10:07,162 - INFO - The pg_id 0's rank 0's Computational task took too long, causing the other ranks' HCCL task to time out.
```
