# 飞行记录器分析

训练任务卡住是阻塞AI大规模分布式集群训练任务的主要和关键问题，当前需要等待集合通信超时才能感知，影响集群可用性。框架需要支持检测训练任务卡住问题，做到提前识别并保存必要的诊断信息，提高问题定位效率和集群设备可用性。当HeartbeatMonitor长时间未检测到心跳时，即可认为训练任务已经卡住，需要触发诊断信息保存。
主要功能包括：解析多个 rank 的跟踪数据文件、重建进程组和成员关系、检测通信不匹配（类型、大小、状态等、提供详细的错误报告和调试信息

# 使用方法

## 1 飞行记录器开启方法

按照如下方法设置环境变量开启飞行记录器

```
export TORCH_HCCL_ENABLE_MONITORING=1 #用于检测是否开启卡住问题检测
export TORCH_HCCL_DUMP_ON_TIMEOUT=1 # 用于控制是否保存诊断信息
export TORCH_HCCL_TRACE_BUFFER_SIZE=1 # 用于控制保存的集合通信状态数量
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=20 # 用于控制心跳超时时间，即训练业务多久未下发集合通信算子时需要判定为卡住，默认10分钟，单位s。（需要小于HCCL_EXEC_TIMEOUT，避免集合通信先报超时错误）
export TORCH_HCCL_DEBUG_INFO_TEMP_FILE=/tmp/  #保存诊断信息的文件路径
```

## 2 工具使用方法

```
python fr_trace.py <trace_dir> [options]
```

## 3 入参说明

| 参数名 | 含义 | 使用限制 |
| --- | --- | --- |
| `trace_dir` | 包含跟踪文件的目录，每个 rank 一个文件，命名格式为 `<prefix>_<rank>`。 | 可选。数据类型：string |
| `--selected-ranks` | 指定需要展示跟踪的 rank 列表。 | 可选。数据类型：int 列表。必须与 `--just-print-entries` 同时使用。 |
| `--allow-incomplete-ranks` | 允许对不完整的 rank 数据进行尽力分析并输出结果。 | 可选。无参数值。默认关闭。 |
| `--pg-filters` | 指定需要展示的进程组（PG）的过滤条件列表，可以是 PG 名称或描述。 | 可选。数据类型：string 列表。必须与 `--just-print-entries` 同时使用。 |
| `-o`, `--output` | 指定输出文件路径。 | 可选。数据类型：string。默认无输出文件。 |
| `-p`, `--prefix` | 指定文件名前缀，用于提取 rank。如果不指定，将尝试推断一个公共前缀。 | 可选。数据类型：string。默认无前缀。 |
| `-j`, `--just-print-entries` | 仅打印跟踪条目，不进行完整分析。 | 可选。无参数值。默认关闭。 |
| `-v`, `--verbose` | 启用详细日志输出。 | 可选。无参数值。默认关闭。 |

## 使用示例

```bash
python script.py trace_dir --selected-ranks 0 1 2 --allow-incomplete-ranks --pg-filters pg1 pg2 -o output.pkl -p prefix_ -j -v
```