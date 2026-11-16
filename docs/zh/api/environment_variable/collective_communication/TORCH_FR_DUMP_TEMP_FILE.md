# TORCH\_FR\_DUMP\_TEMP\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置Flight Recorder诊断数据的输出路径及文件名。

- 静态模式：变量值为包含目录的文件名前缀，生成的文件名为`<prefix><rank>`，其中`rank`为全局rank编号。
- 动态模式：设置[TORCH\_FR\_DUMP\_DYNAMIC\_FILE\_NAME](TORCH_FR_DUMP_DYNAMIC_FILE_NAME.md)为1后，变量值为精确文件路径，每次写盘重新读取，不自动追加rank。

静态模式下，配置优先级从高到低为`TORCH_FR_DUMP_TEMP_FILE`、`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`、默认缓存路径。

两个变量均未设置时，默认路径如下：

| 配置情况 | 默认输出路径 |
| --- | --- |
| 未设置`XDG_CACHE_HOME` | `$HOME/.cache/torch/comm_lib_trace_rank_<rank>`。 |
| 已设置`XDG_CACHE_HOME` | `$XDG_CACHE_HOME/torch/comm_lib_trace_rank_<rank>`。 |
| `XDG_CACHE_HOME`和`HOME`均未设置 | `/tmp/.cache/torch/comm_lib_trace_rank_<rank>`。 |

> [!NOTE]
>
> 静态前缀在默认写入器首次创建时固定。动态模式每次写盘仅读取`TORCH_FR_DUMP_TEMP_FILE`，路径为空或未设置时不回退到兼容变量或默认前缀。HCCL写入器不读取`TORCH_NCCL_DEBUG_INFO_TEMP_FILE`。

## 配置示例

指定静态前缀：

```bash
mkdir -p ./trace_dir
export TORCH_FR_DUMP_TEMP_FILE="$PWD/trace_dir/comm_lib_trace_rank_"
```

rank 0、rank 1的文件分别为`./trace_dir/comm_lib_trace_rank_0`、`./trace_dir/comm_lib_trace_rank_1`。

## 使用约束

- 该变量仅指定输出位置，不触发转储。自动转储和手动采集方法请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。
- 指定目录应提前创建并确保当前训练进程处于可写状态。默认缓存目录会自动创建；无需将自定义前缀仅配置成目录路径。
- 同一rank重复写入同一文件会覆盖已有内容，建议不同训练任务使用独立目录。
- 动态模式下，各rank需配置互不冲突的完整路径；若使用离线分析工具，文件名须遵循命名规范。
- 独立的Python线程调用栈文件按`<prefix><rank>_py_traceback`规则生成，不随动态FR文件名规则改为精确路径。
- 即使指定了自定义输出路径，当前实现仍会准备默认缓存目录，应确保缓存根目录具备创建目录的权限。

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
