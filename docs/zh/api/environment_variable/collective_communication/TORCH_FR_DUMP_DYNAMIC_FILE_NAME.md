# TORCH\_FR\_DUMP\_DYNAMIC\_FILE\_NAME

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制Flight Recorder是否使用动态文件名写入诊断数据。

- 0：使用静态模式，在输出前缀后自动追加全局rank编号。
- 1：使用动态模式，每次写盘读取`TORCH_FR_DUMP_TEMP_FILE`，按精确路径写入，不追加rank编号。

默认值：0。

> [!NOTE]
>
> 该变量在默认写入器首次创建时读取。动态模式允许后续修改`TORCH_FR_DUMP_TEMP_FILE`以更换下一次写盘位置，但不会自动生成时间戳或递增文件名。

## 配置示例

在每个训练worker中、首次转储前分别设置路径。以下为Python配置样例，`RANK`由`torchrun`为worker设置：

```python
import os
from pathlib import Path

trace_dir = Path("./trace_dir").resolve()
trace_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_FR_DUMP_DYNAMIC_FILE_NAME"] = "1"
os.environ["TORCH_FR_DUMP_TEMP_FILE"] = str(
    trace_dir / f"comm_lib_trace_rank_{os.environ['RANK']}"
)
```

## 使用约束

- 必须同时配置[TORCH\_FR\_DUMP\_TEMP\_FILE](TORCH_FR_DUMP_TEMP_FILE.md)。该变量为空或未设置时，打开文件失败，不回退到旧变量或默认路径。
- 多rank不能共用同一个精确路径，否则可能相互覆盖。重复使用同一路径也会覆盖上一份记录。
- 离线分析工具仍要求文件名以rank数字结尾，例如`comm_lib_trace_rank_0`；任意动态文件名需在分析前按rank整理。
- 该变量只影响FR数据文件，不改变独立Python线程调用栈文件的命名规则。

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
