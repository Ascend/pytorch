# 飞行记录器

## 简介

飞行记录器（Flight Recorder，以下简称FR）在HCCL集合通信执行过程中持续记录通信操作、进程组、张量形状、数据类型、执行状态及调用栈等跟踪数据，并保存到内存中的环形缓冲区。环形缓冲区的容量固定，写满后，每写入一条新记录就覆盖最早的一条记录。

当训练任务出现通信超时、卡住或多rank通信不匹配时，可将记录转储为文件，再通过离线分析工具还原通信关系，辅助定位异常操作及其调用位置。

FR记录、心跳监控和常规自动转储默认关闭。设置正数的FR缓冲区条数即可开启记录。离线分析工具随PyTorch发行包提供，可在CPU环境运行，不需要NPU设备。

PyTorch社区提供了FR的使用教程与上游分析工具说明，可结合本文参考：[Flight Recorder for Debugging Stuck Jobs](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)。FR数据使用Python pickle序列化，分析时应只读取来源可信的作业文件，相关约束请参考[Python pickle文档](https://docs.python.org/3/library/pickle.html)。

## 使用场景

- 分布式训练出现通信超时，需要比较各rank在同一进程组中提交的操作是否一致。
- 训练任务长时间卡住，需要保留最近的通信记录并检查WatchDog是否仍有进展。
- 通信操作类型、输入输出大小或数据类型不一致，需要定位对应的rank、通信序列及调用位置。
- 调试集合通信、点对点通信或合并通信，希望保存记录进行比对。

## 使用指导

### 开启与配置飞行记录器

FR配置按“记录与输出”和“触发转储”两类划分。记录与输出相关环境变量如下表，取值、默认行为和支持型号以各变量详情页为准。

**表 1**  记录和输出配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| [TORCH_FR_BUFFER_SIZE](../../api/environment_variable/collective_communication/TORCH_FR_BUFFER_SIZE.md) | 0（关闭） | 环形缓冲区最多保存的记录条数。设为正整数开启记录。兼容旧名称`TORCH_HCCL_TRACE_BUFFER_SIZE`。 |
| [TORCH_FR_CPP_STACK](../../api/environment_variable/collective_communication/TORCH_FR_CPP_STACK.md) | 0（关闭） | 是否采集通信操作的C++调用栈。 |
| [TORCH_HCCL_ENABLE_TIMING](../../api/environment_variable/collective_communication/TORCH_HCCL_ENABLE_TIMING.md) | 0（关闭） | 是否启用可计算设备执行时长的NPU计时事件，用于在记录中生成`duration_ms`。开启`TORCH_HCCL_DESYNC_DEBUG`也会启用计时。 |
| [TORCH_FR_DUMP_TEMP_FILE](../../api/environment_variable/collective_communication/TORCH_FR_DUMP_TEMP_FILE.md) | 默认缓存目录中的`comm_lib_trace_rank_`前缀 | 静态模式下为文件名前缀，动态模式下为精确路径。兼容旧名称`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`。 |
| [TORCH_FR_DUMP_DYNAMIC_FILE_NAME](../../api/environment_variable/collective_communication/TORCH_FR_DUMP_DYNAMIC_FILE_NAME.md) | 0（关闭） | 开启后每次写盘按精确路径写入，不自动追加rank。 |
| [TORCH_INCLUDE_STACK_TRACE](../../api/environment_variable/collective_communication/TORCH_INCLUDE_STACK_TRACE.md) | 1（开启） | 常规监控和管道转储是否输出FR条目的调用栈。 |
| [TORCH_INCLUDE_ONLY_ACTIVE](../../api/environment_variable/collective_communication/TORCH_INCLUDE_ONLY_ACTIVE.md) | 0（关闭） | 转储是否只输出尚未观测到完成的条目。 |

缓冲区条数、C++调用栈采集和转储路径的读取规则如下。存在多个配置时，按表中顺序使用第一个已配置的变量。

**表 2**  配置读取优先级（从高到低）

| 配置项 | 优先级 |
| --- | --- |
| 缓冲区条数 | `TORCH_FR_BUFFER_SIZE`、`TORCH_HCCL_TRACE_BUFFER_SIZE`、默认0。 |
| C++调用栈采集 | `TORCH_FR_CPP_STACK`、`TORCH_HCCL_TRACE_CPP_STACK`、默认关闭。 |
| 静态转储前缀 | `TORCH_FR_DUMP_TEMP_FILE`、`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`、默认缓存前缀。 |
| 动态转储路径 | 每次写盘只读取`TORCH_FR_DUMP_TEMP_FILE`，为空时不回退。 |

缓冲区和C++调用栈配置在记录器首次访问时固定。请在初始化进程组前设置。

#### 转储文件位置

转储文件位置由`TORCH_FR_DUMP_TEMP_FILE`控制。默认使用静态模式，兼容旧变量`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`，文件名为`<prefix><rank>`。

- 两个前缀变量均未设置时，文件通常位于`~/.cache/torch/comm_lib_trace_rank_<rank>`。如果设置了`XDG_CACHE_HOME`，文件位于`$XDG_CACHE_HOME/torch/comm_lib_trace_rank_<rank>`。
- 较早的发布版本可能默认写入 `/tmp/hccl_trace_rank_<rank>`。可显式设置前缀，或在 WatchDog 日志中的 `dumping hccl trace to` 消息确认实际路径。
- 设置其中一个前缀变量时，在指定前缀后追加rank。两者均设置时使用`TORCH_FR_DUMP_TEMP_FILE`。例如设置`TORCH_FR_DUMP_TEMP_FILE=/tmp/comm_lib_trace_rank_`后，rank 0和rank 1分别写入`/tmp/comm_lib_trace_rank_0`和`/tmp/comm_lib_trace_rank_1`。

设置`TORCH_FR_DUMP_DYNAMIC_FILE_NAME=1`可切换为动态模式。每次转储都写入`TORCH_FR_DUMP_TEMP_FILE`指定的精确路径，不追加rank。请为每个rank配置不同的路径。路径为空时转储失败，不回退到默认前缀。离线分析前，按各文件对应的全局rank将文件复制为统一的`<prefix><rank>`命名格式。

### 触发转储

记录保存在内存中，需要通过下列入口触发写盘。自动故障转储依赖WatchDog与心跳监控线程，相关环境变量属于WatchDog/心跳监控范围，本文仅说明配合关系，监控相关说明请参见[WatchDog](watchdog.md)。

- **常规自动转储**：WatchDog检测到Work异常或通信超时，或监控线程检测到WatchDog心跳长期无进展时触发。将`TORCH_HCCL_ENABLE_MONITORING`和`TORCH_HCCL_DUMP_ON_TIMEOUT`均设为1，同时将FR缓冲区设为正整数。心跳超时由`TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC`控制，默认600秒。
- **WatchDog异常出口额外采集**：由[TORCH_HCCL_EXTRA_DUMP_ON_EXEC](../../api/environment_variable/collective_communication/TORCH_HCCL_EXTRA_DUMP_ON_EXEC.md)=1开启。WatchDog捕获标准异常后额外尝试广播诊断请求并在当前rank采集快照，可覆盖communicator预检查发现异步错误、异步错误查询接口本身失败等可能未经过常规转储请求的情况。本地调用不依赖监控或常规转储开关；远端是否响应仍由各rank的监控配置和执行时序决定。
- **命名管道手动触发**：设置[TORCH_HCCL_DEBUG_INFO_PIPE_FILE](../../api/environment_variable/collective_communication/TORCH_HCCL_DEBUG_INFO_PIPE_FILE.md)为管道前缀后，向`<prefix><rank>.pipe`写入非空内容可请求当前rank异步采集。需开启监控且FR缓冲区大于0。

常规自动转储单次默认等待60秒，由`TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC`控制。首次超时后再尝试一次不包含FR条目调用栈的转储。默认进程组监控线程检查协调转储请求的轮询间隔由`TORCH_HCCL_COORD_CHECK_MILSEC`控制，默认1000毫秒。

> [!NOTE]
>
> FR记录、常规监控转储和异常出口额外转储分别受不同配置控制。关闭`TORCH_HCCL_DUMP_ON_TIMEOUT`不会关闭记录或管道采集。关闭监控也不会关闭内存中的FR记录。

### 准备分析目录

在包含`torch.distributed.flight_recorder`的PyTorch 2.15环境中进行离线分析。新建空目录`./fr_analysis`，只将同一作业、同一批次的FR主文件复制到该目录，每个全局rank保留一份。下面是采用默认文件名的双rank分析目录样例。

```text
fr_analysis/
├── comm_lib_trace_rank_0
└── comm_lib_trace_rank_1
```

使用自定义前缀时，各主文件应采用统一的`<prefix><rank>`命名，例如自定义前缀为`comm_lib_trace_rank_`产生的文件是`comm_lib_trace_rank_0`和`comm_lib_trace_rank_1`。采用动态文件名时，按采集时的全局rank将主文件复制为这种命名。

将`<prefix><rank>_py_traceback`等Python线程栈文本、日志和分析结果留在分析目录之外。社区工具会递归扫描输入目录，同前缀的伴随文件会导致rank解析失败。

### 运行离线分析

将准备好的分析目录传给PyTorch安装包中的分析工具。工具会尝试推断文件名前缀并执行跨rank匹配。

```bash
python -m torch.distributed.flight_recorder.fr_trace ./fr_analysis
```

- 若前缀无法自动推断，可加`--prefix comm_lib_trace_rank_`。使用自定义前缀时，传入实际前缀。
- 若需要保存分析数据库，可加`-o ./output.pkl`，将输出写到分析目录之外。
- 若只需打印记录条目，先在分析环境中安装`tabulate`，再加`-j`。可继续加`--selected-ranks 0 1`筛选rank，或加`--pg-filters`筛选进程组。若主文件包含调用栈，可加`--print_stack_trace`打印去重后的调用栈。
- 若缺少部分rank的主文件，可加`--allow-incomplete-ranks`分析已有记录。
- 若分析环境中的设备后端扩展在自动加载时报错或导致启动超时，可在命令前加`TORCH_DEVICE_BACKEND_AUTOLOAD=0`。若当前目录或`PYTHONPATH`影响模块加载，可在`python`后加`-I`。

社区分析器的命令、分析模式和筛选示例可参考PyTorch教程：[Analyzing Flight Recorder Dumps](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#analyzing-flight-recorder-dumps)。

#### 入参说明

**表 3**  离线分析工具参数

| 参数名 | 含义 | 使用限制 |
| --- | --- | --- |
| `trace_dir` | 保存FR主文件的分析目录，文件名为`<prefix><rank>`。 | 必填。目录及子目录只存放同一作业、同一批次的FR主文件，每个全局rank一份。 |
| `-p`、`--prefix` | 用于提取rank的文件名前缀。 | 可选。省略时工具尝试推断前缀。 |
| `-o`、`--output` | 输出数据库路径。 | 可选。父目录应存在且可写。 |
| `-j`、`--just_print_entries` | 仅打印记录条目，不执行完整匹配。 | 可选。默认关闭。启用时需安装`tabulate`。 |
| `--selected-ranks` | 选择要展示的rank，例如`0 1`。 | 可选。须与`-j`同时使用，仅过滤展示。 |
| `--pg-filters` | 按进程组名称或描述筛选展示内容。 | 可选。须与`-j`同时使用。 |
| `--print_stack_trace` | 在`-j`模式下输出去重后的通信提交栈。 | 可选。与`-j`同时使用，需安装`tabulate`，且原始转储需包含调用栈。 |
| `--allow-incomplete-ranks` | 允许缺少rank文件时进行尽力分析。 | 可选。默认关闭；结果不能替代完整rank分析。 |
| `-v`、`--verbose` | 输出详细日志和更多条目信息。 | 可选。默认关闭。 |
| `--mismatch_cap` | 限制单个进程组报告的最大不匹配数量。 | 可选。int类型，默认10。 |
| `--transform-ft` | 转换TorchFT replica的进程组配置以使用全局rank分析。 | 可选。须同时指定`--group-world-size`。 |
| `--group-world-size` | 单个TorchFT replica的rank数量。 | 可选。使用`--transform-ft`时必需，应为正整数。 |
| `-h`、`--help` | 查看帮助信息。 | 可选。 |

`--selected-ranks`和`--pg-filters`只过滤展示，不会变更采集范围，也不会只加载指定rank文件。需要完整匹配时，应准备相关进程组全部成员的文件并不使用`-j`。

### 查看分析结果

排查时先确认文件属于同一次任务，检查缺失rank和版本信息，再查看最早出现不匹配的进程组及通信序列，随后比较相关rank的操作类型、输入输出形状、数据类型和状态，结合调用栈及业务分支定位原因。

**表 4**  原始FR数据中的主要字段

| 字段 | 说明 |
| --- | --- |
| `version` | FR 数据的 schema 版本，由写出转储的 TorchNPU 版本决定。当前主线写出 `2.10`，较早版本可能写出 `2.4`。 |
| `pg_config` | 进程组名称、描述和成员rank信息，用于重建PG关系。 |
| `entries` | 环形缓冲区中本次导出的通信条目列表。 |
| `process_group`、`pg_id` | 条目所属进程组的名称、描述和标识。 |
| `record_id` | 记录编号，用于排列缓冲区内条目。 |
| `collective_seq_id`、`p2p_seq_id`、`is_p2p` | 集合通信和点对点通信各自的序列信息及操作类别，不能将两类序列混为同一计数。 |
| `profiling_name` | 包含后端前缀的操作名称，例如 `hccl:all_reduce`。较早版本的转储可能留空，依赖该字段的社区分析器无法解析这类条目。 |
| `input_sizes`、`output_sizes` | 输入输出张量的形状列表。 |
| `input_dtypes`、`output_dtypes` | 输入输出张量的数据类型列表。 |
| `state` | 记录器观测到的状态：`scheduled`、`started`或`completed`。 |
| `time_created_ns`、`time_discovered_started_ns`、`time_discovered_completed_ns` | 主机创建或观测状态的时间，单位为纳秒；不能直接作为设备执行时长或跨机器统一时钟使用。 |
| `duration_ms` | 可计算时才输出的设备执行时长，单位为毫秒；需启用计时且满足完成等条件。缺失不表示0。 |
| `frames` | 本次转储包含的调用栈信息；关闭调用栈输出或额外快照时可能缺失。 |

`scheduled`表示尚未观测到开始，`started`表示已观测到开始但尚未观测到完成，`completed`表示已观测到完成。单次快照中的状态差异也可能来自采集时刻不同，不能只凭状态差异认定根因。

**表 5**  常见匹配结果

| 结果 | 说明和排查方向 |
| --- | --- |
| `FULLY_MATCHED` | 本次比较满足匹配条件；不代表训练任务或所有通信均已成功结束。 |
| `COLLECTIVE_TYPE_MISMATCH` | 操作类型不一致，检查各rank的通信调用顺序和业务分支。 |
| `SIZE_OR_SYNTAX_MISMATCH` | 输入输出大小或操作格式不符合匹配规则，检查张量形状、元素数量及操作要求。 |
| `COLLECTIVE_STATE_MISMATCH` | 操作状态不一致，结合采集时刻、前序操作及进度判断。 |
| `COLLECTIVE_DTYPE_MISMATCH` | 输入输出数据类型不一致，检查各rank构造张量的逻辑。 |
| `UNDECIDED` | 仅凭当前两条记录无法确定，需要更多rank信息或后续检查。 |
| 缺少参与rank | 检查文件是否缺失、记录是否被环形缓冲区覆盖，以及对应rank是否进入了该通信。 |

`-o`导出的分析数据库为pickle格式。社区工具生成的数据库通过`db.ncclcalls`访问各rank的通信调用，HCCL操作名称仍以`hccl:`开头。读取该数据库的脚本应使用`db.ncclcalls`。

如果手头的数据库由仓库原有的本地工具`tools.flight_recorder.fr_trace`导出，其调用列表字段为`db.hcclcalls`，且文件引用了本地工具的Python类型。可以在生成该数据库的环境中读取，也可以用保留的各rank原始FR主文件通过社区工具重新分析并导出。较早的HCCL schema v2.4原始转储请在与其匹配的分析环境中处理。

## 使用样例

### 配置自动故障转储

在启动训练前配置以下变量，然后按原有方式启动使用HCCL后端的训练脚本：

```bash
umask 077
mkdir -p ./trace_dir
export TORCH_HCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_FR_DUMP_TEMP_FILE="$PWD/trace_dir/comm_lib_trace_rank_"
```

在仅支持旧变量名的 TorchNPU 版本上，将示例中的 `TORCH_FR_BUFFER_SIZE` 和 `TORCH_FR_DUMP_TEMP_FILE` 分别替换为 `TORCH_HCCL_TRACE_BUFFER_SIZE` 和 `TORCH_HCCL_DEBUG_INFO_TEMP_FILE`。

如需设备执行时长或C++调用栈，可额外配置：

```bash
export TORCH_HCCL_ENABLE_TIMING=1
export TORCH_FR_CPP_STACK=1
```

如需覆盖WatchDog异常出口的额外采集，可再设置：

```bash
export TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1
```

以上设置不会主动制造通信错误，只有相应诊断条件满足时才尝试生成文件。

### 分析和导出数据库

按「准备分析目录」将同一批次的FR主文件复制到`./fr_analysis`。以下命令显式指定前缀，并将数据库导出到输入目录之外。

```bash
python -m torch.distributed.flight_recorder.fr_trace ./fr_analysis \
  --prefix comm_lib_trace_rank_ -o ./output.pkl
```

安装`tabulate`后，可只打印指定rank的条目并输出已保存的调用栈。

```bash
python -m torch.distributed.flight_recorder.fr_trace ./fr_analysis \
  -j --selected-ranks 0 1 --print_stack_trace
```

如果需要转换每个replica包含4个rank的TorchFT转储，运行以下命令。

```bash
python -m torch.distributed.flight_recorder.fr_trace ./fr_analysis \
  --transform-ft --group-world-size 4
```

TorchFT参数仅用于对应转储数据的离线转换，不是普通HCCL训练的必需配置。

## 约束说明

- 采集功能面向HCCL后端。自动监控转储、管道转储和Python线程调用栈采集需要相应Python运行环境。
- 使用常规自动转储时，按“触发转储”一节开启FR记录、心跳监控和常规转储，并建议设置`TORCH_HCCL_ASYNC_ERROR_HANDLING=1`和`TORCH_HCCL_BLOCKING_WAIT=0`。如果还希望WatchDog捕获标准异常时尝试额外转储，可单独设置`TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1`。
- 关于WatchDog和通信超时的配置关系，请参见[WatchDog](watchdog.md)。心跳超时、通信执行超时、进程组超时和FR转储等待时间不能混为同一配置。
- 同步通信提交失败后，或在`TORCH_HCCL_ASYNC_ERROR_HANDLING=0`（NoHandling）模式下，主动采集可能受通信事件生命周期限制并导致进程异常退出。
- 部分communicator错误或异步错误查询失败，未开启异常出口额外转储时，可能在生成文件前终止；开启后仍受诊断请求竞争、采集和写盘条件影响，不保证所有rank都有文件。
- Store故障、进程在HCCL错误路径之外退出、获取GIL失败或底层I/O阻塞，均可能导致数据缺失。转储属于尽力采集，不能承诺固定退出时限或文件生成结果。
- 环形缓冲区可能覆盖故障前的记录。工具会按各rank的起始`record_id`对齐可用窗口；窗口不足、缺少rank或复杂合并操作的历史不完整时，匹配结果可能受限。没有报告不匹配不表示所有通信均正确完成。
- 调用栈采集、计时、较大的缓冲区及转储都会增加资源开销，应按排障需要配置。FR及其离线数据库不替代完整性能分析工具，不能据此推断所有通信Kernel的执行情况。
- **已知分析限制**：先执行all-reduce再提交首批P2P时，集合通信和P2P序号相同可能导致分组分析失败；HCCL原生批量P2P的整批记录名为`hccl:UNKNOWN`，社区分析器当前会拒绝这个操作名。
