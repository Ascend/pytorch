# TORCH\_HCCL\_DEBUG\_INFO\_PIPE\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置命名管道文件，用于外部触发Flight Recorder dump。该环境变量默认值为空。

- 配置为指定路径（作为文件名前缀）：在指定路径创建`<stem><rank>.pipe`命名管道文件，其中`<stem>`为该环境变量的值，`<rank>`为本rank序号。例如配置值为`/tmp/hccl_debug_pipe_`时，rank 0创建的管道文件为`/tmp/hccl_debug_pipe_0.pipe`。
- 未配置或默认配置：不创建管道文件。

向该管道文件写入任意内容，即可触发HCCL debug dump。

> [!NOTE]
>
> 管道文件通过`mkfifo`创建，`O_RDONLY | O_NONBLOCK`模式打开。

该变量对应PyTorch的`TORCH_NCCL_DEBUG_INFO_PIPE_FILE`。配置方式一致，默认值均为空。

## 配置示例

创建管道：

```bash
export TORCH_HCCL_DEBUG_INFO_PIPE_FILE=/tmp/hccl_debug_pipe_
```

管道文件在被创建后，用户可通过`echo`等命令向管道写入数据触发dump：

```bash
echo "dump" > /tmp/hccl_debug_pipe_0.pipe
```

## 使用约束

- 此环境变量仅在`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`时生效。
- 管道文件是在ProcessGroupHCCL构造时创建的，需在创建PG前设置环境变量。

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
