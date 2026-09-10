# TORCH\_HCCL\_DEBUG\_INFO\_PIPE\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置命名管道文件，用于外部触发Flight Recorder dump。

- 非空字符串且`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`时，在指定路径创建`<stem><rank>.pipe`命名管道文件。
- 向该管道文件写入任意内容，即可触发HCCL debug dump。
- 空字符串：不创建管道文件。

默认值：空（不创建管道）。

> [!NOTE]
>
> - 此环境变量仅在`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`时生效。
> - 管道文件通过`mkfifo`创建，`O_RDONLY | O_NONBLOCK`模式打开。

该变量对应PyTorch的`TORCH_NCCL_DEBUG_INFO_PIPE_FILE`。配置方式一致，默认值均为空。

## 配置示例

```bash
export TORCH_HCCL_DEBUG_INFO_PIPE_FILE=/tmp/hccl_debug_pipe
```

## 使用约束

- 管道文件在被创建后，用户可通过`echo`等命令向管道写入数据触发dump，例如：
  ```bash
  echo "dump" > /tmp/hccl_debug_pipe_0.pipe
  ```
- 管道文件是在ProcessGroupHCCL构造时创建的，需在创建PG前设置环境变量。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
