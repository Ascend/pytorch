# TORCHINDUCTOR\_WORKER\_SUPPRESS\_LOGGING

## 功能描述

通过此环境变量可控制是否输出Inductor worker子进程的日志。Inductor在编译kernel时会使用子进程进行并行编译，当日志输出关闭时，子进程日志将被重定向到/dev/null（即丢弃，不在终端显示）。

- 默认值为`True`：关闭worker子进程的日志输出，日志被重定向到/dev/null。
- 配置为“0”或空字符串：开启worker子进程的日志输出，日志正常打印。
- 配置为“1”或其他非“0”值：关闭worker子进程的日志输出。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 在`config.debug`开启时，若未显式设置此变量，将强制开启日志输出，便于调试。
> - 关闭worker子进程的日志输出可减少终端打印内容，但会丢失autotune过程中的调试信息。

该变量对应PyTorch的[TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)，配置方式一致。

## 配置示例

开启worker子进程的日志输出（调试时查看子进程输出）：

```bash
export TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING=0
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
