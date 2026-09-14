# TORCHINDUCTOR\_WORKER\_LOGPATH

## 功能描述

通过此环境变量可指定Inductor worker子进程的日志输出路径。Inductor在编译kernel时会使用子进程进行并行编译，此环境变量控制子进程日志文件的保存位置。

- 默认值未配置：不指定日志文件路径，worker子进程日志输出行为由`TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING`决定。
- 配置为路径：worker子进程日志保存到指定位置。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 未配置日志路径时，若开启worker子进程日志输出（`TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING=0`），子进程日志将继承主进程的标准输出与标准错误，直接打印到启动Python进程的终端。
> - 此环境变量在`config.debug`开启时，会被重定向到终端输出（便于实时观察编译日志）。
> - 需确保目标路径可写。

该变量对应PyTorch的[TORCHINDUCTOR_WORKER_LOGPATH](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)，配置方式一致。

## 配置示例

```bash
export TORCHINDUCTOR_WORKER_LOGPATH=/tmp/inductor_worker_logs
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
