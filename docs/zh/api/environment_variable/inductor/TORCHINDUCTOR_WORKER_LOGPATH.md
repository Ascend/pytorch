# TORCHINDUCTOR\_WORKER\_LOGPATH

## 功能描述

通过此环境变量可指定Inductor worker子进程的日志输出路径。Inductor在编译kernel时会使用子进程进行并行编译，此环境变量控制子进程日志文件的保存位置。

- 默认值未设置，不指定日志文件路径，worker子进程日志输出行为由`TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING`决定。
- 设置路径：worker子进程日志保存到指定位置。

> [!NOTE]
>
> - 此环境变量在`INDUCTOR_ASCEND_DEBUG=1`或`config.debug`开启时，会被重定向到终端输出（便于实时观察编译日志）。
> - 需确保目标路径可写。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

```bash
export TORCHINDUCTOR_WORKER_LOGPATH=/tmp/inductor_worker_logs
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 需确保目标目录可写。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
