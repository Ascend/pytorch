# TORCHINDUCTOR\_WORKER\_SUPPRESS\_LOGGING

## 功能描述

通过此环境变量可设置是否抑制Inductor worker子进程的日志输出。Inductor在编译kernel时会使用子进程进行并行编译，开启抑制后，子进程的日志输出将被重定向到/dev/null。

- 默认值为`True`（抑制），子进程日志被重定向到/dev/null。
- 配置为`0`或空字符串：不抑制，worker子进程日志正常输出。
- 配置为`1`或其他非`0`值：抑制worker子进程的日志输出。

> [!NOTE]
>
> - 在`INDUCTOR_ASCEND_DEBUG=1`或`config.debug`开启时，若未显式设置此变量，会强制不抑制（便于调试）。
> - 抑制worker日志可减少终端输出噪声，但会丢失autotune过程中的调试信息。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

不抑制worker日志（调试时查看子进程输出）：

```bash
export TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING=0
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
