# TORCH\_NPU\_LOGS

## 功能描述

此环境变量用于配置Ascend Extension for PyTorch新增模块的日志打印功能，为开发者在Debugging场景下提供精准的调试定位能力。

Ascend Extension for PyTorch新增的模块不支持通过原生`TORCH_LOGS`进行设置，如需设置新增模块的日志信息，需要使用`TORCH_NPU_LOGS`，新增模块列表如下：

| 字段名 | 对应模块 | 功能描述 |
| --- | ---- | --- |
| memory | 内存管理 | 打印内存相关日志 |
| dispatch | 算子下发 | 打印算子下发相关日志 |
| dispatch\_time | 算子下发时间 | 打印算子下发时间日志 |
| silent | 静默检测 | 打印静默检测相关日志 |
| recovery | 快恢 | 打印快恢相关日志 |
| op\_plugin | 算子适配 | 打印算子适配相关日志 |
| shmem | 共享内存 | 打印共享内存相关日志 |
| env | 环境变量 | 打印调用环境变量日志 |
| acl | acl | 打印acl相关日志 |
| aclgraph | aclgraph | 打印aclgraph相关日志 |

Ascend Extension for PyTorch对原生的logging日志打印功能进行了增强，支持C++侧的日志打印功能。

- 配置时，开启logging日志信息打印，指定模块的日志信息会正常打印在首节点的屏幕上。
- 未配置时，关闭logging日志信息打印，日志信息不会打印在屏幕上。

此环境变量默认未配置。

## 配置示例

- 开启logging日志信息打印：

    ```bash
    export TORCH_NPU_LOGS=memory,+dispatch,-all
    ```

    级别说明：

    - ：INFO级别，默认级别，输出常规运行信息。

    - +：DEBUG级别，输出最详细的调试信息。

    - -：ERROR级别，仅输出错误和警告信息。

    以上配置示例表示memory（内存管理）打印INFO级别日志，dispatch（算子下发）打印DEBUG级别日志，all（其余所有模块包含原生PyTorch和Ascend Extension for PyTorch新增模块）打印ERROR级别日志信息。

- 关闭logging日志信息打印：

    ```bash
    unset TORCH_NPU_LOGS
    ```

## 使用约束

shmem模块仅在PyTorch2.7.1及以上版本生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 800I A2 推理产品</term>
- <term>Atlas 推理系列产品</term>
