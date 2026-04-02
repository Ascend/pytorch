# TORCH\_NPU\_LOGS\_FILTER

## 功能描述

此环境变量用于过滤Ascend Extension for PyTorch日志输出内容，通过黑白名单机制筛选需要显示的日志信息，帮助开发者在大量日志中快速定位关键信息。

配合`TORCH_NPU_LOGS`使用时，可以在开启日志打印的基础上，进一步精细化控制日志输出内容，减少无关日志干扰，提升调试效率。

黑白名单机制说明：

- 白名单（+前缀）：仅显示匹配指定关键字的日志信息。
- 黑名单（-前缀）：过滤掉匹配指定关键字的日志信息。

此环境变量默认未配置。

## 配置示例

以下示例使用op_plugin模块演示过滤功能，该功能同样适用于其他模块（memory、dispatch等）。

- 白名单过滤：

    ```bash
    export TORCH_NPU_LOGS="+op_plugin"
    export TORCH_NPU_LOGS_FILTER="+aclnnAdd,+aclnnMul"
    ```

    以上配置表示开启op_plugin模块的DEBUG级别日志，并仅显示包含"aclnnAdd"或"aclnnMul"算子的日志信息。

- 黑名单过滤：

    ```bash
    export TORCH_NPU_LOGS="+op_plugin"
    export TORCH_NPU_LOGS_FILTER="-aclnnAdd,-aclnnMul"
    ```

    以上配置表示开启op_plugin模块的DEBUG级别日志，并过滤掉包含"aclnnAdd"或"aclnnMul"算子的日志信息。

- 关闭过滤：

    ```bash
    unset TORCH_NPU_LOGS_FILTER
    ```

## 使用约束

- 过滤规则区分大小写。
- 白名单和黑名单可以混合使用，黑名单优先级高于白名单。
- 关键字匹配采用子串匹配方式，即日志内容包含指定关键字即视为匹配。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 800I A2 推理产品</term>
- <term>Atlas 推理系列产品</term>
