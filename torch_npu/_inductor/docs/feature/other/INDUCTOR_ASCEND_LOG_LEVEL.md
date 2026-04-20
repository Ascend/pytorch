# INDUCTOR_ASCEND_LOG_LEVEL

## 功能描述

设置Inductor Ascend日志等级，控制日志输出的详细程度，默认值为WARNING。大小写不敏感，会自动转换为大写。

| 值 | 说明 |
|---|---|
| DEBUG | 设置日志等级为DEBUG，输出最详细的调试信息 |
| INFO | 设置日志等级为INFO，输出常规信息 |
| WARNING | 设置日志等级为WARNING，输出警告信息（默认值） |
| ERROR | 设置日志等级为ERROR，仅输出错误信息 |
| CRITICAL | 设置日志等级为CRITICAL，仅输出严重错误信息 |

## 配置示例

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>
