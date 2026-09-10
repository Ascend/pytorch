# INDUCTOR\_ASCEND\_LOG\_LEVEL

## 功能描述

通过此环境变量可配置Inductor模块的日志级别，控制日志输出的详细程度。

- 默认值为`WARNING`，输出警告信息。
- 配置为`DEBUG`：最详细的调试信息。
- 配置为`INFO`：一般信息。
- 配置为`ERROR`：错误信息。
- 配置为`CRITICAL`：严重错误信息。

> [!NOTE]
>
> - 此环境变量在`torch_npu._inductor`模块初始化时读取并缓存，运行中修改不会生效。

PyTorch通过`torch._inductor.config.log_level`配置日志级别，TorchNPU通过此环境变量提供相关配置。

## 配置示例

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
