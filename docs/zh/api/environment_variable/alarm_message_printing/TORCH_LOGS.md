# TORCH\_LOGS

## 功能描述

通过此环境变量可控制PyTorch各模块的日志输出级别，用于调试和排查`torch.compile`及其后端（Inductor、Dynamo等）的行为。设置后，`torch_npu`会将日志配置同步到C++层日志上下文。

- 默认值未配置，使用默认日志级别（WARNING）。
- 配置模块日志：指定需要输出日志的模块及级别。

**语法格式**：

- `TORCH_LOGS="+module"` — 输出指定模块的DEBUG级别日志
- `TORCH_LOGS="-module"` — 仅输出指定模块的ERROR级别日志
- `TORCH_LOGS="module1,module2"` — 同时配置多个模块（INFO级别）
- `TORCH_LOGS="+module1,-module2"` — 混合配置：`+`/`-`前缀逐项生效，可对同一配置中的不同模块设置不同级别

常用模块名：`inductor`、`dynamo`、`aot`、`distribute`等。

> [!NOTE]
>
> - `TORCH_LOGS`（或`TORCH_NPU_LOGS`）设置后，优先级高于`torch._logging.set_logs()` API调用，API调用将不生效。
> - 对于TorchNPU新增的模块（如memory、dispatch、acl等），需使用`TORCH_NPU_LOGS`配置，`TORCH_LOGS`不支持这些模块。
> - 在`INDUCTOR_ASCEND_DEBUG=1`时，等效于`TORCH_LOGS="+inductor"`对`torch._inductor` `logger`的效果。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

输出Inductor的DEBUG级别日志：

```bash
export TORCH_LOGS="+inductor"
```

同时配置多个模块：

```bash
export TORCH_LOGS="+inductor,+dynamo"
```

混合配置（不同模块不同级别，`+`/`-`可混用）：

```bash
export TORCH_LOGS="+inductor,-dynamo"
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
