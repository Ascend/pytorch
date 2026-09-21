# TORCH\_LOGS

## 功能描述

通过此环境变量可控制TorchNPU各模块的日志输出级别，用于调试和排查`torch.compile`及其后端（Inductor、Dynamo等）的行为。设置后，`torch_npu`会将日志配置同步到C++层日志上下文。

- 默认值未配置：使用默认日志级别（WARNING）。
- 配置为`+模块名`：输出指定模块的DEBUG级别日志。
- 配置为`-模块名`：仅输出指定模块的ERROR级别日志。
- 配置为`模块名1,模块名2`：同时配置多个模块，输出INFO级别日志。
- 配置为`+模块名1,-模块名2`：混合配置，`+`/`-`前缀逐项生效，可对同一配置中的不同模块设置不同级别。

常用模块名及功能说明如下：

| 模块名 | 功能描述 |
|:---|:---|
| `dynamo` | 打印TorchDynamo图捕获与追踪日志 |
| `aot` | 打印AOTAutograd前反向图构建日志 |
| `inductor` | 打印Inductor编译日志 |
| `dynamic` | 打印动态shape（symbolic shapes）推导日志 |
| `fake_tensor` | 打印FakeTensor相关日志 |
| `autograd` | 打印autograd相关日志 |
| `distributed` | 打印分布式训练相关日志 |
| `c10d` | 打印ProcessGroup集合通信相关日志 |
| `ddp` | 打印DistributedDataParallel相关日志 |
| `fsdp` | 打印FSDP相关日志 |
| `graph` | 以表格形式打印Dynamo追踪得到的FX图 |
| `graph_code` | 以Python代码形式打印Dynamo追踪得到的FX图 |
| `aot_graphs` | 打印AOTAutograd分区后的前向/反向FX图 |
| `graph_breaks` | 打印Dynamo发生图中断（graph break）的位置及原因 |
| `recompiles` | 打印触发重新编译的原因 |
| `guards` | 打印每个已编译Dynamo帧的守卫（guards） |
| `output_code` | 打印Inductor生成的代码（Triton或C++） |
| `kernel_code` | 按kernel打印Inductor生成的代码 |
| `compiled_autograd` | 打印compiled autograd相关日志（含计算图） |

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 环境变量设置`TORCH_LOGS`后，优先级高于`torch._logging.set_logs()`，API调用将不生效。
> - 完整可配置的模块名可通过`TORCH_LOGS="+help"`查看，不同PyTorch版本支持的模块可能存在差异。

该变量对应PyTorch的[TORCH_LOGS](https://docs.pytorch.org/docs/stable/logging.html)，配置方式一致，完整组件与artifact列表请参考该文档。

## 配置示例

输出Inductor的DEBUG级别日志：

```bash
export TORCH_LOGS="+inductor"
```

仅输出指定模块的ERROR级别日志：

```bash
export TORCH_LOGS="-inductor"
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

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT</term>
<!-- end id4 -->
