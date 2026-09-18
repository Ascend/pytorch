# TORCH\_NPU\_ELASTIC\_USE\_AGENT\_STORE

## 功能描述

该环境变量用于控制是否复用Agent已启动的ParallelStore。启用后，所有Worker将连接至该ParallelStore，并通过`PrefixStore("/worker/attempt_<n>")`前缀实现重启隔离。

- 配置为字符串`True`（区分大小写）：启用agent store模式。
- 配置为其他值：关闭agent store模式。
- 默认值为未配置：关闭agent store模式。

> [!NOTE]
>
> 当启用agent store时，需要同时设置`TORCHELASTIC_RESTART_COUNT`环境变量（由torchrun/elastic启动器注入），否则会抛出`KeyError`。使用`torch_npu_run`启动时，`_create_parallel_handler`中会通过`setdefault`默认写入`TORCH_NPU_ELASTIC_USE_AGENT_STORE=True`，通常无需手动配置此变量。

该变量对应PyTorch的[`TORCHELASTIC_USE_AGENT_STORE`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/rendezvous.py?utm_source=chatgpt.com)，配置方式一致。

## 配置示例

```bash
export TORCH_NPU_ELASTIC_USE_AGENT_STORE=True
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
