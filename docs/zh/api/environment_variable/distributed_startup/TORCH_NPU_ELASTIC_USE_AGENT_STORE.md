# TORCH\_NPU\_ELASTIC\_USE\_AGENT\_STORE

## 功能描述

通过此环境变量可控制是否使用agent已启动的ParallelStore。启用后，所有worker连接agent已启动的ParallelStore，并包裹`PrefixStore("/worker/attempt_<n>")`前缀来隔离每次重启。

- 精确等于字符串"True"（大小写敏感）：启用agent store模式。
- 其他值：关闭。

默认值：未设置。

> [!NOTE]
>
> - 当启用agent store时，需要同时设置`TORCHELASTIC_RESTART_COUNT`环境变量（由torchrun/elastic启动器注入），否则会抛出`KeyError`。
> - 使用`torch_npu_run`启动时，`_create_parallel_handler`中会通过`setdefault`默认写入`TORCH_NPU_ELASTIC_USE_AGENT_STORE=True`，通常无需手动配置此变量。

该变量对应PyTorch的`TORCHELASTIC_USE_AGENT_STORE`，在TorchNPU中用于控制是否使用agent已启动的ParallelStore。

## 配置示例

```bash
export TORCH_NPU_ELASTIC_USE_AGENT_STORE=True
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
