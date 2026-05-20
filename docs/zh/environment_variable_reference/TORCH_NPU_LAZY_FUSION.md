# TORCH\_NPU\_LAZY\_FUSION

## 功能描述

通过此环境变量可开启Ascend Extension for PyTorch Eager模式下的DVM算子融合。DVM将多个相邻的小算子合并成单个融合kernel，减少kernel下发次数和中间张量的搬运，从而加速训练和推理。

- 配置为"True"时：开启DVM算子融合。
- 未配置或配置为"False"时：关闭DVM算子融合。

此环境变量默认为未配置。

## 配置示例

```bash
export TORCH_NPU_LAZY_FUSION=True
```

## 使用约束

- 仅在[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)为1或2时生效，否则自动禁用算子融合。
- 仅在主线程及其反向线程生效，其它独立线程（如dataloader worker）自动禁用。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
