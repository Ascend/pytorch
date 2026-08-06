# torch.utils.checkpoint

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.utils.checkpoint.checkpoint

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.checkpoint](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.checkpoint)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### _`class`_ torch.utils.checkpoint.CheckpointPolicy

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.CheckpointPolicy](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.CheckpointPolicy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### _`class`_ torch.utils.checkpoint.SelectiveCheckpointContext

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.SelectiveCheckpointContext](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.SelectiveCheckpointContext)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.utils.checkpoint.create_selective_checkpoint_contexts

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.create_selective_checkpoint_contexts](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.utils.checkpoint.set_checkpoint_debug_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.set_checkpoint_debug_enabled](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.utils.checkpoint.checkpoint_sequential

<div style="margin-left: 2em">

**原生文档**：[torch.utils.checkpoint.checkpoint_sequential](https://pytorch.org/docs/2.7/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>
