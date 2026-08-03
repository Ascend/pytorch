# torch.amp

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)
- [Autocasting](#autocasting)

## base API

### torch.cuda.amp.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.autocast](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.autocast)

**NPU 形式名称**：torch_npu.npu.amp.autocast

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.cuda.amp.custom_bwd

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.custom_bwd](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.custom_bwd)

**NPU 形式名称**：torch_npu.npu.amp.custom_bwd

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.cpu.amp.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.cpu.amp.autocast](https://pytorch.org/docs/2.7/amp.html#torch.cpu.amp.autocast)

**NPU 形式名称**：torch.cpu.amp.autocast

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.cuda.amp.GradScaler

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.GradScaler](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.GradScaler)

**NPU 形式名称**：torch_npu.npu.amp.GradScaler

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

## Autocasting

### torch.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.autocast](https://pytorch.org/docs/2.7/amp.html#torch.autocast)

**NPU 形式名称**：torch.autocast

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.cuda.amp.custom_fwd

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.custom_fwd](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.custom_fwd)

**NPU 形式名称**：torch_npu.npu.amp.custom_fwd

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>
