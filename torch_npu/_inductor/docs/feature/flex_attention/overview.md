# FlexAttention特性介绍

## 概述

FlexAttention通过`score_mod`和`mask_mod`描述注意力分数变换及可见性规则，支持因果注意力、滑动窗口等场景。Inductor-Ascend将这些规则与Attention计算结合，生成在NPU上执行的Triton kernel。

本文对应master分支与配套PyTorch 2.13接口。不同分支的辅助输出接口和调优配置可能不同，请使用与安装版本一致的文档。

## 关键概念

| 概念 | 说明 |
| --- | --- |
| `score_mod` | 接收`(score, b, h, q_idx, kv_idx)`，返回修改后的标量分数，例如加上位置偏置。`score`已包含`scale`缩放。 |
| `mask_mod` | 接收`(b, h, q_idx, kv_idx)`，返回布尔标量；`True`表示该位置参与注意力计算。 |
| `BlockMask` | `create_block_mask`生成的块稀疏元数据，区分完全可见、部分可见和完全不可见的块，使kernel可以跳过完全不可见的块。 |
| GQA | 分组查询注意力。Query的head数可以大于Key/Value的head数，通过`enable_gqa=True`启用。 |

默认计算为`softmax(Q @ K.transpose(-2, -1) * scale) @ V`，`scale`默认为Query最后一维大小的平方根的倒数。`score_mod`修改缩放后的分数，掩码使不可见位置不参与softmax。

## 环境准备

按照[环境准备与安装](../../installation/installation.md)安装匹配版本的PyTorch、TorchNPU、Triton-Ascend和CANN，并确保NPU可用。使用前导入`torch_npu`以完成NPU后端注册。

本文介绍`torch.compile(backend="inductor")`下的Triton-Ascend路径。建议先使用默认配置完成示例，再根据实际场景调整环境变量。

## 使用方法

### 因果注意力与反向传播

将以下代码保存为`flex_attention_example.py`。示例在编译区域外构造掩码，执行前向计算，并通过autograd计算Query、Key、Value的梯度。

```python
import torch
import torch_npu
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


torch.manual_seed(0)
q = torch.randn(1, 4, 256, 64, device="npu", dtype=torch.float16,
                requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

# 掩码与 batch、head 无关，使用 B=None、H=None 复用掩码元数据。
block_mask = create_block_mask(
    causal_mask, B=None, H=None, Q_LEN=256, KV_LEN=256, device="npu"
)
compiled_flex_attention = torch.compile(
    flex_attention, backend="inductor", fullgraph=True
)
out = compiled_flex_attention(q, k, v, block_mask=block_mask)
out.float().square().mean().backward()
torch.npu.synchronize()

print("Out:", tuple(out.shape))
print("dQ/dK/dV:", tuple(q.grad.shape), tuple(k.grad.shape), tuple(v.grad.shape))
```

运行命令：

```shell
python flex_attention_example.py
```

预期输出的张量形状如下；首次运行包含编译开销：

```text
Out: (1, 4, 256, 64)
dQ/dK/dV: (1, 4, 256, 64) (1, 4, 256, 64) (1, 4, 256, 64)
```

### 自定义分数与掩码

`score_mod`和`mask_mod`可以组合使用。下面的回调分别表示相对位置偏置和长度为128的因果滑动窗口；以下片段复用上例的输入和编译函数，仅演示前向调用。自定义分数变换的反向还受配套编译器支持范围影响，应单独验证。

```python
def relative_bias(score, b, h, q_idx, kv_idx):
    return score + (kv_idx - q_idx) * 0.01


def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < 128)


block_mask = create_block_mask(
    sliding_window, B=None, H=None, Q_LEN=256, KV_LEN=256, device="npu"
)
out = compiled_flex_attention(
    q, k, v, score_mod=relative_bias, block_mask=block_mask
)
```

回调应使用可被Dynamo捕获并由NPU lowering支持的张量运算。组合布尔条件时使用`&`、`|`等张量运算，不要对张量条件使用Python的`and`、`or`或依赖其值的`if`语句。

如果掩码依赖batch或head，请为`B`、`H`传入对应维度，而不是`None`。只有掩码规则、捕获的张量内容及适用形状保持一致时才可复用`BlockMask`；规则或内容变化后应重新构造。

### GQA与辅助输出

对于`q.shape=(B, Hq, M, D)`、`k.shape=(B, Hkv, N, D)`、`v.shape=(B, Hkv, N, Dv)`，当`Hq`是`Hkv`的整数倍时，可以在调用中设置`enable_gqa=True`。输出形状为`(B, Hq, M, Dv)`。例如，`Hq=4`、`Hkv=2`表示每两个Query head共享一个Key/Value head。

通过`return_aux=AuxRequest(lse=True)`可以请求log-sum-exp辅助输出，返回`(out, aux)`并从`aux.lse`访问结果；`AuxRequest`从`torch.nn.attention.flex_attention`导入。`AuxRequest(max_scores=True)`还可请求最大分数。辅助输出会影响可用kernel路径，应按实际形状验证；不要同时设置`return_lse=True`和`return_aux`。

## 执行方式与调优

- `BlockMask`用于跳过完全不可见的块。部分可见块仍需要逐元素处理掩码，性能取决于块稀疏程度、序列长度、head维度和目标硬件。
- NPU上的`create_block_mask`在编译区域外、序列长度为正整数且`_compile=False`时，可按Query方向分段构造元数据，降低构造阶段的峰值临时内存。
- 首次编译、掩码构造和稳态执行的耗时应分别测量。计时时先预热，并在测量边界同步NPU。不要将首次编译时间作为稳态算子执行时间。

可用配置见[FlexAttention配置目录](./flex_attention.md)。环境变量应在导入`torch_npu`并开始编译之前设置；变更后使用新进程。

## 使用约束

- Query、Key、Value使用四维张量并放在同一NPU上；本示例使用相同的FP16类型。Key与Value的序列长度和head数应一致，Query与Key的最后一维应一致。head维度至少为16；可用的形状、数据类型和回调组合还受Triton-Ascend编译能力及硬件资源限制。
- `BlockMask`的序列范围须与Query和Key/Value匹配。示例中每行至少有一个可见位置，使用自定义掩码时也应明确全掩码行的预期行为，并检查输出与梯度。
- 反向传播由autograd生成，包括dQ、dK、dV。捕获的可学习分数偏置受上游自动求导和NPU lowering支持范围约束；`mask_mod`捕获的张量不支持梯度。
- head数和head维度按静态维度处理；动态batch、动态序列及不同掩码布局可能触发重新编译或改变可用优化路径。
- `kernel_options`属于底层调优参数，不应照搬CUDA配置到NPU。关闭某项优化不代表关闭`torch.compile`或回到eager执行。

## 支持的型号

- <term>Ascend 950PR&950DT系列产品</term>
