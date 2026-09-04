# torch.distributed.tensor

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributed.tensor.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [PyTorch DTensor (Distributed Tensor)](#pytorch-dtensor-distributed-tensor)
- [Different ways to create a DTensor](#different-ways-to-create-a-dtensor)
- [Experimental Features](#experimental-features)

</div>

<div style="display:none;">

## &#8203;torch.distributed.tensor

</div>

## PyTorch DTensor (Distributed Tensor)

### <code><i>class</i></code> torch.distributed.tensor.DTensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.DTensor)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

> <font size="3">from_local()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor.from_local](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.DTensor.from_local)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`local_tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">redistribute()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor.redistribute](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.DTensor.redistribute)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

</div>

### <code><i>class</i></code> torch.distributed.tensor.placement_types.Shard

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.placement_types.Shard](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.placement_types.Shard)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool，complex64，complex128

</div>

### torch.distributed.tensor.placement_types._StridedShard

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.placement_types._StridedShard](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.placement_types._StridedShard)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

## Different ways to create a DTensor

### torch.distributed.tensor.distribute_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.distribute_module](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.distribute_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.distributed.tensor.distribute_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.distribute_tensor](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.distribute_tensor)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

## Experimental Features

### torch.distributed.tensor.experimental.context_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.experimental.context_parallel](https://pytorch.org/docs/2.13/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：仅支持NPU fused SDPA路径；`q`/`k`/`v`仅支持BNSD布局；暂不支持`pse`、`padding_mask`、`prefix`、`actual_seq_qlen`、`actual_seq_kvlen`、`sink`以及任意非causal的attention mask；启用load balance时要求使用causal attention；暂不支持通过`torch.compile`编译为计算图

</div>
