# torch.distributed.tensor

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### torch.distributed.tensor.distribute_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.distribute_module](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.distribute_module)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.distributed.tensor.distribute_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.distribute_tensor](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.distribute_tensor)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

### _`class`_ torch.distributed.tensor.DTensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.DTensor)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

> <font size="3">from_local()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor.from_local](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.DTensor.from_local)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">redistribute()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.DTensor.redistribute](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.DTensor.redistribute)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

</div>

### _`class`_ torch.distributed.tensor.placement_types.Shard

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.placement_types.Shard](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.placement_types.Shard)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool，complex64，complex128

</div>

### torch.distributed.tensor.experimental.context_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.experimental.context_parallel](https://pytorch.org/docs/2.10/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 仅支持NPU fused SDPA路径；q/k/v仅支持BNSD布局；暂不支持pse、padding_mask、prefix、actual_seq_qlen、actual_seq_kvlen、sink以及任意非causal attention mask；load balance要求causal attention；暂不支持通过torch.compile编译为计算图

</div>
