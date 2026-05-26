# torch.distributed.tensor

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.tensor.distribute_module|是|支持bf16，fp16，fp32|
|torch.distributed.tensor.distribute_tensor|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool|
|torch.distributed.tensor.DTensor|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128|
|torch.distributed.tensor.DTensor.from_local|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128|
|torch.distributed.tensor.DTensor.redistribute|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool|
|torch.distributed.tensor.placement_types.Shard|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool，complex64，complex128|
|torch.distributed.tensor.placement_types._StridedShard|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool|
|torch.distributed.tensor.experimental.context_parallel|是|仅支持NPU fused SDPA路径；q/k/v仅支持BNSD布局；暂不支持pse、padding_mask、prefix、actual_seq_qlen、actual_seq_kvlen、sink以及任意非causal attention mask；load balance要求causal attention。|
