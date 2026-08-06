# torch.distributed.fsdp.fully_shard

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.distributed.fsdp.fully_shard

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.fully_shard](https://pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.fully_shard)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>
