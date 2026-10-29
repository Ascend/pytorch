# torch.utils.data

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/data.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Memory Pinning](#memory-pinning)

</div>

<div style="display:none;">

## &#8203;torch.utils.data

</div>

## Memory Pinning

### <code><i>class</i></code> torch.utils.data.TensorDataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.TensorDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.TensorDataset)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.utils.data.StackDataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.StackDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.StackDataset)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.utils.data.ConcatDataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.ConcatDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.ConcatDataset)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.utils.data.ChainDataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.ChainDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.ChainDataset)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

### torch.utils.data._utils.collate.collate

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data._utils.collate.collate](https://pytorch.org/docs/2.10/data.html#torch.utils.data._utils.collate.collate)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

**Constraints & Notes**: `batch` only supports int64

</div>

### torch.utils.data.default_convert

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.default_convert](https://pytorch.org/docs/2.10/data.html#torch.utils.data.default_convert)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

### torch.utils.data.random_split

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.random_split](https://pytorch.org/docs/2.10/data.html#torch.utils.data.random_split)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

### <code><i>class</i></code> torch.utils.data.SequentialSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.SequentialSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.SequentialSampler)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

**Constraints & Notes**: `data_source` only supports fp32

</div>

### <code><i>class</i></code> torch.utils.data.RandomSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.RandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.RandomSampler)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

**Constraints & Notes**: `data_source` only supports fp32

</div>

### <code><i>class</i></code> torch.utils.data.SubsetRandomSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.SubsetRandomSampler)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

**Constraints & Notes**: `indices` only supports fp32

</div>

### <code><i>class</i></code> torch.utils.data.WeightedRandomSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.WeightedRandomSampler)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

**Constraints & Notes**: `weights` only supports fp32

</div>

### <code><i>class</i></code> torch.utils.data.BatchSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.BatchSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.BatchSampler)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### <code><i>class</i></code> torch.utils.data.distributed.DistributedSampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.distributed.DistributedSampler)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

**Constraints & Notes**: `dataset` only supports int32

</div>

### <code><i>class</i></code> torch.utils.data.DataLoader

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.DataLoader](https://pytorch.org/docs/2.10/data.html#torch.utils.data.DataLoader)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

**Constraints & Notes**:

- `dataset` only supports fp32
- Does not support loading NPU data through multiple processes. `num_workers` only supports 0.

</div>

### <code><i>class</i></code> torch.utils.data.Dataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.Dataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Dataset)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.utils.data.IterableDataset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.IterableDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.IterableDataset)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

### <code><i>class</i></code> torch.utils.data.Subset

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.Subset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Subset)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

**Constraints & Notes**: `dataset` only supports int64

</div>

### torch.utils.data.default_collate

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.default_collate](https://pytorch.org/docs/2.10/data.html#torch.utils.data.default_collate)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

**Constraints & Notes**: `batch` only supports fp64, int64, bool

</div>

### torch.utils.data.get_worker_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.get_worker_info](https://pytorch.org/docs/2.10/data.html#torch.utils.data.get_worker_info)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

### <code><i>class</i></code> torch.utils.data.Sampler

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.data.Sampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Sampler)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

**Constraints & Notes**: `data_source` only supports int64

</div>
