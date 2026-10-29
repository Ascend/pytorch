# torch.accelerator

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.14/accelerator.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Memory management](#memory-management)

</div>

<div style="display:none;">

## &#8203;torch.accelerator

</div>

### torch.accelerator.device_count

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.device_count](https://pytorch.org/docs/2.14/generated/torch.accelerator.device_count.html)

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

### torch.accelerator.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.is_available](https://pytorch.org/docs/2.14/generated/torch.accelerator.is_available.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### torch.accelerator.current_accelerator

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_accelerator](https://pytorch.org/docs/2.14/generated/torch.accelerator.current_accelerator.html)

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

### torch.accelerator.set_device_index

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_device_index](https://pytorch.org/docs/2.14/generated/torch.accelerator.set_device_index.html)

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

### torch.accelerator.set_device_idx

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_device_idx](https://pytorch.org/docs/2.14/generated/torch.accelerator.set_device_idx.html)

**Version Notes**: This API has been deprecated; use `torch.accelerator.set_device_index` instead.

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

</div>

### torch.accelerator.current_device_index

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_device_index](https://pytorch.org/docs/2.14/generated/torch.accelerator.current_device_index.html)

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

### torch.accelerator.current_device_idx

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_device_idx](https://pytorch.org/docs/2.14/generated/torch.accelerator.current_device_idx.html)

**Version Notes**: This API has been deprecated; use `torch.accelerator.current_device_index` instead.

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

### torch.accelerator.get_device_capability

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.get_device_capability](https://pytorch.org/docs/2.14/generated/torch.accelerator.get_device_capability.html)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

### torch.accelerator.set_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_stream](https://pytorch.org/docs/2.14/generated/torch.accelerator.set_stream.html)

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

</div>

### torch.accelerator.current_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_stream](https://pytorch.org/docs/2.14/generated/torch.accelerator.current_stream.html)

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

</div>

### torch.accelerator.synchronize

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.synchronize](https://pytorch.org/docs/2.14/generated/torch.accelerator.synchronize.html)

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

</div>

### torch.accelerator.device_index

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.device_index](https://pytorch.org/docs/2.14/generated/torch.accelerator.device_index.html)

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

## Memory management

### torch.accelerator.memory.empty_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.empty_cache](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.empty_cache.html)

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

</div>

### torch.accelerator.memory.empty_host_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.empty_host_cache](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.empty_host_cache.html)

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

</div>

### torch.accelerator.memory.get_memory_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.get_memory_info](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.get_memory_info.html)

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

</div>

### torch.accelerator.memory.max_memory_allocated

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.max_memory_allocated](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.max_memory_allocated.html)

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

### torch.accelerator.memory.max_memory_reserved

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.max_memory_reserved](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.max_memory_reserved.html)

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

</div>

### torch.accelerator.memory.memory_allocated

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.memory_allocated](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.memory_allocated.html)

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

</div>

### torch.accelerator.memory.memory_reserved

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.memory_reserved](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.memory_reserved.html)

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

### torch.accelerator.memory.memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.memory_stats](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.memory_stats.html)

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

</div>

### torch.accelerator.memory.reset_accumulated_memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.reset_accumulated_memory_stats](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.reset_accumulated_memory_stats.html)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

### torch.accelerator.memory.reset_peak_memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.memory.reset_peak_memory_stats](https://pytorch.org/docs/2.14/generated/torch.accelerator.memory.reset_peak_memory_stats.html)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>
