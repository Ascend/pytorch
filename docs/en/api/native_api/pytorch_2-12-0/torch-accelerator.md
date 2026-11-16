# torch.accelerator

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/accelerator.html).

<div style="display:none;">

## &#8203;torch.accelerator

</div>

<!-- npu="910b,A3" id37 -->
### torch.accelerator.device_count

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.device_count](https://pytorch.org/docs/2.12/generated/torch.accelerator.device_count.html)

**Support Status**:
<!-- end id37 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id38 -->
</div>
<!-- end id38 -->

### torch.accelerator.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.is_available](https://pytorch.org/docs/2.12/generated/torch.accelerator.is_available.html)

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

**Native Documentation**: [torch.accelerator.current_accelerator](https://pytorch.org/docs/2.12/generated/torch.accelerator.current_accelerator.html)

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

<!-- npu="910b,A3" id39 -->
### torch.accelerator.set_device_index

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_device_index](https://pytorch.org/docs/2.12/generated/torch.accelerator.set_device_index.html)

**Support Status**:
<!-- end id39 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->

<!-- npu="910b,A3" id40 -->
</div>
<!-- end id40 -->

<!-- npu="910b,A3" id41 -->
### torch.accelerator.set_device_idx

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_device_idx](https://pytorch.org/docs/2.12/generated/torch.accelerator.set_device_idx.html)

**Support Status**:
<!-- end id41 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id42 -->
</div>
<!-- end id42 -->

### torch.accelerator.current_device_index

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_device_index](https://pytorch.org/docs/2.12/generated/torch.accelerator.current_device_index.html)

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

<!-- npu="910b,A3" id43 -->
### torch.accelerator.current_device_idx

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.current_device_idx](https://pytorch.org/docs/2.12/generated/torch.accelerator.current_device_idx.html)

**Support Status**:
<!-- end id43 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->

<!-- npu="910b,A3" id44 -->
</div>
<!-- end id44 -->

### torch.accelerator.set_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.accelerator.set_stream](https://pytorch.org/docs/2.12/generated/torch.accelerator.set_stream.html)

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

**Native Documentation**: [torch.accelerator.current_stream](https://pytorch.org/docs/2.12/generated/torch.accelerator.current_stream.html)

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

**Native Documentation**: [torch.accelerator.synchronize](https://pytorch.org/docs/2.12/generated/torch.accelerator.synchronize.html)

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

**Native Documentation**: [torch.accelerator.device_index](https://pytorch.org/docs/2.12/generated/torch.accelerator.device_index.html)

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