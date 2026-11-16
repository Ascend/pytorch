# torch.cpu

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/cpu.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Streams and events](#streams-and-events)

</div>

<div style="display:none;">

## &#8203;torch.cpu

</div>

<!-- npu="910b,A3" id22 -->
### torch.cpu.current_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.current_stream](https://pytorch.org/docs/2.13/generated/torch.cpu.current_stream.html)

**Support Status**:
<!-- end id22 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id23 -->
</div>
<!-- end id23 -->

<!-- npu="910b,A3" id24 -->
### torch.cpu.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.is_available](https://pytorch.org/docs/2.13/generated/torch.cpu.is_available.html)

**Support Status**:
<!-- end id24 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id25 -->
</div>
<!-- end id25 -->

<!-- npu="910b,A3" id26 -->
### torch.cpu.synchronize

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.synchronize](https://pytorch.org/docs/2.13/generated/torch.cpu.synchronize.html)

**Support Status**:
<!-- end id26 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id27 -->
</div>
<!-- end id27 -->

### torch.cpu.stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.stream](https://docs.pytorch.org/docs/2.13/generated/torch.cpu.stream_function.html#torch.cpu.stream)

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

<!-- npu="910b,A3" id28 -->
### torch.cpu.device_count

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.device_count](https://pytorch.org/docs/2.13/generated/torch.cpu.device_count.html)

**Support Status**:
<!-- end id28 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id29 -->
</div>
<!-- end id29 -->

### <code><i>class</i></code> torch.cpu.StreamContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.StreamContext](https://pytorch.org/docs/2.13/generated/torch.cpu.StreamContext.html)

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

## Streams and events

### <code><i>class</i></code> torch.cpu.Stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.Stream](https://pytorch.org/docs/2.13/cpu.html#torch.cpu.Stream)

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