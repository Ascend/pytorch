# torch.profiler

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - When using supported profiler interfaces, replace the API name with the **NPU-Adapted Name** before use. The supported NPU-adapted profiler interfaces are listed in the following table.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/profiler.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)
- [Intel Instrumentation and Tracing Technology APIs](#intel-instrumentation-and-tracing-technology-apis)

</div>

<div style="display:none;">

## &#8203;torch.profiler

</div>

## API Reference

### torch.profiler._KinetoProfile

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile

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

### torch.profiler._KinetoProfile.add_metadata

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.add_metadata)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata

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

### torch.profiler._KinetoProfile.add_metadata_json

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata_json

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

### torch.profiler._KinetoProfile.events

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.events](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.events)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id12 -->

</div>

### torch.profiler._KinetoProfile.export_chrome_trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_chrome_trace

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

### torch.profiler._KinetoProfile.export_memory_timeline

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_memory_timeline

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

</div>

### torch.profiler._KinetoProfile.export_stacks

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_stacks)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_stacks

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

### torch.profiler._KinetoProfile.key_averages

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.key_averages](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.key_averages)

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

### torch.profiler.profile

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.profile)

**NPU-Adapted Name**: torch_npu.profiler.profile

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

</div>

### torch.profiler.profile.step

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.step](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.profile.step)

**NPU-Adapted Name**: torch_npu.profiler.profile.step

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id30 -->

</div>

### <code><i>class</i></code> torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.ProfilerAction](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerAction)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerAction

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id33 -->

</div>

### <code><i>class</i></code> torch.profiler.ProfilerActivity

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerActivity)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerActivity

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.ProfilerActivity.name](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerActivity.name)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

</div>

### torch.profiler.schedule

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.schedule](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.schedule)

**NPU-Adapted Name**: torch_npu.profiler.schedule

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id42 -->

</div>

### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.tensorboard_trace_handler)

**NPU-Adapted Name**: torch_npu.profiler.tensorboard_trace_handler

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id45 -->

</div>

## Intel Instrumentation and Tracing Technology APIs

### torch.profiler.itt.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.itt.is_available](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.is_available)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id48 -->

</div>

### torch.profiler.itt.mark

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.itt.mark](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.mark)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id51 -->

</div>

### torch.profiler.itt.range_push

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.itt.range_push](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.range_push)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id54 -->

</div>

### torch.profiler.itt.range_pop

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.itt.range_pop](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.range_pop)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id57 -->

</div>
