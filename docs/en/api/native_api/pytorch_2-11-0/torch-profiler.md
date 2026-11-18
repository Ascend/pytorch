# torch.profiler

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - When using the supported profiler interfaces, replace the API name with the **NPU-adapted name** before use. The supported NPU-adapted profiler interfaces are listed in the following table.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/profiler.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.profiler

</div>

## API Reference

### torch.profiler._KinetoProfile

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile)

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

**Native Documentation**: [torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.add_metadata)

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

**Native Documentation**: [torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)

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

### torch.profiler._KinetoProfile.export_chrome_trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)

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

<!-- npu="910b,A3" id45 -->
### torch.profiler._KinetoProfile.export_memory_timeline

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_memory_timeline

**Support Status**:
<!-- end id45 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->

<!-- npu="910b,A3" id46 -->
</div>
<!-- end id46 -->

### torch.profiler._KinetoProfile.export_stacks

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.11/profiler.html#torch.profiler._KinetoProfile.export_stacks)

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

<!-- npu="910b,A3" id47 -->
### torch.profiler.profile

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.profile)

**NPU-Adapted Name**: torch_npu.profiler.profile

**Support Status**:
<!-- end id47 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id48 -->
</div>
<!-- end id48 -->

<!-- npu="910b,A3" id49 -->
### torch.profiler.profile.step

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.step](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.profile.step)

**NPU-Adapted Name**: torch_npu.profiler.profile.step

**Support Status**:
<!-- end id49 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->

<!-- npu="910b,A3" id50 -->
</div>
<!-- end id50 -->

<!-- npu="910b,A3" id51 -->
### <code><i>class</i></code> torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.ProfilerAction](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.ProfilerAction)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerAction

**Support Status**:
<!-- end id51 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->

<!-- npu="910b,A3" id52 -->
</div>
<!-- end id52 -->

<!-- npu="910b,A3" id59 -->
<!-- npu="910b,A3" id53 -->
### <code><i>class</i></code> torch.profiler.ProfilerActivity
<!-- end id53 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id54 -->
**Native Documentation**: [torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.ProfilerActivity)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerActivity

**Support Status**:
<!-- end id54 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->

</div>
<!-- end id59 -->

<!-- npu="910b,A3" id55 -->
### torch.profiler.schedule

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.schedule](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.schedule)

**NPU-Adapted Name**: torch_npu.profiler.schedule

**Support Status**:
<!-- end id55 -->

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->

<!-- npu="910b,A3" id56 -->
</div>
<!-- end id56 -->

<!-- npu="910b,A3" id57 -->
### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.11/profiler.html#torch.profiler.tensorboard_trace_handler)

**NPU-Adapted Name**: torch_npu.profiler.tensorboard_trace_handler

**Support Status**:
<!-- end id57 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->

<!-- npu="910b,A3" id58 -->
</div>
<!-- end id58 -->
