# torch.profiler

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - When using the supported profiler interfaces, replace the API name with the **NPU-adapted name** before use. The supported NPU-adapted profiler interfaces are listed in the following table.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/profiler.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.profiler

</div>

## API Reference

<!-- npu="910b,A3" id60 -->
### torch.profiler.schedule

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.schedule](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profiler.schedule)

**NPU-Adapted Name**: torch_npu.profiler.schedule

**Support Status**:
<!-- end id60 -->

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->

<!-- npu="910b,A3" id61 -->
</div>
<!-- end id61 -->

<!-- npu="910b,A3" id62 -->
### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.tensorboard_trace_handler](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profiler.tensorboard_trace_handler)

**NPU-Adapted Name**: torch_npu.profiler.tensorboard_trace_handler

**Support Status**:
<!-- end id62 -->

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->

<!-- npu="910b,A3" id63 -->
</div>
<!-- end id63 -->

<!-- npu="910b,A3" id52 -->
### torch.profiler.profile

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile](https://pytorch.org/docs/2.13/profiler.html#torch.profiler.profile)

**NPU-Adapted Name**: torch_npu.profiler.profile

**Support Status**:
<!-- end id52 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->

<!-- npu="910b,A3" id53 -->
</div>
<!-- end id53 -->

<!-- npu="910b,A3" id42 -->
### torch.profiler.profile.add_metadata

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.add_metadata](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.add_metadata)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata

**Support Status**:
<!-- end id42 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id43 -->
</div>
<!-- end id43 -->

<!-- npu="910b,A3" id44 -->
### torch.profiler.profile.add_metadata_json

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.add_metadata_json](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.add_metadata_json)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata_json

**Support Status**:
<!-- end id44 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id45 -->
</div>
<!-- end id45 -->

<!-- npu="910b,A3" id46 -->
### torch.profiler.profile.export_chrome_trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.export_chrome_trace](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.export_chrome_trace)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_chrome_trace

**Support Status**:
<!-- end id46 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->

<!-- npu="910b,A3" id47 -->
</div>
<!-- end id47 -->

<!-- npu="910b,A3" id48 -->
### torch.profiler.profile.export_memory_timeline

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.export_memory_timeline](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.export_memory_timeline)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_memory_timeline

**Support Status**:
<!-- end id48 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id49 -->
</div>
<!-- end id49 -->

<!-- npu="910b,A3" id50 -->
### torch.profiler.profile.export_stacks

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.export_stacks](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.export_stacks)

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_stacks

**Support Status**:
<!-- end id50 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->

<!-- npu="910b,A3" id51 -->
</div>
<!-- end id51 -->

<!-- npu="910b,A3" id58 -->
### <code><i>class</i></code> torch.profiler.ProfilerActivity
<!-- end id58 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id59 -->
**Native Documentation**: [torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.13/profiler.html#torch.profiler.ProfilerActivity)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerActivity

**Support Status**:
<!-- end id59 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->

</div>

<!-- npu="910b,A3" id54 -->
### torch.profiler.profile.step

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.profile.step](https://pytorch.org/docs/2.13/profiler.html#torch.profiler.profile.step)

**NPU-Adapted Name**: torch_npu.profiler.profile.step

**Support Status**:
<!-- end id54 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id55 -->
</div>
<!-- end id55 -->

<!-- npu="910b,A3" id56 -->
### <code><i>class</i></code> torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**Native Documentation**: [torch.profiler.ProfilerAction](https://pytorch.org/docs/2.13/profiler.html#torch.profiler.ProfilerAction)

**NPU-Adapted Name**: torch_npu.profiler.ProfilerAction

**Support Status**:
<!-- end id56 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->

<!-- npu="910b,A3" id57 -->
</div>
<!-- end id57 -->