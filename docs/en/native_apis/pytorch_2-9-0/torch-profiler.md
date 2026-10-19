# torch.profiler

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/profiler.html).

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

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.add_metadata

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.add_metadata_json

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.add_metadata_json

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.events

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.export_chrome_trace

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_chrome_trace

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.export_memory_timeline

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_memory_timeline

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.export_stacks

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler._KinetoProfile.export_stacks

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler._KinetoProfile.key_averages

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.profile

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.profile

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.profile.step

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.profile.step

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.ProfilerAction

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.profiler.ProfilerActivity

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.ProfilerActivity

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### torch.profiler.schedule

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.schedule

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.profiler.tensorboard_trace_handler

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Intel Instrumentation and Tracing Technology APIs

### torch.profiler.itt.is_available

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.itt.mark

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.itt.range_push

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.profiler.itt.range_pop

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>
