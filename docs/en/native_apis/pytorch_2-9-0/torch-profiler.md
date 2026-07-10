# torch.profiler

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:39:12.420Z pushedAt=2026-07-09T08:44:08.368Z -->

> [!NOTE]
> If the "Supported" column shows "Yes" and the "Restrictions and Notes" column shows "-", it means the API support is consistent with the native API.
>
> When using supported profiler APIs, you need to replace the API name with the NPU-adapted name. The supported NPU-adapted profiler APIs are listed in the table below.

|API Name|NPU-adapted Name|Supported|Restrictions and Notes|
|--|--|--|--|
|torch.profiler._KinetoProfile|torch_npu.profiler._KinetoProfile|Yes|-|
|torch.profiler._KinetoProfile.add_metadata|torch_npu.profiler.profile.add_metadata|Yes|-|
|torch.profiler._KinetoProfile.add_metadata_json|torch_npu.profiler.profile.add_metadata_json|Yes|-|
|torch.profiler._KinetoProfile.events|-|No|-|
|torch.profiler._KinetoProfile.export_chrome_trace|torch_npu.profiler.profile.export_chrome_trace|Yes|-|
|torch.profiler._KinetoProfile.export_memory_timeline|torch_npu.profiler.profile.export_memory_timeline|Yes|-|
|torch.profiler._KinetoProfile.export_stacks|torch_npu.profiler.profile.export_stacks|Yes|-|
|torch.profiler._KinetoProfile.key_averages|-|No|-|
|torch.profiler.profile|torch_npu.profiler.profile|Yes|-|
|torch.profiler.profile.step|torch_npu.profiler.profile.step|Yes|-|
|torch.profiler.ProfilerAction|torch_npu.profiler.ProfilerAction|Yes|-|
|torch.profiler.ProfilerActivity|torch_npu.profiler.ProfilerActivity|Yes|-|
|torch.profiler.ProfilerActivity.name|-|No|-|
|torch.profiler.schedule|torch_npu.profiler.schedule|Yes|-|
|torch.profiler.tensorboard_trace_handler|torch_npu.profiler.tensorboard_trace_handler|Yes|-|
|torch.profiler.itt.is_available|-|No|-|
|torch.profiler.itt.mark|-|No|-|
|torch.profiler.itt.range_push|-|No|-|
|torch.profiler.itt.range_pop|-|No|-|
