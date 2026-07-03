# torch.profiler

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:09:51.606Z pushedAt=2026-06-15T02:04:36.608Z -->

> [!NOTE]
>
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.
>
> When using supported profiler APIs, replace the API name with the **NPU-Adapted Name**. The supported profiler APIs with NPU-adapted names are shown in the table below.

|API Name|NPU-Adapted Name|Supported|Restrictions and Notes|
|--|--|--|--|
|[torch.profiler._KinetoProfile](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile)|torch_npu.profiler._KinetoProfile|Yes|-|
|[torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.add_metadata)|torch_npu.profiler.profile.add_metadata|Yes|-|
|[torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)|torch_npu.profiler.profile.add_metadata_json|Yes|-|
|[torch.profiler._KinetoProfile.events](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.events)|-|No|-|
|[torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)|torch_npu.profiler.profile.export_chrome_trace|Yes|-|
|[torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)|torch_npu.profiler.profile.export_memory_timeline|Yes|-|
|[torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_stacks)|torch_npu.profiler.profile.export_stacks|Yes|-|
|[torch.profiler._KinetoProfile.key_averages](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.key_averages)|-|No|-|
|[torch.profiler.profile](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.profile)|torch_npu.profiler.profile|Yes|-|
|[torch.profiler.profile.step](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.profile.step)|torch_npu.profiler.profile.step|Yes|-|
|[torch.profiler.ProfilerAction](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerAction)|torch_npu.profiler.ProfilerAction|Yes|-|
|[torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerActivity)|torch_npu.profiler.ProfilerActivity|Yes|-|
|[torch.profiler.ProfilerActivity.name](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerActivity.name)|-|No|-|
|[torch.profiler.schedule](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.schedule)|torch_npu.profiler.schedule|Yes|-|
|[torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.tensorboard_trace_handler)|torch_npu.profiler.tensorboard_trace_handler|Yes|-|
|[torch.profiler.itt.is_available](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.is_available)|-|No|-|
|[torch.profiler.itt.mark](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.mark)|-|No|-|
|[torch.profiler.itt.range_push](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.range_push)|-|No|-|
|[torch.profiler.itt.range_pop](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.range_pop)|-|No|-|
