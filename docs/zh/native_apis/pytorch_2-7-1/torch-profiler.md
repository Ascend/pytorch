# torch.profiler

> [!NOTE]
>
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。
>
> 在使用支持的profiler接口时，需要将API名称替换为**NPU形式名称**才能使用，已支持的NPU形式profiler接口如下表所示。

|API名称|NPU形式名称|是否支持|限制与说明|
|--|--|--|--|
|[torch.profiler._KinetoProfile](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile)|torch_npu.profiler._KinetoProfile|是|-|
|[torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.add_metadata)|torch_npu.profiler.profile.add_metadata|是|-|
|[torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)|torch_npu.profiler.profile.add_metadata_json|是|-|
|[torch.profiler._KinetoProfile.events](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.events)|-|否|-|
|[torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)|torch_npu.profiler.profile.export_chrome_trace|是|-|
|[torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)|torch_npu.profiler.profile.export_memory_timeline|是|-|
|[torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.export_stacks)|torch_npu.profiler.profile.export_stacks|是|-|
|[torch.profiler._KinetoProfile.key_averages](https://pytorch.org/docs/2.10/profiler.html#torch.profiler._KinetoProfile.key_averages)|-|否|-|
|[torch.profiler.profile](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.profile)|torch_npu.profiler.profile|是|-|
|[torch.profiler.profile.step](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.profile.step)|torch_npu.profiler.profile.step|是|-|
|[torch.profiler.ProfilerAction](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerAction)|torch_npu.profiler.ProfilerAction|是|-|
|[torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerActivity)|torch_npu.profiler.ProfilerActivity|是|-|
|[torch.profiler.ProfilerActivity.name](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.ProfilerActivity.name)|-|否|-|
|[torch.profiler.schedule](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.schedule)|torch_npu.profiler.schedule|是|-|
|[torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.tensorboard_trace_handler)|torch_npu.profiler.tensorboard_trace_handler|是|-|
|[torch.profiler.itt.is_available](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.is_available)|-|否|-|
|[torch.profiler.itt.mark](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.mark)|-|否|-|
|[torch.profiler.itt.range_push](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.range_push)|-|否|-|
|[torch.profiler.itt.range_pop](https://pytorch.org/docs/2.10/profiler.html#torch.profiler.itt.range_pop)|-|否|-|
