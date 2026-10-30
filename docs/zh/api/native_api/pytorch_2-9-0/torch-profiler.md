# torch.profiler

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 在使用支持的profiler接口时，需要将API名称替换为**NPU形式名称**才能使用，已支持的NPU形式profiler接口如下表所示。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/profiler.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.profiler

</div>

## API Reference

### torch.profiler._KinetoProfile

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

</div>

### torch.profiler._KinetoProfile.add_metadata

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.add_metadata)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.add_metadata

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

</div>

### torch.profiler._KinetoProfile.add_metadata_json

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.add_metadata_json

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

### torch.profiler._KinetoProfile.export_chrome_trace

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_chrome_trace

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

<!-- npu="910b,A3" id45 -->
### torch.profiler._KinetoProfile.export_memory_timeline

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_memory_timeline

**产品支持情况**：
<!-- end id45 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->

<!-- npu="910b,A3" id46 -->
</div>
<!-- end id46 -->

### torch.profiler._KinetoProfile.export_stacks

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.export_stacks)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_stacks

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

<!-- npu="910b,A3" id47 -->
### torch.profiler.profile

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.profile](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.profile)

**NPU 形式名称**：torch_npu.profiler.profile

**产品支持情况**：
<!-- end id47 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

<!-- npu="910b,A3" id48 -->
</div>
<!-- end id48 -->

<!-- npu="910b,A3" id49 -->
### torch.profiler.profile.step

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.profile.step](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.profile.step)

**NPU 形式名称**：torch_npu.profiler.profile.step

**产品支持情况**：
<!-- end id49 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->

<!-- npu="910b,A3" id50 -->
</div>
<!-- end id50 -->

<!-- npu="910b,A3" id51 -->
### <code><i>class</i></code> torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.ProfilerAction](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.ProfilerAction)

**NPU 形式名称**：torch_npu.profiler.ProfilerAction

**产品支持情况**：
<!-- end id51 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
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
**原生文档**：[torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.ProfilerActivity)

**NPU 形式名称**：torch_npu.profiler.ProfilerActivity

**产品支持情况**：
<!-- end id54 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->

</div>
<!-- end id59 -->

<!-- npu="910b,A3" id55 -->
### torch.profiler.schedule

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.schedule](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.schedule)

**NPU 形式名称**：torch_npu.profiler.schedule

**产品支持情况**：
<!-- end id55 -->

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->

<!-- npu="910b,A3" id56 -->
</div>
<!-- end id56 -->

<!-- npu="910b,A3" id57 -->
### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.9/profiler.html#torch.profiler.tensorboard_trace_handler)

**NPU 形式名称**：torch_npu.profiler.tensorboard_trace_handler

**产品支持情况**：
<!-- end id57 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->

<!-- npu="910b,A3" id58 -->
</div>
<!-- end id58 -->
