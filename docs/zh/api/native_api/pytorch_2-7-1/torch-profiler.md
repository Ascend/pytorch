# torch.profiler

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
>
> 在使用支持的profiler接口时，需要将API名称替换为**NPU形式名称**才能使用，已支持的NPU形式profiler接口如下表所示。

## 目录

- [base API](#base-api)
- [API Reference](#api-reference)

## base API

### torch.profiler._KinetoProfile

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.add_metadata

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.add_metadata](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.add_metadata)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.add_metadata

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.add_metadata_json

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.add_metadata_json](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.add_metadata_json)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.add_metadata_json

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.events

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.events](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.events)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.export_chrome_trace

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_chrome_trace](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_chrome_trace

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.export_memory_timeline

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_memory_timeline](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_memory_timeline

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.export_stacks

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.export_stacks](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.export_stacks)

**NPU 形式名称**：torch_npu.profiler._KinetoProfile.export_stacks

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler._KinetoProfile.key_averages

<div style="margin-left: 2em">

**原生文档**：[torch.profiler._KinetoProfile.key_averages](https://pytorch.org/docs/2.7/profiler.html#torch.profiler._KinetoProfile.key_averages)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.profile

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.profile](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.profile)

**NPU 形式名称**：torch_npu.profiler.profile

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.profile.step

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.profile.step](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.profile.step)

**NPU 形式名称**：torch_npu.profiler.profile.step

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.profiler.ProfilerAction

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.ProfilerAction](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerAction)

**NPU 形式名称**：torch_npu.profiler.ProfilerAction

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.profiler.ProfilerActivity

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.ProfilerActivity](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerActivity)

**NPU 形式名称**：torch_npu.profiler.ProfilerActivity

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.ProfilerActivity.name](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.ProfilerActivity.name)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### torch.profiler.itt.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.itt.is_available](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.itt.mark

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.itt.mark](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.mark)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.itt.range_push

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.itt.range_push](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.range_push)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.itt.range_pop

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.itt.range_pop](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.itt.range_pop)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## API Reference

### torch.profiler.schedule

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.schedule](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.schedule)

**NPU 形式名称**：torch_npu.profiler.schedule

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.profiler.tensorboard_trace_handler

<div style="margin-left: 2em">

**原生文档**：[torch.profiler.tensorboard_trace_handler](https://pytorch.org/docs/2.7/profiler.html#torch.profiler.tensorboard_trace_handler)

**NPU 形式名称**：torch_npu.profiler.tensorboard_trace_handler

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
