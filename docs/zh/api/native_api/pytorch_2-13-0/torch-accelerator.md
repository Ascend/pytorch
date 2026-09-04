# torch.accelerator

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/accelerator.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Memory management](#memory-management)

</div>

<div style="display:none;">

## &#8203;torch.accelerator

</div>

### torch.accelerator.device_count

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.device_count](https://pytorch.org/docs/2.13/generated/torch.accelerator.device_count.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.is_available](https://pytorch.org/docs/2.13/generated/torch.accelerator.is_available.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.current_accelerator

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.current_accelerator](https://pytorch.org/docs/2.13/generated/torch.accelerator.current_accelerator.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.set_device_index

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.set_device_index](https://pytorch.org/docs/2.13/generated/torch.accelerator.set_device_index.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.set_device_idx

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.set_device_idx](https://pytorch.org/docs/2.13/generated/torch.accelerator.set_device_idx.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.current_device_index

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.current_device_index](https://pytorch.org/docs/2.13/generated/torch.accelerator.current_device_index.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.current_device_idx

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.current_device_idx](https://pytorch.org/docs/2.13/generated/torch.accelerator.current_device_idx.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.get_device_capability

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.get_device_capability](https://pytorch.org/docs/2.13/generated/torch.accelerator.get_device_capability.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.accelerator.set_stream

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.set_stream](https://pytorch.org/docs/2.13/generated/torch.accelerator.set_stream.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.current_stream

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.current_stream](https://pytorch.org/docs/2.13/generated/torch.accelerator.current_stream.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.synchronize

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.synchronize](https://pytorch.org/docs/2.13/generated/torch.accelerator.synchronize.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.device_index

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.device_index](https://pytorch.org/docs/2.13/generated/torch.accelerator.device_index.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## Memory management

### torch.accelerator.memory.empty_cache

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.empty_cache](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.empty_cache.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.empty_host_cache

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.empty_host_cache](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.empty_host_cache.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.get_memory_info

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.get_memory_info](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.get_memory_info.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.max_memory_allocated](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.max_memory_allocated.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.max_memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.max_memory_reserved](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.max_memory_reserved.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.memory_allocated](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.memory_allocated.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.memory_reserved](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.memory_reserved.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.memory_stats](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.memory_stats.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.reset_accumulated_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.reset_accumulated_memory_stats](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.reset_accumulated_memory_stats.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.accelerator.memory.reset_peak_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.accelerator.memory.reset_peak_memory_stats](https://pytorch.org/docs/2.13/generated/torch.accelerator.memory.reset_peak_memory_stats.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>
