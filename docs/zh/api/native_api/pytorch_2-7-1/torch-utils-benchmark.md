# torch.utils.benchmark

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/benchmark_utils.html)。

<div style="display:none;">

## &#8203;torch.utils.benchmark

</div>

### <code><i>class</i></code> torch.utils.benchmark.Timer

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.Timer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">blocked_autorange()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.blocked_autorange](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.Timer.blocked_autorange)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">collect_callgrind()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.collect_callgrind](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.Timer.collect_callgrind)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">timeit()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.timeit](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.Timer.timeit)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.utils.benchmark.Measurement

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Measurement](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.Measurement)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.utils.benchmark.CallgrindStats

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.CallgrindStats)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">as_standardized()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.as_standardized](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.as_standardized)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">counts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.counts](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.counts)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">delta()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.delta](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.delta)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">stats()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.stats](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.stats)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.utils.benchmark.FunctionCounts

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.FunctionCounts)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">denoise()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.denoise](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.denoise)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">filter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.filter](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.filter)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.transform](https://pytorch.org/docs/2.7/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.transform)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>
