# torch.utils.benchmark

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.14/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/benchmark_utils.html)。

<div style="display:none;">

## &#8203;torch.utils.benchmark

</div>

### <code><i>class</i></code> torch.utils.benchmark.Timer

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Timer)

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

<a class="method_anchor" id="torch_utils_benchmark_Timer_adaptive_autorange"></a>

> <font size="3">adaptive_autorange()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.adaptive_autorange](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Timer.adaptive_autorange)

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

<a class="method_anchor" id="torch_utils_benchmark_Timer_blocked_autorange"></a>

> <font size="3">blocked_autorange()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.blocked_autorange](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Timer.blocked_autorange)

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

<a class="method_anchor" id="torch_utils_benchmark_Timer_collect_callgrind"></a>

> <font size="3">collect_callgrind()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.collect_callgrind](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Timer.collect_callgrind)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Timer_timeit"></a>

> <font size="3">timeit()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.timeit](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Timer.timeit)

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

</div>

### <code><i>class</i></code> torch.utils.benchmark.Measurement

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Measurement](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Measurement)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

<a class="method_anchor" id="torch_utils_benchmark_Measurement_merge"></a>

> <font size="3">merge()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Measurement.merge](https://docs.pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Measurement.merge)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Measurement_significant_figures"></a>

> <font size="3">significant_figures</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Measurement.significant\_figures](https://docs.pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Measurement.significant_figures)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

</div>

### <code><i>class</i></code> torch.utils.benchmark.CallgrindStats

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.CallgrindStats)

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

<a class="method_anchor" id="torch_utils_benchmark_CallgrindStats_as_standardized"></a>

> <font size="3">as_standardized()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.as_standardized](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.as_standardized)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_CallgrindStats_counts"></a>

> <font size="3">counts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.counts](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.counts)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_CallgrindStats_delta"></a>

> <font size="3">delta()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.delta](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.delta)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_CallgrindStats_stats"></a>

> <font size="3">stats()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.stats](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.stats)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

</div>

### <code><i>class</i></code> torch.utils.benchmark.FunctionCounts

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.FunctionCounts)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

<a class="method_anchor" id="torch_utils_benchmark_FunctionCounts_denoise"></a>

> <font size="3">denoise()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.denoise](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.denoise)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_FunctionCounts_filter"></a>

> <font size="3">filter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.filter](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.filter)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_FunctionCounts_transform"></a>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.transform](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.transform)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

</div>

</div>

### <code><i>class</i></code> torch.utils.benchmark.Compare

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

<a class="method_anchor" id="torch_utils_benchmark_Compare_colorize"></a>

> <font size="3">colorize()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare.colorize](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare.colorize)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Compare_extend_results"></a>

> <font size="3">extend_results()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare.extend_results](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare.extend_results)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Compare_highlight_warnings"></a>

> <font size="3">highlight_warnings()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare.highlight_warnings](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare.highlight_warnings)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Compare_print"></a>

> <font size="3">print()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare.print](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare.print)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

</div>

<a class="method_anchor" id="torch_utils_benchmark_Compare_trim_significant_figures"></a>

> <font size="3">trim_significant_figures()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Compare.trim_significant_figures](https://pytorch.org/docs/2.14/benchmark_utils.html#torch.utils.benchmark.Compare.trim_significant_figures)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

</div>
