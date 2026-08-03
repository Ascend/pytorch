# torch.utils.benchmark

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.utils.benchmark.Timer

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.Timer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">blocked_autorange()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.blocked_autorange](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.Timer.blocked_autorange)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">collect_callgrind()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.collect_callgrind](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.Timer.collect_callgrind)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">timeit()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Timer.timeit](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.Timer.timeit)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### _`class`_ torch.utils.benchmark.Measurement

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.Measurement](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.Measurement)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.utils.benchmark.CallgrindStats

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.CallgrindStats)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">as_standardized()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.as_standardized](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.as_standardized)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">counts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.counts](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.counts)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">delta()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.delta](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.delta)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">stats()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.CallgrindStats.stats](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.CallgrindStats.stats)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### _`class`_ torch.utils.benchmark.FunctionCounts

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.FunctionCounts)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">denoise()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.denoise](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.denoise)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">filter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.filter](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.filter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.benchmark.FunctionCounts.transform](https://pytorch.org/docs/2.12/benchmark_utils.html#torch.utils.benchmark.FunctionCounts.transform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>
