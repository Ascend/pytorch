# torch.futures

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.futures.Future

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">add_done_callback()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.add_done_callback](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.add_done_callback)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">done()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.done](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.done)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">set_exception()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.set_exception](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.set_exception)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">set_result()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.set_result](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.set_result)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">then()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.then](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.then)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.value](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.value)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.wait](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.wait)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### torch.futures.collect_all

<div style="margin-left: 2em">

**原生文档**：[torch.futures.collect_all](https://pytorch.org/docs/2.13/futures.html#torch.futures.collect_all)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.futures.wait_all

<div style="margin-left: 2em">

**原生文档**：[torch.futures.wait_all](https://pytorch.org/docs/2.13/futures.html#torch.futures.wait_all)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>
