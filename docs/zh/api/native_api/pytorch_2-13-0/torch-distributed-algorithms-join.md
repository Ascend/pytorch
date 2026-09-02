# torch.distributed.algorithms.join

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributed.algorithms.join.html)。

<div style="display:none;">

## &#8203;torch.distributed.algorithms.join

</div>

### <code><i>class</i></code> torch.distributed.algorithms.Join

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Join](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Join)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">notify_join_context()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Join.notify_join_context](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Join.notify_join_context)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.algorithms.Joinable

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_device](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_hook](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_process_group](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_process_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.algorithms.JoinHook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.JoinHook](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">main_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.JoinHook.main_hook](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.main_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.JoinHook.post_hook](https://pytorch.org/docs/2.13/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.post_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>
