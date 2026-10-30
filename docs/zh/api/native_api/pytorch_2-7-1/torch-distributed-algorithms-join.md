# torch.distributed.algorithms.join

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/distributed.algorithms.join.html)。

<div style="display:none;">

## &#8203;torch.distributed.algorithms.join

</div>

<!-- npu="910b,A3" id43 -->
<!-- npu="910b,A3" id27 -->
### <code><i>class</i></code> torch.distributed.algorithms.Join
<!-- end id27 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id28 -->
**原生文档**：[torch.distributed.algorithms.Join](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Join)

**产品支持情况**：
<!-- end id28 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b,A3" id29 -->
> <font size="3">notify_join_context()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Join.notify_join_context](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Join.notify_join_context)

**产品支持情况**：
<!-- end id29 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->

<!-- npu="910b,A3" id30 -->
</div>
<!-- end id30 -->

</div>
<!-- end id43 -->

<!-- npu="910b,A3,950" id31 -->
### <code><i>class</i></code> torch.distributed.algorithms.Joinable
<!-- end id31 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id32 -->
**原生文档**：[torch.distributed.algorithms.Joinable](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable)

**产品支持情况**：
<!-- end id32 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->

<!-- npu="910b,A3" id33 -->
> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_device](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_device)

**产品支持情况**：
<!-- end id33 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->

<!-- npu="910b,A3" id34 -->
</div>
<!-- end id34 -->

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_hook](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_hook)

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

<!-- npu="910b,A3" id35 -->
> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.Joinable.join_process_group](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_process_group)

**产品支持情况**：
<!-- end id35 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->

<!-- npu="910b,A3" id36 -->
</div>
<!-- end id36 -->

</div>

<!-- npu="910b,A3" id44 -->
<!-- npu="910b,A3" id37 -->
### <code><i>class</i></code> torch.distributed.algorithms.JoinHook
<!-- end id37 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id38 -->
**原生文档**：[torch.distributed.algorithms.JoinHook](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook)

**产品支持情况**：
<!-- end id38 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->

<!-- npu="910b,A3" id39 -->
> <font size="3">main_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.JoinHook.main_hook](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.main_hook)

**产品支持情况**：
<!-- end id39 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->

<!-- npu="910b,A3" id40 -->
</div>
<!-- end id40 -->

<!-- npu="910b,A3" id41 -->
> <font size="3">post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.JoinHook.post_hook](https://pytorch.org/docs/2.7/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.post_hook)

**产品支持情况**：
<!-- end id41 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

<!-- npu="910b,A3" id42 -->
</div>
<!-- end id42 -->

</div>
<!-- end id44 -->
