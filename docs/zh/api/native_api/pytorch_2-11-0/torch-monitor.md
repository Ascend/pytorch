# torch.monitor

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/monitor.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.monitor

</div>

## API Reference

### <code><i>class</i></code> torch.monitor.Aggregation

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Aggregation](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Aggregation)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id3 -->

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Aggregation.name](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Aggregation.name)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id6 -->

</div>

</div>

### <code><i>class</i></code> torch.monitor.Stat

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id9 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.\_\_init\_\_](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat.__init__)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.add](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat.add)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.count](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat.count)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id18 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.get](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat.get)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.name](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Stat.name)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id24 -->

</div>

</div>

### <code><i>class</i></code> torch.monitor.Event

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Event)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id27 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.\_\_init\_\_](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Event.__init__)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.data](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Event.data)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.name](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Event.name)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">timestamp()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.timestamp](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.Event.timestamp)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id39 -->

</div>

</div>

### <code><i>class</i></code> torch.monitor.EventHandlerHandle

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.EventHandlerHandle](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.EventHandlerHandle)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id42 -->

</div>

### torch.monitor.log_event

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.log_event](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.log_event)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id45 -->

</div>

### torch.monitor.register_event_handler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.register_event_handler](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.register_event_handler)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id48 -->

</div>

### torch.monitor.unregister_event_handler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.unregister_event_handler](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.unregister_event_handler)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

</div>

### <code><i>class</i></code> torch.monitor.TensorboardEventHandler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.TensorboardEventHandler](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.TensorboardEventHandler)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.TensorboardEventHandler.\_\_init\_\_](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.TensorboardEventHandler.__init__)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id57 -->

</div>

</div>

### torch.monitor.data_value_t

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.data_value_t](https://pytorch.org/docs/2.11/monitor.html#torch.monitor.data_value_t)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

</div>
